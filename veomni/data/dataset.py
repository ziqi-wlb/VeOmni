# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Callable, Dict, List, Literal, Optional

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import Dataset, IterableDataset

from ..distributed.parallel_state import get_parallel_state
from ..utils import logging
from ..utils.dist_utils import main_process_first


logger = logging.get_logger(__name__)


class DummyDataset(Dataset):
    def __init__(self, size: int, seq_length: int):
        """
        Args:
            size (int): Nums of datasets
            seq_length (int, optional): seq_length
        """
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        input_ids = torch.randint(low=0, high=self.vocab_size, size=(self.seq_length,))
        attention_mask = torch.ones((self.seq_length,), dtype=torch.long)
        labels = input_ids.clone()
        return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]


class MappingDataset(Dataset):
    """
    Mapping dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, "torch.Tensor"]]:
        if self._transform is not None:
            return self._transform(self._data[index])
        else:
            return self._data[index]


class IterativeDataset(IterableDataset):
    """
    Iterative dataset.
    Args:
        data (Dataset): Dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for sample in self._data:
            if self._transform is not None:
                yield self._transform(sample)
            else:
                yield sample

    def load_state_dict(self, state_dict):
        self._data.load_state_dict(state_dict["dataset"])

    def state_dict(self):
        return {"dataset": self._data.state_dict()}

    def set_epoch(self, epoch: int):
        self._data.set_epoch(epoch)


class EnergonDataset(IterativeDataset):
    """
    A specialized wrapper for Megatron-Energon datasets that provides:
    - Automatic WorkerConfig management
    - TextSample to dict conversion
    - Native state management using save_state/restore_state
    - Epoch-based state reset
    
    Args:
        data (Dataset): underlying Megatron-Energon dataset
        transform (Optional[Callable]): transform function
    """

    def __init__(self, data: "Dataset", transform: Optional[Callable] = None):
        self._data = data
        self._transform = transform

    def __len__(self):
        """Get the length of the dataset."""
        if hasattr(self._data, '__len__'):
            return len(self._data)

    def __iter__(self):
        """Iterate over the dataset with WorkerConfig management and TextSample conversion."""
        # For Megatron-Energon datasets, we need to set up the WorkerConfig properly
        if hasattr(self._data, 'worker_config'):
            try:
                from megatron.energon import WorkerConfig
                # Ensure active_worker_config is None before activation
                WorkerConfig.active_worker_config = None
                # Activate the worker config
                self._data.worker_config.worker_activate(sample_index=0)
                logger.debug("Activated WorkerConfig for Megatron-Energon dataset")
            except Exception as e:
                logger.warning(f"Failed to activate WorkerConfig: {e}")
        
        try:
            for sample in self._data:
                # Convert Megatron-Energon TextSample to dict for compatibility
                if hasattr(sample, '__dict__') and not isinstance(sample, dict):
                    # Convert TextSample or similar objects to dict
                    sample_dict = {}
                    for key, value in sample.__dict__.items():
                        if not key.startswith('_'):  # Skip private attributes
                            sample_dict[key] = value
                    
                    # Handle special case for TextSample
                    if hasattr(sample, 'text'):
                        sample_dict['text'] = sample.text
                    
                    sample = sample_dict
                
                if self._transform is not None:
                    yield self._transform(sample)
                else:
                    yield sample
        finally:
            # Clean up WorkerConfig
            if hasattr(self._data, 'worker_config'):
                try:
                    self._data.worker_config.worker_deactivate()
                    logger.debug("Deactivated WorkerConfig for Megatron-Energon dataset")
                except Exception as e:
                    logger.warning(f"Failed to deactivate WorkerConfig: {e}")

    def load_state_dict(self, state_dict):
        """Load the state of the dataset from checkpointing."""
        if hasattr(self._data, 'restore_state'):
            # Use Megatron-Energon's native restore_state method
            try:
                self._data.restore_state(state_dict["dataset"])
            except Exception as e:
                logger.warning(f"Failed to restore state using restore_state: {e}")
        elif hasattr(self._data, 'load_state_dict'):
            # Fallback to load_state_dict if available
            self._data.load_state_dict(state_dict["dataset"])
        else:
            logger.warning(f"Dataset {type(self._data).__name__} does not support state restoration")

    def state_dict(self):
        """Get the state of the dataset for checkpointing."""
        if hasattr(self._data, 'save_state'):
            # Use Megatron-Energon's native save_state method
            try:
                state = self._data.save_state()
                return {"dataset": state}
            except Exception as e:
                logger.warning(f"Failed to save state using save_state: {e}")
                return {"dataset": {}}
        elif hasattr(self._data, 'state_dict'):
            # Fallback to state_dict if available
            return {"dataset": self._data.state_dict()}
        else:
            # Return empty state dict for datasets that don't support state management
            return {"dataset": {}}

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset."""
        if hasattr(self._data, 'set_epoch'):
            self._data.set_epoch(epoch)
        elif hasattr(self._data, 'reset_state_deep'):
            # For Megatron-Energon datasets, reset state when epoch changes
            try:
                self._data.reset_state_deep()
                logger.debug(f"Reset state for epoch {epoch}")
            except Exception as e:
                logger.warning(f"Failed to reset state for epoch {epoch}: {e}")
        else:
            logger.debug(f"Dataset {type(self._data).__name__} does not support set_epoch or state reset")


def build_dummy_dataset(size: int, max_seq_len: int) -> "Dataset":
    return DummyDataset(size=size, seq_length=max_seq_len)


def build_mapping_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
) -> "Dataset":
    """
    Build mapping dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
    Returns:
        Dataset: mapping dataset
    """
    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
        elif os.path.isfile(data_path):
            data_files.append(data_files)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    with main_process_first():
        dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace)

    return MappingDataset(data=dataset, transform=transform)


def build_iterative_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    seed: int = 42,
) -> "IterableDataset":
    """ "
    Build iterative dataset.
    Args:
        data_path (str): data path
        transform (Optional[Callable]): transform function
        namespace (Literal["train", "test"]): dataset namespace
        seed (int): random seed
    Returns:
        IterableDataset: iterative dataset
    """

    data_files = []
    data_paths = data_path.split(",")
    for data_path in data_paths:
        if os.path.isdir(data_path):
            data_files.extend([os.path.join(data_path, fn) for fn in os.listdir(data_path)])
        elif os.path.isfile(data_path):
            data_files.append(data_files)
        else:
            raise FileNotFoundError(f"Dataset {data_path} not exists.")

    parallel_state = get_parallel_state()
    file_extenstion = os.path.splitext(data_files[0])[-1][1:]
    if file_extenstion not in ["parquet", "jsonl", "json", "csv", "arrow"]:
        raise ValueError(f"{file_extenstion} files are not supported.")

    file_extenstion = "json" if file_extenstion == "jsonl" else file_extenstion
    dataset = load_dataset(file_extenstion, data_files=data_files, split=namespace, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset = split_dataset_by_node(dataset, parallel_state.dp_rank, parallel_state.dp_size)

    return IterativeDataset(dataset, transform)


def build_energon_dataset(
    data_path: str,
    transform: Optional[Callable] = None,
    namespace: Literal["train", "test"] = "train",
    max_samples_per_sequence: Optional[int] = None,
    virtual_epoch_length: Optional[int] = 0,
    shuffle_buffer_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> "Dataset":
    """
    Build Megatron-Energon native dataset using the official get_train_dataset function.
    
    This is the recommended way to use Megatron-Energon datasets as it provides:
    - Automatic length calculation based on virtual_epoch_length
    - Built-in field mapping (txt -> text)
    - Professional streaming dataset support
    - Built-in error handling and performance optimizations
    
    Args:
        data_path (str): Path to the energon dataset directory
        transform (Optional[Callable]): Transform function to apply to samples
        namespace (Literal["train", "test"]): Dataset namespace (not used for energon)
        max_samples_per_sequence (Optional[int]): Maximum samples per sequence
        virtual_epoch_length (Optional[int]): Virtual epoch length for length calculation
        shuffle_buffer_size (Optional[int]): Shuffle buffer size
        num_workers (Optional[int]): Number of workers (if None, will be auto-detected)
    
    Returns:
        Dataset: Megatron-Energon native dataset
    """
    try:
        from megatron.energon import get_train_dataset, WorkerConfig
        from megatron.energon.epathlib import EPath
    except ImportError:
        raise ImportError(
            "Megatron-Energon is not installed. Please install it with: "
            "pip install megatron-energon"
        )
    
    # Get parallel state for distributed training
    parallel_state = get_parallel_state()
    
    # Auto-detect number of workers if not provided
    if num_workers is None:
        # Try to get from environment or use a reasonable default
        num_workers = int(os.environ.get('TORCH_DATA_WORKERS', '1'))

    # Create base WorkerConfig
    base_worker_config = WorkerConfig(
        rank=parallel_state.dp_rank,
        world_size=parallel_state.dp_size,
        num_workers=num_workers
    )
    
    # Wrap it with our compatible version
    worker_config = base_worker_config
    
    logger.info(f"Created WorkerConfig: rank={parallel_state.dp_rank}, world_size={parallel_state.dp_size}")

    if virtual_epoch_length is None:
        # Estimate based on data path - look for .nv-meta/info.json
        try:
            meta_path = os.path.join(data_path, ".nv-meta", "info.json")
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    info = json.load(f)
                    if 'splits' in info and 'train' in info['splits']:
                        virtual_epoch_length = info['splits']['train'].get('num_samples', 1000000)
                    else:
                        virtual_epoch_length = 0
        except Exception as e:
            logger.warning(f"Could not determine virtual_epoch_length from metadata: {e}")
        if virtual_epoch_length is None:
            virtual_epoch_length = 0  # Fallback

    logger.info(f"Building Megatron-Energon native dataset from {data_path}")
    logger.info(f"  - max_samples_per_sequence: {max_samples_per_sequence}")
    logger.info(f"  - virtual_epoch_length: {virtual_epoch_length}")
    logger.info(f"  - shuffle_buffer_size: {shuffle_buffer_size}")

    # Get the dataset using Megatron-Energon's official function
    dataset = get_train_dataset(
        path=data_path,
        split_part=namespace,
        worker_config=worker_config,
        batch_size=None,  # No batching at dataset level
        shuffle_buffer_size=shuffle_buffer_size,
        max_samples_per_sequence=max_samples_per_sequence,
        virtual_epoch_length=virtual_epoch_length,
        repeat=True,  # Always repeat for training
    )
    
    logger.info(f"Dataset type: {type(dataset)} Dataset length: {len(dataset)}")
    
    # Wrap in our EnergonDataset for Megatron-Energon specific functionality
    return EnergonDataset(dataset, transform)