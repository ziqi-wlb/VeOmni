
<div align="center">

<img src="./assets/logo.png" width="50%">

## VeOmni: Scaling any Modality Model Training to any Accelerators with PyTorch native Training Framework

<p align="center">
  <a href="https://github.com/Byte dance-Seed/VeOmni/stargazers">
    <img src="https://img.shields.io/github/stars/ByteDance-Seed/VeOmni?style=social"></a>
  <a href="https://github.com/ByteDance-Seed/VeOmni">
    <img src="https://img.shields.io/badge/VeOmni-Project Page-yellow"></a>
  <a href="https://arxiv.org/pdf/2502.19811">
    <img src="https://img.shields.io/badge/VeOmni-Tech Report-red"></a>
  <a href="XXXX">
    <img src="https://img.shields.io/badge/VeOmni-Hugging Face-orange"></a>
  <br>
  <a href="https://github.com/user-attachments/assets/d3fcb3bf-466b-4efe-8c3f-5f85258202ae">
    <img src="https://img.shields.io/badge/VeOmni-Wechat Communication Group-07C160"></a>
  <a href="XXX">
    <img src="https://img.shields.io/badge/License-Apache--2.0-blue"></a>
</p>

</div>

## News
<br>
[2025/04/03]🔥We release VeOmni.

## Overview
VeOmni is a versatile single-modal/multi-modal pre-training/post-training framework. With VeOmni, users can easily scale any modality model to any accelerator, making it flexible and user-friendly.


## Table of Contents
- [VeOmni: Scaling any Modality Model Training to any Accelerators with PyTorch native Training Framework](#veomni-scaling-any-modality-model-training-to-any-accelerators-with-pytorch-native-training-framework)
- [News](#news)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Key Features](#key-features)
  - [Upcoming Features](#upcoming-features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Merge checkpoints](#merge-checkpoints)
  - [Build Docker](#build-docker)
- [Training Examples](#training-examples)
- [Supported Models](#supported-models)
- [Performance](#performance)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [About ByteDance Seed Team](#about-bytedance-seed-team)


## Key Features

- **Parallelism**
    - Parallel state by [DeviceMesh](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html)
    - Torch FSDP1/2
    - Experts parallelism
    - Tensor parallelism
    - Easy to add new parallelism plan
    - Sequence parallelism
      - [Ulysess](https://arxiv.org/abs/2309.14509)
      - Async ulysses
    - Activation offloading
    - Activation checkpointing
- **Kernels**
    - GroupGemm ops for moe
    - [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) integrations
- **Model**
  - Any [transformers](https://github.com/huggingface/transformers) models.
  - Multi-modal
    - Qwen2VL
    - Seed-Omni
    - ...
- **Data IO**
    - Dynamic batching strategy
    - Omnidata processing
- **Distributed Checkpointing**
  - [ByteCheckpoint](https://arxiv.org/abs/2407.20143)(Recommend)
  - Torch Distributed checkpointing
  - Dcp merge tools
- **Other tools**
  - Profiling tools
  - Easy yaml configuration and argument parsing

### Upcoming Features
- [ ] [veScale](https://github.com/volcengine/veScale/tree/main) FSDP
- [ ] torch.compile
- [ ] Torch native pipeline parallelism
- [ ] [Flux: Fine-grained Computation-communication Overlapping GPU Kernel](https://github.com/bytedance/flux/tree/main/test) integrations
- [ ] Better offloading strategy
- [ ] Ring attention
- [ ] More models support

## Getting Started
### Installation
```shell
pip3 install -e .
```

- Install veScale(optional)
```shell
git clone https://github.com/volcengine/veScale.git
pip3 install .
```

### Quick Start
Start training like this:

```shell
bash train.sh $TRAIN_SCRIPT $CONFIG.yaml
```

You can override arguments in yaml by passing arguments from an external command line
```shell
bash train.sh $TRAIN_SCRIPT $CONFIG.yaml \
    --model.model_path PATH/TO/MODEL \
    --data.train_path PATH/TO/DATA \
    --train.global_batch_size GLOBAL_BATCH_SIZE \
```

Here is an end-to-end workflow for preparing a subset of the fineweb dataset, continue training an qwen2_5 model with sequence parallel 2 for 20 steps, and than merge the global_step_10 distributed checkpoint to hf weight by ByteCheckpoint.


- Download fineweb dataset
```shell
python3 scripts/download_hf_data.py \
  --repo_id HuggingFaceFW/fineweb \
  --local_dir ./fineweb/ \
  --allow_patterns sample/10BT/*
```

- Download qwen2_5 model
```shell
python3 scripts/download_hf_model.py \
  --repo_id Qwen/Qwen2.5-7B \
  --local_dir Qwen2.5-7B
```

- Start training
```shell
bash train.sh tasks/train_torch.py configs/pretrain/qwen2_5.yaml \
    --model.model_path Qwen2.5-7B \
    --data.train_path ./fineweb/sample/10BT/ \
    --train.ulysses_parallel_size 2 \
    --train.save_steps 10 \
    --train.max_steps 20 \
    --train.output_dir Qwen2.5-7B-Instruct_CT
```

- merge checkpoints
```shell
python3 scripts/mereg_dcp_to_hf.py \
    --load-dir Qwen2.5-7B-Instruct_CT/checkpoints/global_step_10 \
    --model_assets_dir Qwen2.5-7B-Instruct_CT/model_assets \
    --save-dir Qwen2.5-7B-Instruct_CT/checkpoints/global_step_10/hf_ckpt
```

- test inference
```shell
python3 tasks/infer.py \
  --infer.model_path Qwen2.5-7B-Instruct_CT/checkpoints/global_step_10/hf_ckpt \
```


### Merge checkpoints
we use ByteCheckpoint to save checkpoints in torch.distributed.checkpoint(dcp) format. you can merge dcp file by this command:

```shell
python3 scripts/mereg_dcp_to_hf.py \
    --load-dir PATH/TO/CHECKPOINTS \
    --model_assets_dir PATH/TO/MODEL_ASSETS \
    --save-dir PATH/TO/SAVE_HF_WEIGHT \
```

for example, your output_dir is like this `seed_omni`, you want to merge global_step_100 checkpoint to hf weight.

```shell
python3 scripts/mereg_dcp_to_hf.py \
    --load-dir seed_omni/checkpoints/global_step_100 \
    --model_assets_dir seed_omni/model_assets \
    --save-dir seed_omni/hf_ckpt \
```

### Build Docker
```shell
cd docker/
docker compose up -d
docker compose exec VeOmni bash
```

## Training Examples
- PyTorch FSDP2 Qwen2VL
```shell
bash train.sh tasks/multimodal/omni/train_qwen2_vl.py configs/multimodal/qwen2_vl/qwen2_vl.yaml
```

- PyTorch FSDP2 Qwen2 CT

```shell
bash train.sh tasks/train_torch.py  configs/pretrain/qwen2_5.yaml
```

- PyTorch FSDP2 llama3-8b-instruct CT
```shell
bash train.sh  tasks/train_torch.py configs/pretrain/llama3.yaml
```

## Supported Models
| Model                                                             | Model size                       | Example config File                                       |
| ----------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------- |
| [DeepSeek 2.5/3/R1](https://huggingface.co/deepseek-ai)           | 236B/671B                        | [deepseek.yaml](configs/pretrain/deepseek.yaml)           |
| [Llama 3-3.3](https://huggingface.co/meta-llama)                  | 1B/3B/8B/70B                     | [llama3.yaml](configs/pretrain/llama3.yaml)               |
| [Qwen 2-2.5](https://huggingface.co/Qwen)                         | 0.5B/1.5B/3B/7B/14B/32B/72B/     | [qwen2_5.yaml](configs/pretrain/qwen2_5.yaml)       |
| [Qwen2-VL/Qwen2.5-VL/QVQ](https://huggingface.co/Qwen)            | 2B/3B/7B/32B/72B                 | [qwen2_vl.yaml](configs/multimodal/qwen2_vl/qwen2_vl.yaml)|
| Seed_omni                                      | any foundation model with any omni encoder&&decoder | [seed_omni.yaml](configs/multimodal/omni/seed_omni.yaml)  |


> VeOmni Support all [transformers](https://github.com/huggingface/transformers) models if you don't need sequence parallelism or experts parallelism or other parallelism and cuda kernal optimize in VeOmni. We design a [model registry mechanism](veomni/models/registry.py). When the model is registered in veomni, we will automatically load the model and optimizer in VeOmni. Otherwise, it will default to load the modeling file in transformers.

> If you want to add a new model, you can add a new model in the model registry. See in [Support costom model]() docs.

## Performance
Coming soon with tech report.

## Acknowledgement
Thanks to the following projects for their excellent work:
- [veScale](https://github.com/volcengine/veScale)
- [torchtitan](https://github.com/pytorch/torchtitan/)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [torchtune](https://github.com/pytorch/torchtune)
- [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- [ByteCheckpoint](https://arxiv.org/abs/2407.20143)


## Citation
If you find VeOmni useful for your research and applications, feel free to give us a star ⭐ or cite us using:

```bibtex
@software{VeOmni,
      title={VeOmni: Scaling any Modality Model Training to any Accelerators with PyTorch native Training Framework},
      author={Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin jia, Ziyue Huang, Zhi Zhang},
      year={2025},
      howpublished={GitHub repository},
      publisher={ByteDance Seed},
      url={https://github.com/ByteDance-Seed/VeOmni},
}
```

## About [ByteDance Seed Team](https://team.doubao.com/)

![seed logo](https://github.com/user-attachments/assets/c42e675e-497c-4508-8bb9-093ad4d1f216)

Founded in 2023, ByteDance Seed Team is dedicated to crafting the industry's most advanced AI foundation models. The team aspires to become a world-class research team and make significant contributions to the advancement of science and society.

You can get to know us better through the following channels👇
<p align="center">
  <br>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/93481cda-a7f3-47f3-b333-fe6b3da86b78">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>
</p>
