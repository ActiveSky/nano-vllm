"""加载 safetensors 权重，并支持打包参数的拆分映射。

学习重点：本实现把 HF 官方的独立矩阵（q_proj/k_proj/v_proj/gate_proj/up_proj）
合并进了并行层的大参数（qkv_proj/gate_up_proj），加载时需要通过
packed_modules_mapping 反向拆回，再交给各并行层的自定义 weight_loader
完成「切分 + 写入本卡分片」。
"""

import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """默认的权重拷贝方式，直接把张量写入参数。

    用于普通（未打包、未并行切分）的参数，如 RMSNorm 的 weight、
    RowParallelLinear 在 tp_size=1 时的完整权重等。
    """

    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str) -> None:
    """从指定目录中的 safetensors 文件加载模型权重。

    流程（学习重点）：
    1. 遍历目录下所有 *.safetensors 分片文件（大模型通常拆成多个分片）。
    2. 逐个权重名检查是否命中 packed_modules_mapping：
       - 命中：例如权重名含 "q_proj"，则映射到合并参数 "qkv_proj" 的
         "q" 区段，用该参数的 weight_loader 按 shard_id 写入对应切片
         （q/k/v 三段之一、或 gate_up 的 0/1 段）。
       - 未命中：普通参数，直接用参数自带的 weight_loader（若未自定义
         则退回默认 loader 整块拷贝）。
    3. 为什么加载到 CPU（safe_open(..., "cpu")）而不是直接上 GPU：
       先全部落在主机内存，再由各参数拷贝到自己的设备/分片，
       避免 GPU 显存同时驻留完整权重与目标参数造成峰值暴涨。
    """

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # for/else：只有 for 循环正常走完（未 break）时才执行 else，
                # 用于区分「命中打包映射」与「普通参数」两条分支。
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # k -> (合并参数名, 区段 id)，例如
                        # "q_proj" -> ("qkv_proj", "q")。
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 并行层在构造时给参数挂上了自定义 weight_loader
                        # （见 linear.py 中 weight.weight_loader = ...）。
                        weight_loader = getattr(param, "weight_loader")
                        # 打包权重会按子模块名拆分，交给自定义 loader 处理。
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, f.get_tensor(weight_name))
