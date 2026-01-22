import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str, prefix: str = ""):
    """
    从safetensors文件加载模型权重

    Args:
        model: 目标模型
        path: 模型路径
        prefix: 权重名称前缀 (例如 "llm." 表示只加载以llm.开头的权重)
    """
    assert os.path.isdir(path)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 如果指定了prefix，只加载匹配的权重
                if prefix:
                    if not weight_name.startswith(prefix):
                        continue
                    # 移除prefix以匹配模型参数名
                    param_name_in_model = weight_name[len(prefix):]
                else:
                    param_name_in_model = weight_name

                # 对所有包含packed_modules_mapping中键的权重应用映射
                # 支持 language_model.model.layers... 和 model.layers... 两种格式
                applied_mapping = False

                # 检查是否需要应用packed_modules_mapping
                for k in packed_modules_mapping:
                    if k in param_name_in_model:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = param_name_in_model.replace(k, v)
                        try:
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                            applied_mapping = True
                        except AttributeError:
                            # 如果参数不存在，跳过映射尝试
                            pass
                        break

                if not applied_mapping:
                    try:
                        param = model.get_parameter(param_name_in_model)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, f.get_tensor(weight_name))
                    except AttributeError as e:
                        # 对于不匹配的权重，打印警告但继续
                        if not prefix:  # 只在非prefix模式下打印警告
                            print(f"Warning: Could not load weight {weight_name}: {e}")
