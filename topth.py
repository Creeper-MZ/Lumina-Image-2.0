#!/usr/bin/env python3
"""
safetensors转pth格式脚本
支持单个文件转换和批量转换
"""

import os
import torch
import argparse
from safetensors import safe_open
from pathlib import Path


def convert_safetensors_to_pth(input_path, output_path=None):
    """
    将safetensors文件转换为pth文件

    Args:
        input_path (str): 输入的safetensors文件路径
        output_path (str): 输出的pth文件路径，如果为None则自动生成
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if not input_path.suffix == '.safetensors':
        raise ValueError(f"输入文件必须是.safetensors格式: {input_path}")

    # 如果没有指定输出路径，则自动生成
    if output_path is None:
        output_path = input_path.with_suffix('.pth')
    else:
        output_path = Path(output_path)

    print(f"正在转换: {input_path} -> {output_path}")

    # 读取safetensors文件
    state_dict = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # 保存为pth格式
    torch.save(state_dict, output_path)
    print(f"转换完成: {output_path}")


def batch_convert(input_dir, output_dir=None):
    """
    批量转换目录下的所有safetensors文件

    Args:
        input_dir (str): 包含safetensors文件的目录
        output_dir (str): 输出目录，如果为None则使用输入目录
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有safetensors文件
    safetensors_files = list(input_dir.glob("*.safetensors"))

    if not safetensors_files:
        print(f"在目录 {input_dir} 中没有找到safetensors文件")
        return

    print(f"找到 {len(safetensors_files)} 个safetensors文件")

    for file_path in safetensors_files:
        output_path = output_dir / file_path.with_suffix('.pth').name
        try:
            convert_safetensors_to_pth(file_path, output_path)
        except Exception as e:
            print(f"转换失败 {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="将safetensors文件转换为pth格式")
    parser.add_argument("input", help="输入文件路径或目录路径")
    parser.add_argument("-o", "--output", help="输出文件路径或目录路径")
    parser.add_argument("-b", "--batch", action="store_true", help="批量转换模式")

    args = parser.parse_args()

    try:
        if args.batch:
            batch_convert(args.input, args.output)
        else:
            convert_safetensors_to_pth(args.input, args.output)
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())


# 简单使用示例
def example_usage():
    """使用示例"""
    # 单个文件转换
    # convert_safetensors_to_pth("model.safetensors", "model.pth")

    # 批量转换
    # batch_convert("./models/", "./converted/")
    pass