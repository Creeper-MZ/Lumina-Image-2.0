#!/usr/bin/env python3
"""
Gemma3模型结构深度调查脚本
用于分析Gemma3的内部结构，找到正确的layers访问路径
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import json
import sys
from pathlib import Path

def print_separator(title, char="=", width=80):
    """打印分隔线"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def analyze_object_structure(obj, name="", max_depth=3, current_depth=0, visited=None):
    """
    递归分析对象结构
    """
    if visited is None:
        visited = set()

    # 避免循环引用
    obj_id = id(obj)
    if obj_id in visited or current_depth >= max_depth:
        return {}
    visited.add(obj_id)

    result = {
        "type": str(type(obj)),
        "class_name": obj.__class__.__name__,
        "attributes": {}
    }

    # 添加长度信息（如果是容器）
    if hasattr(obj, '__len__'):
        try:
            result["length"] = len(obj)
        except:
            pass

    # 分析属性
    for attr_name in dir(obj):
        if attr_name.startswith('_'):
            continue

        try:
            attr = getattr(obj, attr_name)

            # 跳过方法
            if callable(attr) and not isinstance(attr, nn.Module):
                continue

            attr_info = {
                "type": str(type(attr)),
                "class_name": attr.__class__.__name__
            }

            # 添加长度信息
            if hasattr(attr, '__len__'):
                try:
                    attr_info["length"] = len(attr)
                except:
                    pass

            # 如果是nn.Module，递归分析
            if isinstance(attr, nn.Module) and current_depth < max_depth - 1:
                attr_info["structure"] = analyze_object_structure(
                    attr, f"{name}.{attr_name}", max_depth, current_depth + 1, visited
                )

            result["attributes"][attr_name] = attr_info

        except Exception as e:
            result["attributes"][attr_name] = {"error": str(e)}

    return result

def find_layers_in_structure(obj, path="model", max_depth=5, current_depth=0, visited=None):
    """
    在模型结构中查找所有可能的layers
    """
    if visited is None:
        visited = set()

    obj_id = id(obj)
    if obj_id in visited or current_depth >= max_depth:
        return []
    visited.add(obj_id)

    layers_found = []

    # 检查当前对象的属性
    for attr_name in dir(obj):
        if attr_name.startswith('_'):
            continue

        try:
            attr = getattr(obj, attr_name)
            current_path = f"{path}.{attr_name}"

            # 检查是否是层列表
            if hasattr(attr, '__len__') and hasattr(attr, '__getitem__'):
                try:
                    if len(attr) > 0:
                        first_item = attr[0]
                        if isinstance(first_item, nn.Module):
                            # 检查是否像transformer层
                            item_type = str(type(first_item))
                            if any(keyword in item_type.lower() for keyword in
                                   ['layer', 'block', 'transformer', 'attention', 'decoder', 'encoder']):
                                layers_found.append({
                                    "path": current_path,
                                    "length": len(attr),
                                    "item_type": item_type,
                                    "item_class": first_item.__class__.__name__
                                })
                except:
                    pass

            # 递归搜索
            if isinstance(attr, nn.Module) and current_depth < max_depth - 1:
                layers_found.extend(find_layers_in_structure(
                    attr, current_path, max_depth, current_depth + 1, visited
                ))

        except Exception as e:
            continue

    return layers_found

def inspect_gemma3_model(model_path_or_name="google/gemma-3-4b-it"):
    """
    深度检查Gemma3模型结构
    """
    print_separator("🔍 GEMMA3 模型结构深度调查", "=", 80)

    try:
        print(f"📥 Loading model: {model_path_or_name}")

        # 尝试加载模型
        model = AutoModel.from_pretrained(
            model_path_or_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # 先用CPU避免内存问题
        )

        print(f"✅ Successfully loaded model")
        print(f"📊 Model type: {type(model)}")
        print(f"📊 Model class: {model.__class__.__name__}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Please check if the model path/name is correct")
        return None

    print_separator("🏗️ 基本模型信息", "-", 80)

    # 基本信息
    print(f"Model type: {type(model)}")
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model name: {model.__class__.__module__}")

    # 计算参数数量
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    except:
        print("Could not calculate parameter count")

    print_separator("🔍 模型属性列表", "-", 80)

    # 列出所有属性
    all_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
    module_attrs = []
    other_attrs = []

    for attr_name in all_attrs:
        try:
            attr = getattr(model, attr_name)
            if isinstance(attr, nn.Module):
                module_attrs.append(attr_name)
            elif not callable(attr):
                other_attrs.append(attr_name)
        except:
            pass

    print(f"🧩 Module attributes ({len(module_attrs)}):")
    for attr in sorted(module_attrs):
        try:
            attr_obj = getattr(model, attr)
            print(f"  - {attr}: {type(attr_obj)} ({attr_obj.__class__.__name__})")
            if hasattr(attr_obj, '__len__'):
                try:
                    print(f"    └── Length: {len(attr_obj)}")
                except:
                    pass
        except:
            print(f"  - {attr}: <error accessing>")

    print(f"\n📋 Other attributes ({len(other_attrs)}):")
    for attr in sorted(other_attrs[:20]):  # 只显示前20个
        try:
            attr_obj = getattr(model, attr)
            print(f"  - {attr}: {type(attr_obj)}")
        except:
            print(f"  - {attr}: <error accessing>")

    print_separator("🎯 查找Layers结构", "-", 80)

    # 查找所有可能的layers
    layers_candidates = find_layers_in_structure(model)

    print(f"🔍 Found {len(layers_candidates)} potential layer collections:")
    for i, candidate in enumerate(layers_candidates, 1):
        print(f"\n{i}. Path: {candidate['path']}")
        print(f"   Length: {candidate['length']}")
        print(f"   Item type: {candidate['item_type']}")
        print(f"   Item class: {candidate['item_class']}")

    # 重点检查最可能的candidates
    priority_paths = ['layers', 'model.layers', 'transformer.layers', 'decoder.layers']

    print(f"\n🎯 检查优先路径:")
    for path in priority_paths:
        try:
            # 尝试通过路径访问
            obj = model
            for part in path.split('.'):
                obj = getattr(obj, part)

            print(f"✅ {path}: Found! Length = {len(obj)}")
            if len(obj) > 0:
                print(f"   First item type: {type(obj[0])}")
                print(f"   First item class: {obj[0].__class__.__name__}")
        except AttributeError as e:
            print(f"❌ {path}: Not found ({e})")
        except Exception as e:
            print(f"⚠️  {path}: Error ({e})")

    print_separator("🧪 测试FSDP兼容性", "-", 80)

    # 测试不同的FSDP lambda函数
    test_cases = [
        ("model.layers", "getattr(model, 'layers', [])"),
        ("model.model.layers", "getattr(getattr(model, 'model', None), 'layers', [])"),
        ("Auto-detect", "None")  # 将在后面实现
    ]

    for name, code in test_cases:
        try:
            if code == "None":
                # 自动检测最佳路径
                if layers_candidates:
                    best_candidate = layers_candidates[0]  # 使用第一个找到的
                    path_parts = best_candidate['path'].split('.')[1:]  # 去掉'model'前缀
                    test_obj = model
                    for part in path_parts:
                        test_obj = getattr(test_obj, part)
                    layers_list = list(test_obj)
                    print(f"✅ {name}: Found {len(layers_list)} layers via {best_candidate['path']}")
                else:
                    print(f"❌ {name}: No candidates found")
            else:
                layers_list = eval(code)
                if layers_list:
                    print(f"✅ {name}: Found {len(layers_list)} layers")
                    # 测试lambda函数
                    lambda_fn = lambda m: m in layers_list
                    test_count = sum(1 for _ in model.modules() if lambda_fn(_))
                    print(f"   Lambda test: {test_count} modules would be wrapped")
                else:
                    print(f"❌ {name}: Empty or None")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")

    print_separator("📊 推荐的FSDP配置", "-", 80)

    # 生成推荐配置
    if layers_candidates:
        best_candidate = layers_candidates[0]
        path = best_candidate['path']

        print(f"🎯 推荐使用路径: {path}")
        print(f"📊 层数: {best_candidate['length']}")
        print(f"🔧 建议的修复代码:")
        print()
        print("```python")
        print(f"# 在 finetune.py 第193行替换为:")

        if path == "model.layers":
            print("lambda_fn = lambda m: m in list(model.layers)")
        elif path == "model.model.layers":
            print("lambda_fn = lambda m: m in list(model.model.layers)")
        else:
            # 生成动态代码
            path_parts = path.split('.')[1:]  # 去掉'model'前缀
            access_code = "model"
            for part in path_parts:
                access_code += f".{part}"
            print(f"lambda_fn = lambda m: m in list({access_code})")

        print("```")

        # 提供安全的兜底方案
        print(f"\n🛡️ 安全兜底方案:")
        print("```python")
        path_parts = path.split('.')[1:]
        nested_getattr = "model"
        for part in path_parts:
            nested_getattr = f"getattr({nested_getattr}, '{part}', [])"
        print(f"lambda_fn = lambda m: m in list({nested_getattr})")
        print("```")
    else:
        print("❌ 未找到合适的layers结构")
        print("🔧 建议使用通用兜底方案:")
        print("```python")
        print("lambda_fn = lambda m: 'Layer' in str(type(m)) or 'Block' in str(type(m))")
        print("```")

    print_separator("✅ 调查完成", "=", 80)

    return {
        "model_type": str(type(model)),
        "model_class": model.__class__.__name__,
        "layers_candidates": layers_candidates,
        "module_attributes": module_attrs,
        "recommended_path": layers_candidates[0]['path'] if layers_candidates else None
    }

def save_structure_report(result, output_file="gemma3_structure_report.json"):
    """保存结构调查报告"""
    if result:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"📄 结构报告已保存到: {output_file}")

if __name__ == "__main__":
    print("🚀 Gemma3 模型结构调查工具")
    print("=" * 80)

    # 检查命令行参数
    model_path = "google/gemma-3-4b-it"  # 默认模型

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"📍 使用指定模型: {model_path}")
    else:
        print(f"📍 使用默认模型: {model_path}")
        print("💡 你也可以指定模型路径: python script.py /path/to/model")

    try:
        # 执行调查
        result = inspect_gemma3_model(model_path)

        # 保存报告
        if result:
            save_structure_report(result)

        print("\n🎉 调查完成！请查看上面的建议来修复FSDP配置。")

    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except Exception as e:
        print(f"\n❌ 调查过程中出错: {e}")
        import traceback
        traceback.print_exc()