#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Generate data.json from images and their corresponding text files')
    parser.add_argument('--data_path', required=True, help='Path to directory containing images and text files')
    parser.add_argument('--json_path', required=True, help='Output path for the data.json file')
    
    args = parser.parse_args()
    
    # 验证输入目录是否存在
    data_path = Path(args.data_path)
    if not data_path.exists() or not data_path.is_dir():
        print(f"错误: 目录 '{args.data_path}' 不存在或不是一个目录")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # 存储结果的列表
    data_list = []
    
    # 遍历目录中的所有文件
    for file_path in data_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # 获取文件名（不含扩展名）
            file_stem = file_path.stem
            
            # 查找对应的txt文件
            txt_file = data_path / f"{file_stem}.txt"
            
            if txt_file.exists():
                try:
                    # 读取prompt文本
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    
                    # 创建数据条目
                    data_entry = {
                        "path": str(file_path),
                        "prompt": prompt
                    }
                    
                    data_list.append(data_entry)
                    print(f"已处理: {file_path.name} -> {txt_file.name}")
                    
                except Exception as e:
                    print(f"警告: 读取文件 {txt_file} 时出错: {e}")
            else:
                print(f"警告: 找不到对应的文本文件 {txt_file.name} for {file_path.name}")
    
    # 检查是否找到了数据
    if not data_list:
        print("警告: 没有找到任何有效的图片-文本对")
        return
    
    # 确保输出目录存在
    json_path = Path(args.json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存JSON文件
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"\n成功生成 {len(data_list)} 条数据到 '{json_path}'")
        
    except Exception as e:
        print(f"错误: 保存JSON文件时出错: {e}")

if __name__ == "__main__":
    main()
