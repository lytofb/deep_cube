#!/bin/bash

# 检查参数个数
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <参数 (1-10)>"
  exit 1
fi

param=$1

# 检查参数是否在 1 到 10 之间
if ! [[ "$param" =~ ^[1-9]$|^10$ ]]; then
  echo "错误：参数必须为 1 到 10 的整数"
  exit 1
fi

# 根据参数计算需要复制的文件数（参数 * 10）
num_files=$(( param * 10 ))

# 根据参数计算目标目录名：rubik_100k, rubik_200k, ... rubik_1000k
dest_dir="rubik_$(( param * 100 ))k"

# 如果目标目录不存在，则创建
if [ ! -d "$dest_dir" ]; then
  mkdir -p "$dest_dir"
fi

# 从 rubik_1m_shards/ 中复制文件到目标目录
for (( i=0; i<num_files; i++ )); do
  # 格式化文件名：5位数字，如 00000, 00001, ...
  src_file=$(printf "rubik_1m_shards/part_%05d.pkl" "$i")
  if [ -f "$src_file" ]; then
    cp "$src_file" "$dest_dir/"
  else
    echo "警告：源文件 $src_file 不存在"
  fi
done

echo "已将 $num_files 个文件复制到 $dest_dir 目录。"
