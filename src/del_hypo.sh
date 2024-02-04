#!/bin/bash

# 检查是否提供了目录参数
if [ "$#" -ne 1 ]; then
    echo "用法: $0 [目录]"
    exit 1
fi

# 目录赋值
DIRECTORY=$1

# 检查目录是否存在
if [ ! -d "$DIRECTORY" ]; then
    echo "错误: 目录 '$DIRECTORY' 不存在。"
    exit 1
fi

# 删除以 'hypo' 开头的文件
find "$DIRECTORY" -type f -name 'hypo*' -exec rm {} +

echo "删除完成。"
