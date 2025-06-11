#!/bin/bash

# 安装依赖
pip install --force-reinstall -r requirements.txt

# 验证环境
echo "=== 环境状态 ==="
pip list | grep -E "streamlit|scikit-learn|imbalanced-learn"

# 检查端口监听（关键步骤）
echo "=== 检查端口监听 ==="
if ! netstat -tuln | grep -q ':8502'; then
    echo "❌ 错误: 未检测到8502端口监听状态"
    exit 1
else
    echo "✅ 成功: 检测到8502端口已启动监听"
fi
