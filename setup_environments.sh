#!/bin/bash
"""
环境设置脚本 - 创建两个隔离的conda环境
- voiceai: 运行ASR+LLM的主环境
- tts_service: 专门运行TTS服务的环境
"""

set -e  # 遇到错误时退出

echo "🔧 开始设置语音AI系统的双环境架构..."
echo "=" * 60

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到conda，请先安装Anaconda或Miniconda"
    exit 1
fi

echo "📋 当前conda环境列表:"
conda env list

echo ""
echo "🏗️  开始创建环境..."

# 1. 创建主环境 (voiceai)
echo "1️⃣  创建主环境 (voiceai) - 用于ASR+LLM..."
if conda env list | grep -q "^voiceai "; then
    echo "⚠️  voiceai环境已存在，是否重新创建？[y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有voiceai环境..."
        conda env remove -n voiceai -y
    else
        echo "📦 跳过voiceai环境创建"
        goto_tts=true
    fi
fi

if [[ "$goto_tts" != true ]]; then
    echo "📦 创建voiceai环境 (Python 3.10)..."
    conda create -n voiceai python=3.10 -y
    
    echo "📥 激活voiceai环境并安装依赖..."
    conda activate voiceai
    
    # 安装PyTorch (CPU版本，避免与TTS环境冲突)
    echo "🔥 安装PyTorch..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    
    # 安装其他依赖
    echo "📦 安装其他依赖包..."
    pip install transformers>=4.30.0
    pip install peft>=0.4.0
    pip install accelerate>=0.20.0
    pip install bitsandbytes>=0.39.0
    pip install funasr>=1.0.0
    pip install modelscope>=1.8.0
    pip install onnxruntime>=1.15.0
    pip install pyaudio>=0.2.11
    pip install sounddevice>=0.4.6
    pip install scipy>=1.10.0
    pip install numpy>=1.24.0
    pip install pyyaml>=6.0
    pip install tqdm>=4.65.0
    pip install loguru>=0.7.0
    pip install requests>=2.25.0
    
    echo "✅ voiceai环境创建完成"
    conda deactivate
fi

echo ""

# 2. 创建TTS服务环境
echo "2️⃣  创建TTS服务环境 (tts_service) - 专门用于XTTS..."
if conda env list | grep -q "^tts_service "; then
    echo "⚠️  tts_service环境已存在，是否重新创建？[y/N]"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有tts_service环境..."
        conda env remove -n tts_service -y
    else
        echo "📦 跳过tts_service环境创建"
        goto_summary=true
    fi
fi

if [[ "$goto_summary" != true ]]; then
    echo "📦 创建tts_service环境 (Python 3.10)..."
    conda create -n tts_service python=3.10 -y
    
    echo "📥 激活tts_service环境并安装依赖..."
    conda activate tts_service
    
    # 安装PyTorch (GPU版本用于TTS)
    echo "🔥 安装PyTorch (GPU版本)..."
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    
    # 安装TTS相关依赖
    echo "🎵 安装TTS相关依赖..."
    pip install TTS>=0.15.0
    pip install transformers>=4.30.0
    pip install fastapi>=0.104.0
    pip install uvicorn>=0.24.0
    pip install pydantic>=2.5.0
    pip install soundfile>=0.12.1
    pip install numpy>=1.24.0
    pip install scipy>=1.10.0
    pip install loguru>=0.7.0
    
    echo "✅ tts_service环境创建完成"
    conda deactivate
fi

echo ""
echo "🎉 环境设置完成！"
echo "=" * 60
echo "📋 使用说明:"
echo ""
echo "🔥 启动TTS服务 (在tts_service环境中):"
echo "   conda activate tts_service"
echo "   cd tts_service"
echo "   python start_tts_service.py"
echo ""
echo "🎤 运行主程序 (在voiceai环境中):"
echo "   conda activate voiceai" 
echo "   python main.py"
echo ""
echo "🌐 TTS服务将在 http://127.0.0.1:8888 提供API"
echo "📖 API文档地址: http://127.0.0.1:8888/docs"
echo ""
echo "⚠️  注意: 请先启动TTS服务，再运行主程序"
echo "=" * 60
