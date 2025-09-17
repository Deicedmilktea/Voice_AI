# Voice AI 对话系统 - 双环境架构

一个基于微服务架构的语音 AI 对话系统，通过隔离的 conda 环境解决 XTTS 与 LLM 的依赖冲突问题，支持甄嬛风格的角色扮演对话。

## ✨ 特色功能

- 🎤 **实时语音识别**: 基于FunASR + SenseVoice，支持中文语音识别
- 🤖 **智能对话生成**: 使用已微调的Qwen3-4B-Instruct模型，具有甄嬛的说话风格
- 🎵 **高质量语音合成**: 基于Coqui XTTS v2的独立TTS服务
- 🔄 **微服务架构**: TTS服务与主程序隔离，解决依赖冲突
- 🌐 **RESTful API**: TTS服务提供标准的HTTP API接口
- 📱 **完整对话流程**: 语音输入 → 语音识别 → AI回复 → TTS API → 语音输出

## 🏗️ 新架构设计

### 双环境架构

```
┌─────────────────────────────────────────┐
│              主环境 (voiceai)            │
│  ┌─────────────┐  ┌─────────────────────┐│
│  │    ASR      │  │        LLM          ││
│  │ SenseVoice  │  │   Qwen3-4B +        ││
│  │             │  │   LoRA微调          ││
│  └─────────────┘  └─────────────────────┘│
│           │                 │             │
│           └─────────┬───────┘             │
│                     │                     │
│               ┌─────────────┐             │
│               │ TTS Client  │             │
│               │ (HTTP API)  │             │
│               └─────────────┘             │
└─────────────────────┼─────────────────────┘
                      │ HTTP请求
                      ▼
┌─────────────────────────────────────────┐
│           TTS服务环境 (tts_service)      │
│  ┌─────────────┐  ┌─────────────────────┐│
│  │ FastAPI     │  │    XTTS v2          ││
│  │ TTS Server  │  │   语音合成引擎       ││
│  │             │  │                     ││
│  └─────────────┘  └─────────────────────┘│
└─────────────────────────────────────────┘
```

### 系统流程

```
用户语音输入 → 音频录制 → 语音识别(ASR) → 大语言模型(LLM) → TTS API调用 → 语音合成 → 音频播放
```

## 📋 环境要求

- Python 3.10+
- Conda (Anaconda 或 Miniconda)
- CUDA 11.8+ (推荐，用于GPU加速)
- 麦克风和扬声器设备
- 至少16GB内存
- 推荐12GB+ 显存的GPU

## 🛠️ 安装和配置

### 1. 克隆项目

```bash
git clone https://github.com/Deicedmilktea/Voice_AI.git
cd Voice_AI
```

### 2. 自动环境设置

运行环境设置脚本，将自动创建两个隔离的conda环境：

```bash
chmod +x setup_environments.sh
./setup_environments.sh
```

这将创建：
- `voiceai`: 主环境，运行ASR+LLM
- `tts_service`: TTS服务环境，专门运行XTTS

### 3. 手动环境设置（可选）

如果自动脚本失败，可以手动创建环境：

#### 3.1 创建主环境

```bash
# 创建voiceai环境
conda create -n voiceai python=3.10 -y
conda activate voiceai

# 安装PyTorch (CPU版本)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 安装其他依赖
pip install -r requirements.txt
```

#### 3.2 创建TTS服务环境

```bash
# 创建tts_service环境
conda create -n tts_service python=3.10 -y
conda activate tts_service

# 安装PyTorch (GPU版本)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装TTS服务依赖
cd tts_service
pip install -r requirements.txt
```

### 4. 模型准备

确保以下模型文件在正确位置：

- **Qwen3基础模型**: `./models/Qwen3-4B/Qwen/Qwen3-4B-Instruct-2507`
- **LoRA微调权重**: `./models/lora_model`
- **XTTS参考音频**: `./models/xtts-v2/AI-ModelScope/XTTS-v2/samples/zh-cn-sample.wav`

## 🚀 快速开始

### 启动系统

#### 1. 启动TTS服务（终端1）

```bash
conda activate tts_service
cd tts_service
python start_tts_service.py
```

TTS服务将在 `http://127.0.0.1:8888` 启动

#### 2. 运行主程序（终端2）

```bash
conda activate voiceai
python main.py
```

### 验证安装

#### 检查TTS服务

访问 `http://127.0.0.1:8888/docs` 查看API文档

#### 测试模块

```bash
conda activate voiceai
python test_modules.py
```

## 📖 使用指南

### 基本对话流程

1. 确保TTS服务已启动
2. 运行主程序 `python main.py`
3. 系统初始化，连接TTS服务
4. 看到提示后开始说话
5. 系统自动识别并生成回复
6. 通过TTS API合成语音并播放
7. 按 Ctrl+C 退出

### 系统状态监控

- TTS服务状态: `http://127.0.0.1:8888/health`
- 主程序会显示TTS连接状态
- 日志文件记录详细运行信息

## 🗂️ 项目结构（更新）

```
Voice_AI/
├── main.py                 # 主程序入口
├── test_modules.py         # 测试程序入口
├── setup_environments.sh   # 环境自动设置脚本
├── requirements.txt        # 主环境依赖
├── config/                 # 配置模块
│   ├── __init__.py
│   └── config.py          # 系统配置
├── modules/                # 核心功能模块
│   ├── __init__.py
│   ├── recording.py       # 音频录制
│   ├── asr.py            # 语音识别
│   ├── llm.py            # 大语言模型
│   ├── tts.py            # 原TTS模块（已弃用）
│   ├── tts_client.py     # TTS API客户端
│   └── audio_player.py   # 音频播放
├── tts_service/            # TTS微服务（新增）
│   ├── app.py            # FastAPI应用
│   ├── tts_processor.py  # TTS处理器
│   ├── config.py         # TTS服务配置
│   ├── logger.py         # TTS日志
│   ├── requirements.txt  # TTS服务依赖
│   ├── start_tts_service.py # TTS服务启动脚本
│   └── utils/
│       └── audio_utils.py # 音频处理工具
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── audio_utils.py    # 音频处理工具
│   └── logger.py         # 日志工具
├── temp/                   # 临时文件目录
├── logs/                   # 日志文件目录
└── models/                 # 模型文件目录
```

## 🔧 API接口文档

### TTS服务 API

TTS服务提供以下主要接口：

#### 1. 语音合成（同步）

```http
POST /tts/synthesize
Content-Type: application/json

{
    "text": "嬛嬛今日心情甚好",
    "output_format": "wav"
}
```

#### 2. 语音合成（异步）

```http
POST /tts/synthesize_async
Content-Type: application/json

{
    "text": "嬛嬛今日心情甚好",
    "output_format": "wav"
}
```

#### 3. 获取任务状态

```http
GET /tts/status/{task_id}
```

#### 4. 健康检查

```http
GET /health
```

#### 5. 模型信息

```http
GET /tts/models/info
```

详细API文档请访问: `http://127.0.0.1:8888/docs`

## ⚙️ 配置说明

### 主环境配置 (config/config.py)

```python
# TTS客户端配置
tts_service_url = "http://127.0.0.1:8888"

# 其他配置保持不变
```

### TTS服务配置 (tts_service/config.py)

```python
# 服务配置
host = "127.0.0.1"
port = 8888
debug = False

# TTS模型配置
tts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
tts_language = "zh-cn"
reference_audio = "./models/xtts-v2/AI-ModelScope/XTTS-v2/samples/zh-cn-sample.wav"
```

## 🆕 新特性

### 相比原版的改进

1. **解决依赖冲突**: XTTS与其他组件完全隔离
2. **服务化架构**: TTS作为独立微服务运行
3. **更好的扩展性**: 可以轻松替换或升级TTS模型
4. **监控能力**: 提供健康检查和状态监控
5. **API标准化**: 遵循RESTful API设计规范

### 向后兼容性

- 主程序接口保持不变
- 原有配置文件继续有效
- 支持回退到原始TTS模块（如需要）

## 🔮 未来规划

- [ ] 支持多TTS模型切换
- [ ] 添加语音情感控制
- [ ] 实现TTS服务集群
- [ ] 添加音频流式传输
- [ ] 支持实时语音转换

### 开发环境设置

1. Fork项目
2. 运行环境设置脚本
3. 在`voiceai`环境中开发主功能
4. 在`tts_service`环境中开发TTS相关功能

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 语音识别框架
- [Qwen](https://github.com/QwenLM/Qwen) - 大语言模型
- [Coqui TTS](https://github.com/coqui-ai/TTS) - 语音合成框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Web框架

💡 **重要提示**: 
- 首次运行需要先启动TTS服务，再运行主程序
- 确保两个环境的依赖正确安装，避免版本冲突
