# 语音AI对话系统 - 甄嬛模式

一个基于FunASR、微调Qwen3和Coqui XTTS v2的完整语音AI对话系统，支持语音输入、AI生成回复和语音输出，具有甄嬛的说话风格。

## ✨ 特色功能

- 🎤 **实时语音识别**: 基于FunASR + SenseVoice，支持中文语音识别
- 🤖 **智能对话生成**: 使用已微调的Qwen3-4B-Instruct模型，具有甄嬛的说话风格
- 🎵 **高质量语音合成**: 基于Coqui XTTS v2，支持中文语音合成
- 🔍 **智能语音检测**: 自动检测语音开始和结束，无需手动控制
- 📱 **完整对话流程**: 语音输入 → 语音识别 → AI回复 → 语音输出
- 🎛️ **模块化设计**: 各功能模块独立，便于扩展和维护

## 🏗️ 系统架构

```
用户语音输入 → 音频录制 → 语音识别(ASR) → 大语言模型(LLM) → 语音合成(TTS) → 音频播放
```

### 核心模块

1. **AudioRecorder**: 音频录制模块，支持VAD（语音活动检测）
2. **ASRProcessor**: 语音识别模块，基于FunASR + SenseVoice
3. **LLMProcessor**: 大语言模型模块，使用微调的Qwen3
4. **TTSProcessor**: 语音合成模块，基于Coqui XTTS v2
5. **AudioPlayer**: 音频播放模块

## 📋 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐，用于GPU加速)
- 麦克风和扬声器设备
- 至少8GB内存
- 推荐16GB显存的GPU

## 🛠️ 安装和配置

### 1. 克隆项目

```bash
git clone <repository_url>
cd Voice_AI
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 模型准备

确保以下模型文件在正确位置：

- **Qwen3基础模型**: `../model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507`
- **LoRA微调权重**: `../output_max/lora_model`
- **SenseVoice模型**: 通过modelscope自动下载
- **XTTS v2模型**: 通过TTS库自动下载

### 4. 配置文件

系统使用默认配置，如需自定义，可以修改 `config/config.py` 中的参数：

```python
# 音频配置
sample_rate: int = 16000
channels: int = 1
chunk_size: int = 1024

# LLM配置
max_new_tokens: int = 512
temperature: float = 0.7
top_p: float = 0.9

# 设备配置
device: str = "auto"  # auto, cpu, cuda
use_gpu: bool = True
```

## 🚀 快速开始

### 运行主程序

```bash
python main.py
```

### 运行测试

```bash
# 测试所有模块
python test_modules.py

# 测试单个模块（例如：仅测试LLM）
python -c "from test_modules import test_llm; test_llm()"
```

## 📖 使用指南

### 基本对话流程

1. 运行 `python main.py`
2. 系统初始化完成后，会播放欢迎语音
3. 看到 "🎤 请说话（按Ctrl+C退出）..." 提示时开始说话
4. 系统自动检测语音结束，进行识别和处理
5. AI生成回复并通过语音播放
6. 重复步骤3-5进行连续对话
7. 按 Ctrl+C 退出系统

### 系统提示

- 🎤 录音中
- 🔍 语音识别中
- 🤖 AI思考中
- 🎵 语音合成中
- 🔊 播放回复

### 示例对话

```
👤 用户: 嬛嬛，你今天心情如何？
🎭 嬛嬛: 今日心情倒是不错，如春日暖阳般舒畅。

👤 用户: 给我讲个故事吧
🎭 嬛嬛: 从前有个书生，常在月下吟诗作对，倒也风雅。
```

## 🧪 测试功能

### 音频设备测试

```bash
python -c "from test_modules import test_audio_devices; test_audio_devices()"
```

### 端到端测试

```bash
python -c "from test_modules import test_end_to_end; test_end_to_end()"
```

## 🗂️ 项目结构

```
Voice_AI/
├── main.py                 # 主程序入口
├── test_modules.py         # 模块测试脚本
├── task.md                 # 开发任务清单
├── requirements.txt        # Python依赖
├── config/                 # 配置模块
│   ├── __init__.py
│   └── config.py          # 系统配置
├── modules/                # 核心功能模块
│   ├── __init__.py
│   ├── recording.py       # 音频录制
│   ├── asr.py            # 语音识别
│   ├── llm.py            # 大语言模型
│   ├── tts.py            # 语音合成
│   └── audio_player.py   # 音频播放
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── audio_utils.py    # 音频处理工具
│   └── logger.py         # 日志工具
├── temp/                   # 临时文件目录
├── logs/                   # 日志文件目录
└── models/                 # 模型文件目录
```

## ⚙️ 配置说明

### 音频配置

- `sample_rate`: 音频采样率，默认16000Hz
- `channels`: 音频通道数，默认单声道
- `silence_threshold`: 静音检测阈值
- `silence_duration`: 静音持续时间（秒）
- `record_timeout`: 最大录音时长（秒）

### 模型配置

- `base_model_path`: Qwen3基础模型路径
- `lora_model_path`: LoRA微调权重路径
- `asr_model`: ASR模型名称
- `tts_model`: TTS模型名称

### 系统配置

- `device`: 计算设备（auto/cpu/cuda）
- `use_gpu`: 是否使用GPU加速
- `max_conversation_history`: 最大对话历史记录数

## 🔧 故障排除

### 常见问题

1. **麦克风无法录音**
   - 检查麦克风权限
   - 确认音频设备连接
   - 运行音频设备测试

2. **模型加载失败**
   - 检查模型文件路径
   - 确认模型文件完整性
   - 检查GPU显存是否足够

3. **语音识别失败**
   - 确保环境安静
   - 检查麦克风质量
   - 调整录音音量

4. **语音合成失败**
   - 检查网络连接（首次下载模型）
   - 确认TTS模型下载完整
   - 检查临时文件目录权限

### 日志查看

系统日志保存在 `logs/` 目录下，文件名格式为 `voice_ai_YYYYMMDD.log`

```bash
# 查看最新日志
tail -f logs/voice_ai_$(date +%Y%m%d).log
```

## 🎯 性能优化

### GPU加速

确保安装CUDA版本的PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 内存优化

- 使用4bit量化减少显存占用
- 限制对话历史长度
- 定期清理临时文件

### 延迟优化

- 预加载所有模型
- 使用非阻塞音频播放
- 优化VAD参数

## 🔮 扩展功能

### 自定义音色

1. 准备目标音色的音频样本（10-30秒）
2. 使用 `synthesize_speech_with_reference` 方法
3. 替换默认TTS模型

### 添加新的对话风格

1. 准备新的训练数据
2. 微调Qwen3模型
3. 更新模型路径配置

### 多语言支持

1. 配置多语言ASR模型
2. 训练多语言LLM
3. 添加多语言TTS支持

## 📚 API文档

### VoiceAISystem类

主要的系统控制类

```python
system = VoiceAISystem()
system.initialize()           # 初始化系统
system.start_conversation()   # 开始对话
system.stop()                # 停止系统
system.get_system_info()     # 获取系统信息
```

### 各模块独立使用

```python
# 录音
recorder = AudioRecorder()
audio = recorder.record_with_vad()

# 语音识别
asr = ASRProcessor()
text = asr.transcribe_audio_array(audio)

# LLM对话
llm = LLMProcessor()
response = llm.generate_response(text)

# 语音合成
tts = TTSProcessor()
audio_file = tts.synthesize_speech(response)

# 音频播放
player = AudioPlayer()
player.play_audio_file(audio_file)
```

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 语音识别框架
- [Qwen](https://github.com/QwenLM/Qwen) - 大语言模型
- [Coqui TTS](https://github.com/coqui-ai/TTS) - 语音合成框架
- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) - 语音识别模型

## 📞 联系方式

如有问题或建议，欢迎联系：

- 提交Issue: [GitHub Issues](link-to-issues)
- 邮箱: your-email@example.com

---

💡 **提示**: 首次运行时模型下载可能需要较长时间，请耐心等待。
