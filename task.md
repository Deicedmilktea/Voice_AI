# 语音AI对话系统开发任务清单

## 项目概述
构建一个完整的语音AI对话系统，用户通过麦克风说话，系统经过ASR→LLM→TTS的流程进行语音对话。

## 技术栈
- **ASR**: FunASR + SenseVoice（语音转文字）
- **LLM**: 已微调的Qwen3-4B-Instruct（甄嬛风格）
- **TTS**: Coqui XTTS v2（文字转语音）
- **音频处理**: PyAudio/sounddevice（录音）

## 任务清单

### 阶段一：环境准备
- [x] 安装必要的依赖包（ASR、TTS、音频处理）
- [x] 下载SenseVoice模型
- [x] 下载Coqui XTTS v2模型
- [x] 测试音频设备（麦克风、扬声器）

### 阶段二：模块开发
- [x] 实现音频录制模块（recording.py）
- [x] 实现ASR模块（asr.py）
- [x] 实现LLM推理模块（llm.py）
- [x] 实现TTS模块（tts.py）
- [x] 实现音频播放模块（audio_player.py）

### 阶段三：系统集成
- [x] 创建主程序（main.py）
- [x] 实现对话流程控制
- [x] 添加错误处理和异常管理
- [x] 实现对话历史管理

### 阶段四：测试优化
- [x] 单元测试各个模块
- [x] 集成测试整体流程
- [x] 性能优化（延迟、内存使用）
- [x] 用户体验优化

### 阶段五：高级功能
- [ ] 添加语音活动检测（VAD）
- [ ] 实现流式处理
- [ ] 添加配置文件支持
- [ ] 创建用户界面（可选）

## 文件结构设计
```
Voice_AI/
├── main.py                 # 主程序入口
├── config/
│   ├── __init__.py
│   └── config.py          # 配置文件
├── modules/
│   ├── __init__.py
│   ├── recording.py       # 音频录制
│   ├── asr.py            # 语音识别
│   ├── llm.py            # 大语言模型
│   ├── tts.py            # 语音合成
│   └── audio_player.py   # 音频播放
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py    # 音频工具函数
│   └── logger.py         # 日志工具
├── models/               # 模型文件目录
├── temp/                 # 临时文件目录
├── logs/                 # 日志文件
├── requirements.txt      # 依赖清单
└── README.md            # 项目说明
```

## 完成状态
- [x] 项目分析和框架设计
- [x] 环境准备
- [x] 模块开发
- [x] 系统集成
- [x] 测试优化
- [ ] 高级功能

## 注意事项
1. 第一阶段使用默认TTS模型，确保基本功能正常
2. 后续可训练特定音色的TTS模型
3. 注意音频格式的统一性（采样率、通道数等）
4. 考虑实时性要求，优化处理延迟
5. 合理管理GPU内存，避免OOM
