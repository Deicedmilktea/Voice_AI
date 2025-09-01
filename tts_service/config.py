#!/usr/bin/env python3
"""
TTS服务配置文件
"""

import os
from dataclasses import dataclass


@dataclass
class TTSConfig:
    """TTS服务配置"""

    # 服务配置
    host: str = "127.0.0.1"
    port: int = 8888
    debug: bool = False

    # TTS模型配置
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_language: str = "zh-cn"
    tts_speaker: str = "default"

    # 音频配置
    sample_rate: int = 16000

    # 文件路径配置
    temp_dir: str = "../temp"
    output_dir: str = "../temp"
    # reference_audio: str = (
    #     "../models/xtts-v2/AI-ModelScope/XTTS-v2/samples/zh-cn-sample.wav"
    # )

    # 系统配置
    use_gpu: bool = True

    def __post_init__(self):
        """初始化后处理"""
        # 确保目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
