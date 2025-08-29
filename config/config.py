import os
from dataclasses import dataclass
from typing import Dict, Any
import yaml


@dataclass
class ModelConfig:
    """模型配置"""

    # LLM配置
    base_model_path: str = "../model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"
    lora_model_path: str = "../output_max/lora_model"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # ASR配置
    asr_model: str = "iic/SenseVoiceSmall"
    asr_language: str = "auto"

    # TTS配置
    tts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_language: str = "zh"
    tts_speaker: str = "default"


@dataclass
class AudioConfig:
    """音频配置"""

    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "wav"

    # 录音配置
    record_timeout: float = 5.0  # 最大录音时长
    silence_threshold: float = 0.01  # 静音阈值
    silence_duration: float = 2.0  # 静音持续时间


@dataclass
class SystemConfig:
    """系统配置"""

    temp_dir: str = "./temp"
    log_dir: str = "./logs"
    log_level: str = "INFO"

    # 设备配置
    device: str = "auto"  # auto, cpu, cuda
    use_gpu: bool = True

    # 性能配置
    max_conversation_history: int = 10


class Config:
    """统一配置管理类"""

    def __init__(self, config_file: str = None):
        self.model = ModelConfig()
        self.audio = AudioConfig()
        self.system = SystemConfig()

        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)

        # 确保目录存在
        self._ensure_directories()

    def load_from_file(self, config_file: str):
        """从YAML文件加载配置"""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if "model" in config_data:
                for key, value in config_data["model"].items():
                    if hasattr(self.model, key):
                        setattr(self.model, key, value)

            if "audio" in config_data:
                for key, value in config_data["audio"].items():
                    if hasattr(self.audio, key):
                        setattr(self.audio, key, value)

            if "system" in config_data:
                for key, value in config_data["system"].items():
                    if hasattr(self.system, key):
                        setattr(self.system, key, value)

        except Exception as e:
            print(f"加载配置文件失败: {e}")

    def save_to_file(self, config_file: str):
        """保存配置到YAML文件"""
        config_data = {
            "model": self.model.__dict__,
            "audio": self.audio.__dict__,
            "system": self.system.__dict__,
        }

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"保存配置文件失败: {e}")

    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [self.system.temp_dir, self.system.log_dir, "models"]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_device(self):
        """获取计算设备"""
        if self.system.device == "auto":
            import torch

            return (
                "cuda" if torch.cuda.is_available() and self.system.use_gpu else "cpu"
            )
        return self.system.device


# 全局配置实例
config = Config()
