#!/usr/bin/env python3
"""
音频处理工具 - 用于TTS服务
"""

import os
import numpy as np
import soundfile as sf
from typing import Tuple, Optional
from logger import tts_logger


class AudioProcessor:
    """音频处理器"""

    def __init__(self):
        pass

    def load_audio(
        self, file_path: str, target_sr: int = 16000
    ) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            # 使用soundfile加载音频
            audio, sr = sf.read(file_path)

            # 如果是立体声，转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # 重采样到目标采样率（如果需要）
            if sr != target_sr:
                audio = self._resample_audio(audio, sr, target_sr)
                sr = target_sr

            return audio, sr

        except Exception as e:
            tts_logger.error(f"加载音频文件失败: {e}")
            raise

    def save_audio(self, audio: np.ndarray, file_path: str, sample_rate: int = 16000):
        """保存音频文件"""
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 归一化音频数据
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95

            # 保存音频文件
            sf.write(file_path, audio, sample_rate)
            tts_logger.info(f"音频文件已保存: {file_path}")

        except Exception as e:
            tts_logger.error(f"保存音频文件失败: {e}")
            raise

    def _resample_audio(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """重采样音频"""
        try:
            import scipy.signal

            # 计算重采样比例
            ratio = target_sr / orig_sr

            # 使用scipy进行重采样
            resampled_length = int(len(audio) * ratio)
            resampled_audio = scipy.signal.resample(audio, resampled_length)

            return resampled_audio.astype(np.float32)

        except ImportError:
            tts_logger.warning("scipy not available, using simple interpolation")
            # 简单的线性插值重采样
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def normalize_audio(
        self, audio: np.ndarray, target_db: float = -20.0
    ) -> np.ndarray:
        """音频归一化"""
        try:
            # 计算RMS
            rms = np.sqrt(np.mean(audio**2))

            if rms > 0:
                # 计算目标RMS
                target_rms = 10 ** (target_db / 20)

                # 归一化
                normalized_audio = audio * (target_rms / rms)

                # 防止削波
                max_val = np.max(np.abs(normalized_audio))
                if max_val > 0.95:
                    normalized_audio = normalized_audio * (0.95 / max_val)

                return normalized_audio
            else:
                return audio

        except Exception as e:
            tts_logger.error(f"音频归一化失败: {e}")
            return audio

    def trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """移除音频首尾的静音"""
        try:
            # 找到非静音部分的开始和结束
            energy = np.abs(audio)
            above_threshold = energy > threshold

            if not np.any(above_threshold):
                return audio

            start = np.argmax(above_threshold)
            end = len(audio) - np.argmax(above_threshold[::-1])

            return audio[start:end]

        except Exception as e:
            tts_logger.error(f"移除静音失败: {e}")
            return audio

    def get_audio_info(self, file_path: str) -> dict:
        """获取音频文件信息"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")

            info = sf.info(file_path)

            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype,
                "file_size": os.path.getsize(file_path),
            }

        except Exception as e:
            tts_logger.error(f"获取音频信息失败: {e}")
            return {}
