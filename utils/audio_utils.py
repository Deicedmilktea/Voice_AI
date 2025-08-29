import os
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
from config.config import config
from utils.logger import voice_logger


class AudioProcessor:
    """音频处理工具类"""

    def __init__(self):
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        try:
            audio, sr = librosa.load(
                file_path, sr=self.sample_rate, mono=(self.channels == 1)
            )
            voice_logger.info(
                f"成功加载音频文件: {file_path}, 采样率: {sr}, 长度: {len(audio)/sr:.2f}秒"
            )
            return audio, sr
        except Exception as e:
            voice_logger.error(f"加载音频文件失败: {file_path}, 错误: {e}")
            raise

    def save_audio(
        self, audio: np.ndarray, file_path: str, sample_rate: Optional[int] = None
    ) -> bool:
        """保存音频文件"""
        try:
            sr = sample_rate or self.sample_rate

            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 归一化音频数据
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # 防止音频过载
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio)) * 0.95

            sf.write(file_path, audio, sr)
            voice_logger.info(f"成功保存音频文件: {file_path}")
            return True

        except Exception as e:
            voice_logger.error(f"保存音频文件失败: {file_path}, 错误: {e}")
            return False

    def resample_audio(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """重采样音频"""
        try:
            if orig_sr == target_sr:
                return audio

            resampled_audio = librosa.resample(
                audio, orig_sr=orig_sr, target_sr=target_sr
            )
            voice_logger.info(f"音频重采样: {orig_sr}Hz -> {target_sr}Hz")
            return resampled_audio

        except Exception as e:
            voice_logger.error(f"音频重采样失败: {e}")
            raise

    def normalize_audio(
        self, audio: np.ndarray, target_level: float = -20.0
    ) -> np.ndarray:
        """音频归一化"""
        try:
            # 计算RMS
            rms = np.sqrt(np.mean(audio**2))

            if rms > 0:
                # 计算目标增益
                target_rms = 10 ** (target_level / 20)
                gain = target_rms / rms

                # 应用增益
                normalized_audio = audio * gain

                # 防止削波
                if np.max(np.abs(normalized_audio)) > 0.95:
                    normalized_audio = (
                        normalized_audio / np.max(np.abs(normalized_audio)) * 0.95
                    )

                voice_logger.debug(f"音频归一化完成，增益: {20*np.log10(gain):.2f}dB")
                return normalized_audio
            else:
                voice_logger.warning("音频信号为零，跳过归一化")
                return audio

        except Exception as e:
            voice_logger.error(f"音频归一化失败: {e}")
            return audio

    def trim_silence(
        self, audio: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        """去除首尾静音"""
        try:
            threshold = threshold or config.audio.silence_threshold

            # 使用librosa的trim函数
            trimmed_audio, _ = librosa.effects.trim(
                audio, top_db=20 * np.log10(1 / threshold) if threshold > 0 else 20
            )

            if len(trimmed_audio) < len(audio):
                voice_logger.info(
                    f"去除静音：{len(audio)} -> {len(trimmed_audio)} 采样点"
                )

            return trimmed_audio

        except Exception as e:
            voice_logger.error(f"去除静音失败: {e}")
            return audio

    def detect_speech_segments(
        self, audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512
    ) -> list:
        """检测语音段落"""
        try:
            # 计算短时能量
            energy = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]

            # 能量阈值
            threshold = np.mean(energy) * 0.1

            # 找到语音段落
            speech_frames = energy > threshold

            # 转换为时间段
            segments = []
            start_frame = None

            for i, is_speech in enumerate(speech_frames):
                if is_speech and start_frame is None:
                    start_frame = i
                elif not is_speech and start_frame is not None:
                    start_time = start_frame * hop_length / self.sample_rate
                    end_time = i * hop_length / self.sample_rate
                    segments.append((start_time, end_time))
                    start_frame = None

            # 处理最后一个段落
            if start_frame is not None:
                start_time = start_frame * hop_length / self.sample_rate
                end_time = len(audio) / self.sample_rate
                segments.append((start_time, end_time))

            voice_logger.info(f"检测到 {len(segments)} 个语音段落")
            return segments

        except Exception as e:
            voice_logger.error(f"语音段落检测失败: {e}")
            return []

    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """转换为单声道"""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            voice_logger.info("音频转换为单声道")
        return audio

    def get_audio_info(self, file_path: str) -> dict:
        """获取音频文件信息"""
        try:
            info = sf.info(file_path)
            audio_info = {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype,
            }
            voice_logger.info(f"音频信息: {audio_info}")
            return audio_info

        except Exception as e:
            voice_logger.error(f"获取音频信息失败: {file_path}, 错误: {e}")
            return {}
