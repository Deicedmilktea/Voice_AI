import os
import numpy as np
from typing import Optional, Union
from funasr import AutoModel
from config.config import config
from utils.logger import voice_logger
from utils.audio_utils import AudioProcessor


class ASRProcessor:
    """语音识别处理器，使用FunASR + SenseVoice"""

    def __init__(self):
        self.model = None
        self.audio_processor = AudioProcessor()
        self.model_name = config.model.asr_model
        self.language = config.model.asr_language
        self._load_model()

    def _load_model(self):
        """加载ASR模型"""
        try:
            voice_logger.info(f"正在加载ASR模型: {self.model_name}")

            # 加载SenseVoice模型
            self.model = AutoModel(
                model=self.model_name,
                trust_remote_code=True,
                remote_code_revision="main",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=config.get_device(),
                disable_update=True,
            )

            voice_logger.info("ASR模型加载成功")

        except Exception as e:
            voice_logger.error(f"ASR模型加载失败: {e}")
            raise

    def transcribe_audio_file(self, audio_file: str) -> str:
        """转录音频文件"""
        try:
            if not os.path.exists(audio_file):
                raise FileNotFoundError(f"音频文件不存在: {audio_file}")

            voice_logger.info(f"开始转录音频文件: {audio_file}")

            # 使用模型进行转录
            result = self.model.generate(
                input=audio_file,
                cache={},
                language=self.language,
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )

            # 提取转录文本
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "text" in result[0]:
                    text = result[0]["text"]
                else:
                    text = str(result[0])
            else:
                text = str(result)

            # 清理文本
            text = self._clean_text(text)

            voice_logger.info(f"转录完成，结果: {text}")
            return text

        except Exception as e:
            voice_logger.error(f"音频转录失败: {e}")
            return ""

    def transcribe_audio_array(self, audio: np.ndarray, sample_rate: int = None) -> str:
        """转录音频数组"""
        try:
            # 保存为临时文件
            temp_file = os.path.join(config.system.temp_dir, "temp_audio.wav")
            sr = sample_rate or config.audio.sample_rate

            # 预处理音频
            processed_audio = self._preprocess_audio(audio, sr)

            # 保存音频
            success = self.audio_processor.save_audio(processed_audio, temp_file, sr)
            if not success:
                raise RuntimeError("临时音频文件保存失败")

            # 转录音频
            text = self.transcribe_audio_file(temp_file)

            # 清理临时文件
            try:
                os.remove(temp_file)
            except:
                pass

            return text

        except Exception as e:
            voice_logger.error(f"音频数组转录失败: {e}")
            return ""

    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """预处理音频数据"""
        try:
            # 转换为单声道
            audio = self.audio_processor.convert_to_mono(audio)

            # 重采样到模型要求的采样率
            if sample_rate != config.audio.sample_rate:
                audio = self.audio_processor.resample_audio(
                    audio, sample_rate, config.audio.sample_rate
                )

            # 去除静音
            audio = self.audio_processor.trim_silence(audio)

            # 归一化
            audio = self.audio_processor.normalize_audio(audio)

            return audio

        except Exception as e:
            voice_logger.error(f"音频预处理失败: {e}")
            return audio

    def _clean_text(self, text: str) -> str:
        """清理转录文本"""
        if not text or not isinstance(text, str):
            return ""

        # 移除多余的空格和换行
        text = " ".join(text.split())

        # 移除可能的时间戳标记
        import re

        text = re.sub(r"\[\d+:\d+:\d+\.\d+\]", "", text)
        text = re.sub(r"<\|.*?\|>", "", text)

        # 移除多余的标点符号
        text = re.sub(r'[，。！？；：""' "（）【】《》]", "", text)

        return text.strip()

    def is_speech_detected(self, audio: np.ndarray, threshold: float = 0.002) -> bool:
        """检测音频中是否包含语音"""
        try:
            # 计算音频能量
            energy = np.mean(audio**2)

            # 检测语音段落
            segments = self.audio_processor.detect_speech_segments(audio)

            has_speech = energy > threshold and len(segments) > 0
            voice_logger.debug(
                f"语音检测结果: {has_speech}, 能量: {energy:.6f}, 段落数: {len(segments)}"
            )

            return has_speech

        except Exception as e:
            voice_logger.error(f"语音检测失败: {e}")
            return False

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "language": self.language,
            "device": config.get_device(),
            "sample_rate": config.audio.sample_rate,
        }

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
        except:
            pass
