import os
import torch
import numpy as np
from typing import Optional, Union
from TTS.api import TTS
from config.config import config
from utils.logger import voice_logger
from utils.audio_utils import AudioProcessor


class TTSProcessor:
    """文字转语音处理器，使用Coqui XTTS v2"""

    def __init__(self):
        self.model = None
        self.audio_processor = AudioProcessor()
        self.model_name = config.model.tts_model
        self.language = config.model.tts_language
        self.speaker = config.model.tts_speaker
        self._load_model()

    def _load_model(self):
        """加载TTS模型"""
        try:
            voice_logger.info(f"正在加载TTS模型: {self.model_name}")

            # 加载XTTS v2模型
            self.model = TTS(
                model_name=self.model_name,
                progress_bar=False,
                gpu=config.system.use_gpu and torch.cuda.is_available(),
            )

            voice_logger.info("TTS模型加载成功")

        except Exception as e:
            voice_logger.error(f"TTS模型加载失败: {e}")
            raise

    def synthesize_speech(self, text: str, output_path: Optional[str] = None) -> str:
        """合成语音"""
        try:
            if not text.strip():
                voice_logger.warning("输入文本为空，跳过语音合成")
                return ""

            voice_logger.info(f"开始合成语音，文本: {text}")

            # 如果没有指定输出路径，生成临时文件路径
            if output_path is None:
                output_path = os.path.join(config.system.temp_dir, "tts_output.wav")

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 合成语音
            self.model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=self.speaker,
                language=self.language,
                split_sentences=True,
            )

            voice_logger.info(f"语音合成完成，输出文件: {output_path}")
            return output_path

        except Exception as e:
            voice_logger.error(f"语音合成失败: {e}")
            return ""

    def synthesize_speech_with_reference(
        self, text: str, reference_audio: str, output_path: Optional[str] = None
    ) -> str:
        """使用参考音频进行语音合成（克隆音色）"""
        try:
            if not text.strip():
                voice_logger.warning("输入文本为空，跳过语音合成")
                return ""

            if not os.path.exists(reference_audio):
                voice_logger.error(f"参考音频文件不存在: {reference_audio}")
                return ""

            voice_logger.info(f"开始使用参考音频合成语音，文本: {text}")

            # 如果没有指定输出路径，生成临时文件路径
            if output_path is None:
                output_path = os.path.join(
                    config.system.temp_dir, "tts_cloned_output.wav"
                )

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 使用参考音频合成语音
            self.model.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=reference_audio,
                language=self.language,
                split_sentences=True,
            )

            voice_logger.info(f"语音克隆合成完成，输出文件: {output_path}")
            return output_path

        except Exception as e:
            voice_logger.error(f"语音克隆合成失败: {e}")
            return ""

    def synthesize_to_array(self, text: str) -> Optional[np.ndarray]:
        """合成语音并返回音频数组"""
        try:
            if not text.strip():
                return None

            voice_logger.info(f"合成语音到数组，文本: {text}")

            # 生成临时文件
            temp_file = os.path.join(config.system.temp_dir, "temp_tts.wav")

            # 合成到文件
            output_path = self.synthesize_speech(text, temp_file)
            if not output_path or not os.path.exists(output_path):
                return None

            # 加载音频文件到数组
            audio, sr = self.audio_processor.load_audio(output_path)

            # 清理临时文件
            try:
                os.remove(output_path)
            except:
                pass

            return audio

        except Exception as e:
            voice_logger.error(f"合成音频到数组失败: {e}")
            return None

    def get_available_speakers(self) -> list:
        """获取可用的说话人列表"""
        try:
            if hasattr(self.model, "speakers") and self.model.speakers:
                return list(self.model.speakers)
            elif hasattr(self.model, "list_speakers"):
                return self.model.list_speakers()
            else:
                voice_logger.warning("无法获取说话人列表")
                return []
        except Exception as e:
            voice_logger.error(f"获取说话人列表失败: {e}")
            return []

    def get_available_languages(self) -> list:
        """获取支持的语言列表"""
        try:
            if hasattr(self.model, "languages") and self.model.languages:
                return list(self.model.languages)
            elif hasattr(self.model, "list_languages"):
                return self.model.list_languages()
            else:
                voice_logger.warning("无法获取语言列表")
                return ["zh", "en"]  # 默认支持中英文
        except Exception as e:
            voice_logger.error(f"获取语言列表失败: {e}")
            return ["zh", "en"]

    def set_speaker(self, speaker: str):
        """设置说话人"""
        self.speaker = speaker
        voice_logger.info(f"TTS说话人设置为: {speaker}")

    def set_language(self, language: str):
        """设置语言"""
        self.language = language
        voice_logger.info(f"TTS语言设置为: {language}")

    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        if not text:
            return ""

        # 移除多余的空格和换行
        text = " ".join(text.split())

        # 处理标点符号，确保自然的语音节奏
        import re

        # 在句号后添加适当的停顿
        text = re.sub(r"。", "。 ", text)
        text = re.sub(r"！", "！ ", text)
        text = re.sub(r"？", "？ ", text)

        # 移除多余的空格
        text = re.sub(r"\s+", " ", text)

        # 确保文本长度合适
        if len(text) > 500:
            # 截断长文本，在句号处截断
            sentences = text.split("。")
            truncated = []
            current_length = 0

            for sentence in sentences:
                if current_length + len(sentence) > 500:
                    break
                truncated.append(sentence)
                current_length += len(sentence)

            text = "。".join(truncated)
            if text and not text.endswith("。"):
                text += "。"

        return text.strip()

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "language": self.language,
            "speaker": self.speaker,
            "available_speakers": self.get_available_speakers(),
            "available_languages": self.get_available_languages(),
            "sample_rate": config.audio.sample_rate,
            "device": (
                "cuda" if config.system.use_gpu and torch.cuda.is_available() else "cpu"
            ),
        }

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
        except:
            pass
