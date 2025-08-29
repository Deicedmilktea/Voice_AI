import os
import threading
import numpy as np
from typing import Optional, Callable
import sounddevice as sd
from config.config import config
from utils.logger import voice_logger
from utils.audio_utils import AudioProcessor


class AudioPlayer:
    """音频播放器"""

    def __init__(self):
        self.is_playing = False
        self.audio_processor = AudioProcessor()
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.play_thread = None

        # 回调函数
        self.on_play_start = None
        self.on_play_finish = None
        self.on_play_error = None

    def play_audio_file(self, file_path: str, blocking: bool = True) -> bool:
        """播放音频文件"""
        try:
            if not os.path.exists(file_path):
                voice_logger.error(f"音频文件不存在: {file_path}")
                return False

            if self.is_playing:
                voice_logger.warning("正在播放其他音频，请稍后")
                return False

            voice_logger.info(f"开始播放音频文件: {file_path}")

            # 加载音频文件
            audio, sr = self.audio_processor.load_audio(file_path)

            # 播放音频
            return self.play_audio_array(audio, sr, blocking)

        except Exception as e:
            voice_logger.error(f"播放音频文件失败: {e}")
            if self.on_play_error:
                self.on_play_error(str(e))
            return False

    def play_audio_array(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        blocking: bool = True,
    ) -> bool:
        """播放音频数组"""
        try:
            if audio is None or len(audio) == 0:
                voice_logger.warning("音频数据为空")
                return False

            if self.is_playing:
                voice_logger.warning("正在播放其他音频，请稍后")
                return False

            sr = sample_rate or self.sample_rate

            voice_logger.info(f"开始播放音频，时长: {len(audio)/sr:.2f}秒")

            # 预处理音频
            processed_audio = self._preprocess_audio(audio, sr)

            if blocking:
                # 阻塞播放
                return self._play_blocking(processed_audio, sr)
            else:
                # 非阻塞播放
                return self._play_non_blocking(processed_audio, sr)

        except Exception as e:
            voice_logger.error(f"播放音频数组失败: {e}")
            if self.on_play_error:
                self.on_play_error(str(e))
            return False

    def _preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """预处理音频数据"""
        try:
            # 转换为单声道（如果需要）
            if len(audio.shape) > 1 and self.channels == 1:
                audio = self.audio_processor.convert_to_mono(audio)

            # 重采样（如果需要）
            if sample_rate != self.sample_rate:
                audio = self.audio_processor.resample_audio(
                    audio, sample_rate, self.sample_rate
                )

            # 归一化
            audio = self.audio_processor.normalize_audio(audio)

            # 确保音频格式正确
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            return audio

        except Exception as e:
            voice_logger.error(f"音频预处理失败: {e}")
            return audio

    def _play_blocking(self, audio: np.ndarray, sample_rate: int) -> bool:
        """阻塞式播放"""
        try:
            self.is_playing = True

            # 触发播放开始回调
            if self.on_play_start:
                self.on_play_start()

            # 播放音频
            sd.play(audio, samplerate=sample_rate)
            sd.wait()  # 等待播放完成

            self.is_playing = False

            # 触发播放完成回调
            if self.on_play_finish:
                self.on_play_finish()

            voice_logger.info("音频播放完成")
            return True

        except Exception as e:
            self.is_playing = False
            voice_logger.error(f"阻塞式播放失败: {e}")
            if self.on_play_error:
                self.on_play_error(str(e))
            return False

    def _play_non_blocking(self, audio: np.ndarray, sample_rate: int) -> bool:
        """非阻塞式播放"""
        try:
            if self.play_thread and self.play_thread.is_alive():
                voice_logger.warning("播放线程仍在运行")
                return False

            # 创建播放线程
            self.play_thread = threading.Thread(
                target=self._play_thread_func, args=(audio, sample_rate)
            )
            self.play_thread.daemon = True
            self.play_thread.start()

            return True

        except Exception as e:
            voice_logger.error(f"非阻塞式播放失败: {e}")
            if self.on_play_error:
                self.on_play_error(str(e))
            return False

    def _play_thread_func(self, audio: np.ndarray, sample_rate: int):
        """播放线程函数"""
        try:
            self.is_playing = True

            # 触发播放开始回调
            if self.on_play_start:
                self.on_play_start()

            # 播放音频
            sd.play(audio, samplerate=sample_rate)
            sd.wait()  # 等待播放完成

            self.is_playing = False

            # 触发播放完成回调
            if self.on_play_finish:
                self.on_play_finish()

            voice_logger.info("音频播放完成")

        except Exception as e:
            self.is_playing = False
            voice_logger.error(f"播放线程出错: {e}")
            if self.on_play_error:
                self.on_play_error(str(e))

    def stop_playing(self):
        """停止播放"""
        try:
            if self.is_playing:
                sd.stop()
                self.is_playing = False
                voice_logger.info("音频播放已停止")
            else:
                voice_logger.warning("当前没有音频在播放")

        except Exception as e:
            voice_logger.error(f"停止播放失败: {e}")

    def set_volume(self, volume: float):
        """设置音量（0.0-1.0）"""
        try:
            # sounddevice 没有直接的音量控制，这里只是记录日志
            # 实际音量控制需要在音频数据层面进行
            volume = max(0.0, min(1.0, volume))
            voice_logger.info(f"音量设置为: {volume:.2f}")

        except Exception as e:
            voice_logger.error(f"设置音量失败: {e}")

    def get_output_devices(self) -> list:
        """获取可用的输出设备列表"""
        try:
            devices = sd.query_devices()
            output_devices = []

            for i, device in enumerate(devices):
                if device["max_output_channels"] > 0:
                    output_devices.append(
                        {
                            "id": i,
                            "name": device["name"],
                            "channels": device["max_output_channels"],
                            "sample_rate": device["default_samplerate"],
                        }
                    )

            return output_devices

        except Exception as e:
            voice_logger.error(f"获取输出设备失败: {e}")
            return []

    def set_output_device(self, device_id: int):
        """设置输出设备"""
        try:
            sd.default.device[1] = device_id
            voice_logger.info(f"输出设备设置为: {device_id}")
        except Exception as e:
            voice_logger.error(f"设置输出设备失败: {e}")

    def test_speaker(self, duration: float = 2.0, frequency: float = 440.0) -> bool:
        """测试扬声器"""
        try:
            voice_logger.info(f"开始测试扬声器，播放{frequency}Hz正弦波{duration}秒...")

            # 生成正弦波测试音频
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)  # 振幅0.3避免过响

            # 播放测试音频
            success = self.play_audio_array(audio, blocking=True)

            if success:
                voice_logger.info("扬声器测试完成")
            else:
                voice_logger.error("扬声器测试失败")

            return success

        except Exception as e:
            voice_logger.error(f"扬声器测试失败: {e}")
            return False

    def play_beep(self, frequency: float = 800.0, duration: float = 0.2) -> bool:
        """播放提示音"""
        try:
            # 生成提示音
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            audio = 0.2 * np.sin(2 * np.pi * frequency * t)

            # 添加渐入渐出效果
            fade_samples = int(0.01 * self.sample_rate)  # 10ms渐变
            if len(audio) > 2 * fade_samples:
                # 渐入
                audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                # 渐出
                audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            return self.play_audio_array(audio, blocking=False)

        except Exception as e:
            voice_logger.error(f"播放提示音失败: {e}")
            return False

    def play_notification(self, notification_type: str = "success") -> bool:
        """播放通知音"""
        try:
            if notification_type == "success":
                # 成功音：上行音调
                frequencies = [523, 659, 784]  # C5, E5, G5
                duration = 0.15
            elif notification_type == "error":
                # 错误音：下行音调
                frequencies = [523, 440, 349]  # C5, A4, F4
                duration = 0.2
            elif notification_type == "info":
                # 信息音：单音
                frequencies = [659]  # E5
                duration = 0.1
            else:
                frequencies = [523]  # C5
                duration = 0.1

            # 生成组合音调
            audio_parts = []
            for freq in frequencies:
                t = np.linspace(0, duration, int(self.sample_rate * duration), False)
                part = 0.2 * np.sin(2 * np.pi * freq * t)
                audio_parts.append(part)

            # 连接音频片段
            audio = np.concatenate(audio_parts)

            # 添加渐入渐出
            fade_samples = int(0.01 * self.sample_rate)
            if len(audio) > 2 * fade_samples:
                audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
                audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

            return self.play_audio_array(audio, blocking=False)

        except Exception as e:
            voice_logger.error(f"播放通知音失败: {e}")
            return False

    def get_player_info(self) -> dict:
        """获取播放器信息"""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "is_playing": self.is_playing,
            "available_devices": self.get_output_devices(),
        }

    def wait_for_completion(self, timeout: Optional[float] = None):
        """等待播放完成"""
        try:
            if self.play_thread and self.play_thread.is_alive():
                self.play_thread.join(timeout=timeout)
        except Exception as e:
            voice_logger.error(f"等待播放完成失败: {e}")

    def __del__(self):
        """清理资源"""
        try:
            self.stop_playing()
        except:
            pass
