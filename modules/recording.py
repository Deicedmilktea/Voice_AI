import os
import time
import threading
import numpy as np
from typing import Optional, Callable
import sounddevice as sd
from config.config import config
from utils.logger import voice_logger
from utils.audio_utils import AudioProcessor


class AudioRecorder:
    """音频录制器"""

    def __init__(self):
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.chunk_size = config.audio.chunk_size
        self.audio_processor = AudioProcessor()
        self.recording_thread = None
        self.stream = None

        # 静音检测参数
        self.silence_threshold = config.audio.silence_threshold
        self.silence_duration = config.audio.silence_duration
        self.record_timeout = config.audio.record_timeout

        # 回调函数
        self.on_audio_data = None
        self.on_recording_start = None
        self.on_recording_stop = None

    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """开始录音"""
        try:
            if self.is_recording:
                voice_logger.warning("录音已在进行中")
                return False

            voice_logger.info("开始录音...")

            # 重置音频数据
            self.audio_data = []
            self.is_recording = True

            # 设置回调函数
            if callback:
                self.on_audio_data = callback

            # 启动录音线程
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()

            # 触发录音开始回调
            if self.on_recording_start:
                self.on_recording_start()

            return True

        except Exception as e:
            voice_logger.error(f"开始录音失败: {e}")
            self.is_recording = False
            return False

    def stop_recording(self) -> Optional[np.ndarray]:
        """停止录音并返回音频数据"""
        try:
            if not self.is_recording:
                voice_logger.warning("当前没有在录音")
                return None

            voice_logger.info("停止录音...")
            self.is_recording = False

            # 等待录音线程结束
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)

            # 关闭音频流
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None

            # 处理录音数据
            if self.audio_data:
                audio = np.concatenate(self.audio_data, axis=0)

                # 转换为单声道
                if len(audio.shape) > 1:
                    audio = self.audio_processor.convert_to_mono(audio)

                # 去除静音
                audio = self.audio_processor.trim_silence(audio)

                # 归一化
                audio = self.audio_processor.normalize_audio(audio)

                voice_logger.info(
                    f"录音完成，时长: {len(audio)/self.sample_rate:.2f}秒"
                )

                # 触发录音停止回调
                if self.on_recording_stop:
                    self.on_recording_stop(audio)

                return audio
            else:
                voice_logger.warning("没有录制到音频数据")
                return None

        except Exception as e:
            voice_logger.error(f"停止录音失败: {e}")
            return None

    def _recording_loop(self):
        """录音主循环"""
        try:
            # 配置音频流
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            )

            # 开始录音
            self.stream.start()

            # 静音检测
            silence_start_time = None
            recording_start_time = time.time()

            while self.is_recording:
                current_time = time.time()

                # 检查录音超时
                if current_time - recording_start_time > self.record_timeout:
                    voice_logger.info(f"录音超时（{self.record_timeout}秒），自动停止")
                    break

                # 检查是否检测到静音
                if len(self.audio_data) > 0:
                    # 获取最近的音频数据进行静音检测
                    recent_audio = (
                        self.audio_data[-1] if self.audio_data else np.array([])
                    )

                    if len(recent_audio) > 0:
                        # 计算音频能量
                        energy = np.mean(recent_audio**2)

                        if energy < self.silence_threshold:
                            # 检测到静音
                            if silence_start_time is None:
                                silence_start_time = current_time
                            elif (
                                current_time - silence_start_time
                                > self.silence_duration
                            ):
                                voice_logger.info(
                                    f"检测到{self.silence_duration}秒静音，自动停止录音"
                                )
                                break
                        else:
                            # 重置静音检测
                            silence_start_time = None

                time.sleep(0.1)  # 避免CPU占用过高

        except Exception as e:
            voice_logger.error(f"录音循环出错: {e}")
        finally:
            self.is_recording = False

    def _audio_callback(self, indata, frames, time, status):
        """音频回调函数"""
        try:
            if status:
                voice_logger.warning(f"音频流状态: {status}")

            if self.is_recording and len(indata) > 0:
                # 复制音频数据
                audio_chunk = indata.copy()
                self.audio_data.append(audio_chunk)

                # 触发实时音频数据回调
                if self.on_audio_data:
                    self.on_audio_data(audio_chunk)

        except Exception as e:
            voice_logger.error(f"音频回调函数出错: {e}")

    def record_with_vad(
        self, min_duration: float = 1.0, max_duration: float = 30.0
    ) -> Optional[np.ndarray]:
        """使用语音活动检测进行录音"""
        try:
            voice_logger.info("开始VAD录音...")

            # 重置参数
            self.audio_data = []
            self.is_recording = True

            # 配置音频流
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=self.chunk_size,
            ) as stream:

                speech_detected = False
                speech_end_time = None
                recording_start_time = time.time()

                while self.is_recording:
                    current_time = time.time()

                    # 读取音频数据
                    audio_chunk, overflowed = stream.read(self.chunk_size)

                    if overflowed:
                        voice_logger.warning("音频缓冲区溢出")

                    if len(audio_chunk) > 0:
                        self.audio_data.append(audio_chunk)

                        # 检测语音活动
                        energy = np.mean(audio_chunk**2)

                        if energy > self.silence_threshold:
                            # 检测到语音
                            speech_detected = True
                            speech_end_time = None
                        else:
                            # 静音状态
                            if speech_detected and speech_end_time is None:
                                speech_end_time = current_time

                    # 检查停止条件
                    recording_duration = current_time - recording_start_time

                    # 最大录音时长
                    if recording_duration > max_duration:
                        voice_logger.info(f"达到最大录音时长{max_duration}秒，停止录音")
                        break

                    # 最小录音时长后，检测静音
                    if (
                        recording_duration > min_duration
                        and speech_detected
                        and speech_end_time
                        and current_time - speech_end_time > self.silence_duration
                    ):
                        voice_logger.info("检测到语音结束，停止录音")
                        break

            self.is_recording = False

            # 处理录音数据
            if self.audio_data:
                audio = np.concatenate(self.audio_data, axis=0)

                # 转换为单声道
                if len(audio.shape) > 1:
                    audio = self.audio_processor.convert_to_mono(audio)

                # 去除静音
                audio = self.audio_processor.trim_silence(audio)

                # 归一化
                audio = self.audio_processor.normalize_audio(audio)

                voice_logger.info(
                    f"VAD录音完成，时长: {len(audio)/self.sample_rate:.2f}秒"
                )
                return audio
            else:
                voice_logger.warning("VAD录音没有数据")
                return None

        except Exception as e:
            voice_logger.error(f"VAD录音失败: {e}")
            return None

    def save_recording(self, audio: np.ndarray, file_path: str) -> bool:
        """保存录音到文件"""
        try:
            return self.audio_processor.save_audio(audio, file_path, self.sample_rate)
        except Exception as e:
            voice_logger.error(f"保存录音失败: {e}")
            return False

    def get_input_devices(self) -> list:
        """获取可用的输入设备列表"""
        try:
            devices = sd.query_devices()
            input_devices = []

            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    input_devices.append(
                        {
                            "id": i,
                            "name": device["name"],
                            "channels": device["max_input_channels"],
                            "sample_rate": device["default_samplerate"],
                        }
                    )

            return input_devices

        except Exception as e:
            voice_logger.error(f"获取输入设备失败: {e}")
            return []

    def set_input_device(self, device_id: int):
        """设置输入设备"""
        try:
            sd.default.device[0] = device_id
            voice_logger.info(f"输入设备设置为: {device_id}")
        except Exception as e:
            voice_logger.error(f"设置输入设备失败: {e}")

    def test_microphone(self, duration: float = 3.0) -> bool:
        """测试麦克风"""
        try:
            voice_logger.info(f"开始测试麦克风，录音{duration}秒...")

            # 录制测试音频
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
            )
            sd.wait()  # 等待录音完成

            # 检查音频数据
            if len(audio) > 0:
                energy = np.mean(audio**2)
                max_amplitude = np.max(np.abs(audio))

                voice_logger.info(
                    f"麦克风测试完成 - 能量: {energy:.6f}, 最大振幅: {max_amplitude:.3f}"
                )

                # 判断麦克风是否工作正常
                if energy > 1e-6 or max_amplitude > 0.001:
                    voice_logger.info("麦克风工作正常")
                    return True
                else:
                    voice_logger.warning("麦克风可能没有声音输入")
                    return False
            else:
                voice_logger.error("麦克风测试失败，没有录制到数据")
                return False

        except Exception as e:
            voice_logger.error(f"麦克风测试失败: {e}")
            return False

    def get_recorder_info(self) -> dict:
        """获取录音器信息"""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "chunk_size": self.chunk_size,
            "silence_threshold": self.silence_threshold,
            "silence_duration": self.silence_duration,
            "record_timeout": self.record_timeout,
            "is_recording": self.is_recording,
            "available_devices": self.get_input_devices(),
        }
