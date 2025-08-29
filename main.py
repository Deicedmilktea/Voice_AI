#!/usr/bin/env python3
"""
语音AI对话系统主程序
集成ASR、LLM、TTS功能，实现完整的语音对话流程
"""

import time
import signal
import sys
from typing import Optional
from config.config import config
from utils.logger import voice_logger
from modules import AudioRecorder, ASRProcessor, LLMProcessor, TTSProcessor, AudioPlayer


class VoiceAISystem:
    """语音AI对话系统"""

    def __init__(self):
        self.running = False

        # 初始化各个模块
        self.recorder = None
        self.asr = None
        self.llm = None
        self.tts = None
        self.player = None

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        voice_logger.info("接收到退出信号，正在关闭系统...")
        self.stop()

    def initialize(self) -> bool:
        """初始化系统"""
        try:
            voice_logger.info("正在初始化语音AI对话系统...")

            # 初始化音频录制器
            voice_logger.info("初始化音频录制器...")
            self.recorder = AudioRecorder()

            # 初始化ASR处理器
            voice_logger.info("初始化ASR处理器...")
            self.asr = ASRProcessor()

            # 初始化LLM处理器
            voice_logger.info("初始化LLM处理器...")
            self.llm = LLMProcessor()

            # 初始化TTS处理器
            voice_logger.info("初始化TTS处理器...")
            self.tts = TTSProcessor()

            # 初始化音频播放器
            voice_logger.info("初始化音频播放器...")
            self.player = AudioPlayer()

            # 测试音频设备
            if not self._test_audio_devices():
                voice_logger.warning("音频设备测试失败，但系统将继续运行")

            voice_logger.info("语音AI对话系统初始化完成")
            return True

        except Exception as e:
            voice_logger.error(f"系统初始化失败: {e}")
            return False

    def _test_audio_devices(self) -> bool:
        """测试音频设备"""
        try:
            voice_logger.info("测试音频设备...")

            # 测试麦克风
            voice_logger.info("测试麦克风...")
            mic_success = self.recorder.test_microphone(duration=1.0)

            # 测试扬声器
            voice_logger.info("测试扬声器...")
            speaker_success = self.player.test_speaker(duration=1.0, frequency=440.0)

            if mic_success and speaker_success:
                voice_logger.info("音频设备测试通过")
                return True
            else:
                voice_logger.warning(
                    f"音频设备测试部分失败 - 麦克风: {mic_success}, 扬声器: {speaker_success}"
                )
                return False

        except Exception as e:
            voice_logger.error(f"音频设备测试失败: {e}")
            return False

    def start_conversation(self):
        """开始对话"""
        try:
            if not self.running:
                voice_logger.info("启动语音AI对话系统")
                self.running = True

                # 播放启动提示音
                self.player.play_notification("success")
                time.sleep(0.5)

                # 语音提示
                welcome_text = "嬛嬛在此，姐姐有何吩咐？"
                self._speak(welcome_text)

                # 开始对话循环
                self._conversation_loop()

        except Exception as e:
            voice_logger.error(f"启动对话失败: {e}")

    def _conversation_loop(self):
        """对话主循环"""
        conversation_count = 0

        while self.running:
            try:
                conversation_count += 1
                voice_logger.info(f"=== 第 {conversation_count} 轮对话 ===")

                # 1. 录音
                voice_logger.info("等待用户说话...")
                print("\n🎤 请说话（按Ctrl+C退出）...")

                audio = self._record_audio()
                if audio is None:
                    voice_logger.warning("录音失败，跳过本轮对话")
                    continue

                # 2. 语音识别
                voice_logger.info("正在识别语音...")
                print("🔍 正在识别语音...")

                text = self._transcribe_audio(audio)
                if not text.strip():
                    voice_logger.warning("语音识别结果为空，跳过本轮对话")
                    print("❌ 没有识别到有效语音，请重新说话")
                    continue

                print(f"👤 用户说: {text}")

                # 3. 生成回复
                voice_logger.info("正在生成回复...")
                print("🤖 嬛嬛正在思考...")

                response = self._generate_response(text)
                if not response.strip():
                    voice_logger.warning("LLM生成回复为空")
                    response = "嬛嬛一时无言，请稍等片刻。"

                print(f"🎭 嬛嬛说: {response}")

                # 4. 语音合成并播放
                voice_logger.info("正在合成语音...")
                print("🎵 正在合成语音...")

                self._speak(response)

                # 短暂停顿
                time.sleep(0.5)

            except KeyboardInterrupt:
                voice_logger.info("用户中断对话")
                break
            except Exception as e:
                voice_logger.error(f"对话循环出错: {e}")
                # 播放错误提示音
                self.player.play_notification("error")
                time.sleep(1.0)

    def _record_audio(self) -> Optional[any]:
        """录制音频"""
        try:
            # 使用VAD录音，自动检测语音开始和结束
            audio = self.recorder.record_with_vad(min_duration=1.0, max_duration=30.0)

            if audio is not None and len(audio) > 0:
                # 检查是否包含语音
                if self.asr.is_speech_detected(audio):
                    voice_logger.info(
                        f"录音成功，时长: {len(audio)/config.audio.sample_rate:.2f}秒"
                    )
                    return audio
                else:
                    voice_logger.warning("录音中未检测到语音")
                    return None
            else:
                voice_logger.warning("录音数据为空")
                return None

        except Exception as e:
            voice_logger.error(f"录音失败: {e}")
            return None

    def _transcribe_audio(self, audio) -> str:
        """转录音频"""
        try:
            text = self.asr.transcribe_audio_array(audio)
            return text.strip()
        except Exception as e:
            voice_logger.error(f"语音识别失败: {e}")
            return ""

    def _generate_response(self, text: str) -> str:
        """生成回复"""
        try:
            response = self.llm.generate_response(text)
            return response.strip()
        except Exception as e:
            voice_logger.error(f"生成回复失败: {e}")
            return "嬛嬛有些不适，稍后再聊吧。"

    def _speak(self, text: str) -> bool:
        """语音合成并播放"""
        try:
            if not text.strip():
                return False

            # 预处理文本
            processed_text = self.tts.preprocess_text(text)

            # 合成语音
            audio_file = self.tts.synthesize_speech(processed_text)

            if audio_file and self.player:
                # 播放语音
                return self.player.play_audio_file(audio_file, blocking=True)
            else:
                voice_logger.error("语音合成失败")
                return False

        except Exception as e:
            voice_logger.error(f"语音播放失败: {e}")
            return False

    def stop(self):
        """停止系统"""
        try:
            if self.running:
                voice_logger.info("正在停止语音AI对话系统...")
                self.running = False

                # 停止各个模块
                if self.recorder and self.recorder.is_recording:
                    self.recorder.stop_recording()

                if self.player and self.player.is_playing:
                    self.player.stop_playing()

                # 播放结束提示音
                if self.player:
                    goodbye_text = "嬛嬛告退，姐姐保重。"
                    self._speak(goodbye_text)
                    time.sleep(0.5)
                    self.player.play_notification("info")

                voice_logger.info("语音AI对话系统已停止")

        except Exception as e:
            voice_logger.error(f"停止系统时出错: {e}")

    def get_system_info(self) -> dict:
        """获取系统信息"""
        info = {
            "running": self.running,
            "config": {
                "sample_rate": config.audio.sample_rate,
                "device": config.get_device(),
            },
        }

        if self.recorder:
            info["recorder"] = self.recorder.get_recorder_info()

        if self.asr:
            info["asr"] = self.asr.get_model_info()

        if self.llm:
            info["llm"] = self.llm.get_model_info()

        if self.tts:
            info["tts"] = self.tts.get_model_info()

        if self.player:
            info["player"] = self.player.get_player_info()

        return info

    def interactive_mode(self):
        """交互模式"""
        print("\n" + "=" * 60)
        print("🎭 语音AI对话系统 - 甄嬛模式")
        print("=" * 60)
        print("功能说明:")
        print("- 说话时系统会自动检测语音开始和结束")
        print("- 支持中文语音识别和合成")
        print("- 按 Ctrl+C 退出系统")
        print("=" * 60)

        # 显示系统信息
        info = self.get_system_info()
        print(f"📊 系统状态:")
        print(f"   - 采样率: {info['config']['sample_rate']}Hz")
        print(f"   - 设备: {info['config']['device']}")

        if "asr" in info:
            print(f"   - ASR模型: {info['asr']['model_name']}")

        if "tts" in info:
            print(f"   - TTS模型: {info['tts']['model_name']}")

        print("=" * 60)
        print()

        # 开始对话
        self.start_conversation()


def main():
    """主函数"""
    try:
        # 创建系统实例
        system = VoiceAISystem()

        # 初始化系统
        if not system.initialize():
            voice_logger.error("系统初始化失败，退出程序")
            sys.exit(1)

        # 启动交互模式
        system.interactive_mode()

    except KeyboardInterrupt:
        voice_logger.info("用户中断程序")
    except Exception as e:
        voice_logger.error(f"程序运行出错: {e}")
        sys.exit(1)
    finally:
        voice_logger.info("程序退出")


if __name__ == "__main__":
    main()
