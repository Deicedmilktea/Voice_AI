#!/usr/bin/env python3
"""
模块测试脚本
用于测试各个模块的基本功能
"""

import os
import time
import numpy as np
from config.config import config
from utils.logger import voice_logger
from modules import AudioRecorder, ASRProcessor, LLMProcessor, TTSProcessor, AudioPlayer


def test_audio_devices():
    """测试音频设备"""
    print("\n" + "=" * 50)
    print("测试音频设备")
    print("=" * 50)

    try:
        # 测试录音器
        recorder = AudioRecorder()
        print("📱 可用的输入设备:")
        input_devices = recorder.get_input_devices()
        for device in input_devices:
            print(
                f"   {device['id']}: {device['name']} ({device['channels']}ch, {device['sample_rate']}Hz)"
            )

        # 测试播放器
        player = AudioPlayer()
        print("\n🔊 可用的输出设备:")
        output_devices = player.get_output_devices()
        for device in output_devices:
            print(
                f"   {device['id']}: {device['name']} ({device['channels']}ch, {device['sample_rate']}Hz)"
            )

        # 测试麦克风
        print("\n🎤 测试麦克风...")
        mic_success = recorder.test_microphone(duration=2.0)
        print(f"麦克风测试结果: {'✅ 通过' if mic_success else '❌ 失败'}")

        # 测试扬声器
        print("\n🔊 测试扬声器...")
        speaker_success = player.test_speaker(duration=2.0, frequency=440.0)
        print(f"扬声器测试结果: {'✅ 通过' if speaker_success else '❌ 失败'}")

        return mic_success and speaker_success

    except Exception as e:
        print(f"❌ 音频设备测试失败: {e}")
        return False


def test_recording():
    """测试录音功能"""
    print("\n" + "=" * 50)
    print("测试录音功能")
    print("=" * 50)

    try:
        recorder = AudioRecorder()

        print("🎤 开始录音测试，请说话（3秒）...")
        audio = recorder.record_with_vad(min_duration=1.0, max_duration=3.0)

        if audio is not None:
            duration = len(audio) / config.audio.sample_rate
            print(f"✅ 录音成功，时长: {duration:.2f}秒")

            # 保存录音文件
            test_file = os.path.join(config.system.temp_dir, "test_recording.wav")
            success = recorder.save_recording(audio, test_file)
            if success:
                print(f"✅ 录音已保存到: {test_file}")
                return True, test_file
            else:
                print("❌ 保存录音失败")
                return False, None
        else:
            print("❌ 录音失败")
            return False, None

    except Exception as e:
        print(f"❌ 录音测试失败: {e}")
        return False, None


def test_asr():
    """测试语音识别"""
    print("\n" + "=" * 50)
    print("测试语音识别功能")
    print("=" * 50)

    try:
        # 先进行录音测试
        success, audio_file = test_recording()
        if not success or not audio_file:
            print("❌ 无法进行ASR测试，录音失败")
            return False

        # 初始化ASR
        print("🔄 初始化ASR模型...")
        asr = ASRProcessor()

        # 进行语音识别
        print("🔍 正在识别语音...")
        text = asr.transcribe_audio_file(audio_file)

        if text.strip():
            print(f"✅ 识别结果: {text}")
            return True
        else:
            print("❌ 识别结果为空")
            return False

    except Exception as e:
        print(f"❌ ASR测试失败: {e}")
        return False


def test_llm():
    """测试大语言模型"""
    print("\n" + "=" * 50)
    print("测试大语言模型功能")
    print("=" * 50)

    try:
        # 初始化LLM
        print("🔄 初始化LLM模型...")
        llm = LLMProcessor()

        # 测试问题
        test_questions = [
            # "嬛嬛，你今天心情如何？",
            # "嬛嬛，给我讲个故事吧",
            # "你觉得今天天气怎么样？",
            "嬛嬛，今日运气不佳，我们改日再聚吧！"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 测试问题 {i}: {question}")
            response = llm.generate_response(question)
            print(f"🎭 嬛嬛回复: {response}")
            time.sleep(1)  # 避免请求过快

        return True

    except Exception as e:
        print(f"❌ LLM测试失败: {e}")
        return False


def test_tts():
    """测试语音合成"""
    print("\n" + "=" * 50)
    print("测试语音合成功能")
    print("=" * 50)

    try:
        # 初始化TTS
        print("🔄 初始化TTS模型...")
        tts = TTSProcessor()

        # 测试文本
        test_texts = [
            "嬛嬛在此，姐姐有何吩咐？",
            "今日天气甚好，正是踏青的好时节。",
            "谢谢姐姐的关心，嬛嬛心中甚是感激。",
        ]

        player = AudioPlayer()

        for i, text in enumerate(test_texts, 1):
            print(f"\n🎵 合成文本 {i}: {text}")

            # 合成语音
            audio_file = tts.synthesize_speech(text)

            if audio_file and os.path.exists(audio_file):
                print(f"✅ 语音合成成功")

                # 播放语音
                print("🔊 播放语音...")
                success = player.play_audio_file(audio_file, blocking=True)
                print(f"播放结果: {'✅ 成功' if success else '❌ 失败'}")
            else:
                print("❌ 语音合成失败")

            time.sleep(0.5)

        return True

    except Exception as e:
        print(f"❌ TTS测试失败: {e}")
        return False


def test_end_to_end():
    """端到端测试"""
    print("\n" + "=" * 50)
    print("端到端测试")
    print("=" * 50)

    try:
        print("🔄 初始化所有模块...")

        # 初始化所有模块
        recorder = AudioRecorder()
        asr = ASRProcessor()
        llm = LLMProcessor()
        tts = TTSProcessor()
        player = AudioPlayer()

        print("✅ 所有模块初始化完成")

        # 进行一轮完整的对话测试
        print("\n🎤 请说话，系统将进行完整的对话流程测试...")

        # 1. 录音
        audio = recorder.record_with_vad(min_duration=1.0, max_duration=10.0)
        if audio is None:
            print("❌ 录音失败")
            return False

        print("✅ 录音完成")

        # 2. 语音识别
        print("🔍 正在识别语音...")
        text = asr.transcribe_audio_array(audio)
        if not text.strip():
            print("❌ 语音识别失败")
            return False

        print(f"✅ 识别结果: {text}")

        # 3. 生成回复
        print("🤖 正在生成回复...")
        response = llm.generate_response(text)
        if not response.strip():
            print("❌ 生成回复失败")
            return False

        print(f"✅ 生成回复: {response}")

        # 4. 语音合成
        print("🎵 正在合成语音...")
        audio_file = tts.synthesize_speech(response)
        if not audio_file or not os.path.exists(audio_file):
            print("❌ 语音合成失败")
            return False

        print("✅ 语音合成完成")

        # 5. 播放语音
        print("🔊 播放回复...")
        success = player.play_audio_file(audio_file, blocking=True)
        if not success:
            print("❌ 语音播放失败")
            return False

        print("✅ 端到端测试完成")
        return True

    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🧪 语音AI系统模块测试")
    print("=" * 60)

    # 测试结果
    results = {}

    # # 1. 测试音频设备
    # results["audio_devices"] = test_audio_devices()

    # # 2. 测试录音
    # results["recording"] = test_recording()[0]

    # # 3. 测试ASR
    # results["asr"] = test_asr()

    # # 4. 测试LLM
    # results["llm"] = test_llm()

    # 5. 测试TTS
    results["tts"] = test_tts()

    # # 6. 端到端测试
    # if all(results.values()):
    #     results["end_to_end"] = test_end_to_end()
    # else:
    #     print("\n⚠️ 跳过端到端测试，因为某些模块测试失败")
    #     results["end_to_end"] = False

    # 显示测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    overall_success = all(results.values())
    print(f"\n总体结果: {'✅ 所有测试通过' if overall_success else '❌ 部分测试失败'}")

    if not overall_success:
        print("\n💡 提示:")
        print("- 请检查模型文件是否正确下载")
        print("- 确认音频设备工作正常")
        print("- 查看日志文件获取详细错误信息")


if __name__ == "__main__":
    main()
