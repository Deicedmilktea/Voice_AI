#!/usr/bin/env python3
"""
TTS API客户端 - 用于主项目调用TTS服务
"""

import os
import time
import requests
from typing import Optional, Dict, Any
from config.config import config
from utils.logger import voice_logger


class TTSClient:
    """TTS API客户端"""

    def __init__(self, base_url: str = "http://127.0.0.1:8888"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = 30  # 30秒超时

        # 检查服务是否可用
        self._check_service_health()

    def _check_service_health(self) -> bool:
        """检查TTS服务健康状态"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                voice_logger.info("TTS服务连接成功")
                return True
            else:
                voice_logger.warning(f"TTS服务健康检查失败: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            voice_logger.error(f"无法连接TTS服务: {e}")
            voice_logger.warning("请确保TTS服务已启动")
            return False

    def synthesize_speech(self, text: str, output_path: Optional[str] = None) -> str:
        """合成语音"""
        try:
            if not text.strip():
                voice_logger.warning("输入文本为空，跳过语音合成")
                return ""

            voice_logger.info(f"开始TTS合成: {text[:50]}...")

            # 准备请求数据
            data = {"text": text, "output_format": "wav"}

            # 发送请求
            response = self.session.post(f"{self.base_url}/tts/synthesize", json=data)

            if response.status_code != 200:
                voice_logger.error(
                    f"TTS合成请求失败: {response.status_code} - {response.text}"
                )
                return ""

            result = response.json()

            if not result.get("success", False):
                voice_logger.error(
                    f"TTS合成失败: {result.get('message', 'Unknown error')}"
                )
                return ""

            # 获取音频文件URL
            audio_url = result.get("audio_url")
            if not audio_url:
                voice_logger.error("TTS合成响应中未包含音频URL")
                return ""

            # 下载音频文件
            audio_file_path = self._download_audio_file(audio_url, output_path)

            if audio_file_path:
                voice_logger.info(f"TTS合成完成: {audio_file_path}")
                return audio_file_path
            else:
                return ""

        except requests.exceptions.RequestException as e:
            voice_logger.error(f"TTS API请求失败: {e}")
            return ""
        except Exception as e:
            voice_logger.error(f"TTS合成出错: {e}")
            return ""

    def synthesize_speech_with_reference(
        self, text: str, reference_audio: str, output_path: Optional[str] = None
    ) -> str:
        """使用参考音频合成语音"""
        try:
            if not text.strip():
                voice_logger.warning("输入文本为空，跳过语音合成")
                return ""

            if not os.path.exists(reference_audio):
                voice_logger.error(f"参考音频文件不存在: {reference_audio}")
                return ""

            voice_logger.info(f"开始TTS克隆合成: {text[:50]}...")

            # 准备请求数据
            data = {
                "text": text,
                "reference_audio": reference_audio,
                "output_format": "wav",
            }

            # 发送请求
            response = self.session.post(f"{self.base_url}/tts/synthesize", json=data)

            if response.status_code != 200:
                voice_logger.error(f"TTS克隆合成请求失败: {response.status_code}")
                return ""

            result = response.json()

            if not result.get("success", False):
                voice_logger.error(
                    f"TTS克隆合成失败: {result.get('message', 'Unknown error')}"
                )
                return ""

            # 获取音频文件URL并下载
            audio_url = result.get("audio_url")
            if not audio_url:
                voice_logger.error("TTS克隆合成响应中未包含音频URL")
                return ""

            audio_file_path = self._download_audio_file(audio_url, output_path)

            if audio_file_path:
                voice_logger.info(f"TTS克隆合成完成: {audio_file_path}")
                return audio_file_path
            else:
                return ""

        except requests.exceptions.RequestException as e:
            voice_logger.error(f"TTS克隆API请求失败: {e}")
            return ""
        except Exception as e:
            voice_logger.error(f"TTS克隆合成出错: {e}")
            return ""

    def synthesize_speech_async(self, text: str) -> Optional[str]:
        """异步合成语音，返回任务ID"""
        try:
            if not text.strip():
                voice_logger.warning("输入文本为空，跳过异步语音合成")
                return None

            voice_logger.info(f"创建异步TTS任务: {text[:50]}...")

            # 准备请求数据
            data = {"text": text, "output_format": "wav"}

            # 发送请求
            response = self.session.post(
                f"{self.base_url}/tts/synthesize_async", json=data
            )

            if response.status_code != 200:
                voice_logger.error(f"异步TTS请求失败: {response.status_code}")
                return None

            result = response.json()

            if not result.get("success", False):
                voice_logger.error(
                    f"异步TTS任务创建失败: {result.get('message', 'Unknown error')}"
                )
                return None

            task_id = result.get("task_id")
            voice_logger.info(f"异步TTS任务已创建: {task_id}")
            return task_id

        except requests.exceptions.RequestException as e:
            voice_logger.error(f"异步TTS API请求失败: {e}")
            return None
        except Exception as e:
            voice_logger.error(f"异步TTS任务创建出错: {e}")
            return None

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取异步任务状态"""
        try:
            response = self.session.get(f"{self.base_url}/tts/status/{task_id}")

            if response.status_code == 404:
                voice_logger.warning(f"任务不存在: {task_id}")
                return None
            elif response.status_code != 200:
                voice_logger.error(f"获取任务状态失败: {response.status_code}")
                return None

            return response.json()

        except requests.exceptions.RequestException as e:
            voice_logger.error(f"获取任务状态API请求失败: {e}")
            return None

    def wait_for_task_completion(
        self, task_id: str, timeout: int = 60
    ) -> Optional[str]:
        """等待异步任务完成并下载结果"""
        try:
            start_time = time.time()

            while time.time() - start_time < timeout:
                status = self.get_task_status(task_id)

                if not status:
                    return None

                task_status = status.get("status")

                if task_status == "completed":
                    # 任务完成，下载音频文件
                    audio_url = status.get("result")
                    if audio_url:
                        return self._download_audio_file(audio_url)
                    else:
                        voice_logger.error("任务完成但未返回音频URL")
                        return None

                elif task_status == "failed":
                    error_msg = status.get("error", "Unknown error")
                    voice_logger.error(f"异步TTS任务失败: {error_msg}")
                    return None

                elif task_status in ["pending", "processing"]:
                    # 任务进行中，继续等待
                    progress = status.get("progress", 0)
                    voice_logger.debug(f"任务进度: {progress:.1%}")
                    time.sleep(1)

                else:
                    voice_logger.warning(f"未知任务状态: {task_status}")
                    time.sleep(1)

            voice_logger.error(f"等待异步TTS任务超时: {task_id}")
            return None

        except Exception as e:
            voice_logger.error(f"等待异步任务完成出错: {e}")
            return None

    def _download_audio_file(
        self, audio_url: str, output_path: Optional[str] = None
    ) -> Optional[str]:
        """下载音频文件"""
        try:
            # 构建完整URL
            if audio_url.startswith("/"):
                full_url = f"{self.base_url}{audio_url}"
            else:
                full_url = audio_url

            # 下载音频文件
            response = self.session.get(full_url)

            if response.status_code != 200:
                voice_logger.error(f"下载音频文件失败: {response.status_code}")
                return None

            # 确定输出路径
            if output_path is None:
                output_path = os.path.join(
                    config.system.temp_dir, f"tts_output_{int(time.time())}.wav"
                )

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 保存音频文件
            with open(output_path, "wb") as f:
                f.write(response.content)

            voice_logger.debug(f"音频文件已下载: {output_path}")
            return output_path

        except Exception as e:
            voice_logger.error(f"下载音频文件出错: {e}")
            return None

    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """获取TTS模型信息"""
        try:
            response = self.session.get(f"{self.base_url}/tts/models/info")

            if response.status_code != 200:
                voice_logger.error(f"获取模型信息失败: {response.status_code}")
                return None

            return response.json()

        except requests.exceptions.RequestException as e:
            voice_logger.error(f"获取模型信息API请求失败: {e}")
            return None

    def set_speaker(self, speaker: str) -> bool:
        """设置说话人"""
        try:
            response = self.session.post(
                f"{self.base_url}/tts/config/speaker", params={"speaker": speaker}
            )

            if response.status_code != 200:
                voice_logger.error(f"设置说话人失败: {response.status_code}")
                return False

            voice_logger.info(f"说话人已设置为: {speaker}")
            return True

        except requests.exceptions.RequestException as e:
            voice_logger.error(f"设置说话人API请求失败: {e}")
            return False

    def set_language(self, language: str) -> bool:
        """设置语言"""
        try:
            response = self.session.post(
                f"{self.base_url}/tts/config/language", params={"language": language}
            )

            if response.status_code != 200:
                voice_logger.error(f"设置语言失败: {response.status_code}")
                return False

            voice_logger.info(f"语言已设置为: {language}")
            return True

        except requests.exceptions.RequestException as e:
            voice_logger.error(f"设置语言API请求失败: {e}")
            return False

    def preprocess_text(self, text: str) -> str:
        """预处理文本（本地处理）"""
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

    def is_service_available(self) -> bool:
        """检查服务是否可用"""
        return self._check_service_health()
