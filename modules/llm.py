import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from config.config import config
from utils.logger import voice_logger


class LLMProcessor:
    """大语言模型处理器，使用微调后的Qwen3"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.base_model_path = config.model.base_model_path
        self.lora_model_path = config.model.lora_model_path
        self._load_model()

    def _load_model(self):
        """加载LLM模型"""
        try:
            voice_logger.info("正在加载LLM模型...")

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, trust_remote_code=True
            )

            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 量化配置（4bit，更省显存）
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            # 加载基础模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                device_map="auto",
                quantization_config=quant_config,
                trust_remote_code=True,
            ).eval()

            # 加载LoRA权重
            self.model = PeftModel.from_pretrained(
                self.model, model_id=self.lora_model_path
            )

            voice_logger.info("LLM模型加载成功")

        except Exception as e:
            voice_logger.error(f"LLM模型加载失败: {e}")
            raise

    def generate_response(self, user_input: str) -> str:
        """生成对话回复"""
        try:
            if not user_input.strip():
                return "嬛嬛没有听清楚，请再说一遍。"

            voice_logger.info(f"用户输入: {user_input}")

            # 构建对话消息
            messages = self._build_messages(user_input)

            # 生成回复
            response = self._generate_with_model(messages)

            # 更新对话历史
            self._update_conversation_history(user_input, response)

            voice_logger.info(f"模型回复: {response}")
            return response

        except Exception as e:
            voice_logger.error(f"生成回复失败: {e}")
            return "嬛嬛有些不适，稍后再聊吧。"

    def _build_messages(self, user_input: str) -> List[Dict[str, str]]:
        """构建对话消息"""

        messages = [
            {
                # 系统提示词，定义甄嬛的说话风格
                "role": "system",
                "content": "你需要学习甄嬛的说话风格：说话时语气温婉，言辞含蓄，常带比喻或文雅辞藻，但保持简短精炼，每次回答不超过两句话。",
            },
            {"role": "user", "content": "嬛嬛，你今日看起来心情不错啊。"},
            {
                "role": "assistant",
                "content": "心情好坏，不过如天边云卷云舒，姐姐一句关切，倒让我心头暖了几分。",
            },
            {"role": "user", "content": "嬛嬛，你快出牌啊，我等到花都谢了。"},
            {
                "role": "assistant",
                "content": "哪有那么快谢去的花儿，姐姐稍安，嬛嬛这就奉上。",
            },
            {
                "role": "user",
                "content": "嬛嬛，今天你可要手下留情啊，别赢得我们一点银子都不剩。",
            },
            {
                "role": "assistant",
                "content": "哎呀，牌桌上哪有输赢得失，不过是添些笑声热闹。若真要说银子，嬛嬛怎舍得让姐姐们破费。",
            },
            {"role": "user", "content": "嬛嬛，你是不是又偷偷学了新牌技？"},
            {
                "role": "assistant",
                "content": "姐姐取笑了，嬛嬛不过随手摸牌，哪里说得上什么技艺，倒是姐姐才真真是眼明手快。",
            },
        ]

        # 添加对话历史（保持最近的几轮对话）
        max_history = config.system.max_conversation_history
        recent_history = (
            self.conversation_history[-max_history:] if max_history > 0 else []
        )

        for hist in recent_history:
            messages.append({"role": "user", "content": hist["user"]})
            messages.append({"role": "assistant", "content": hist["assistant"]})

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def _generate_with_model(self, messages: List[Dict[str, str]]) -> str:
        """使用模型生成回复"""
        try:
            # 使用chat_template生成输入
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            # 创建attention_mask
            attention_mask = (inputs != self.tokenizer.pad_token_id).long()

            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=config.model.max_new_tokens,
                    do_sample=True,
                    temperature=config.model.temperature,
                    top_p=config.model.top_p,
                    # pad_token_id=self.tokenizer.pad_token_id,
                    # eos_token_id=self.tokenizer.eos_token_id,
                    # repetition_penalty=1.2,  # 增加重复惩罚
                    # no_repeat_ngram_size=3,
                    # early_stopping=True,
                )

            # 解码生成的文本
            response = self.tokenizer.decode(
                outputs[0][inputs.shape[-1] :], skip_special_tokens=True
            )

            # 清理和后处理
            response = self._post_process_response(response)

            return response

        except Exception as e:
            voice_logger.error(f"模型生成失败: {e}")
            raise

    def _post_process_response(self, response: str) -> str:
        """后处理生成的回复"""
        if not response:
            return "嬛嬛一时无言，请稍等片刻。"

        # 移除多余的空格和换行
        response = response.strip()

        # # 移除重复的"Human:"等文本
        # import re

        # response = re.sub(r"(Human:?\s*)+", "", response)
        # response = re.sub(r"(Assistant:?\s*)+", "", response)
        # response = re.sub(r"(用户:?\s*)+", "", response)
        # response = re.sub(r"(助手:?\s*)+", "", response)

        # # 移除多余的空格
        # response = " ".join(response.split())

        # # 查找第一个句号，在此处截断以避免重复
        # if "。" in response:
        #     sentences = response.split("。")
        #     # 保留前两句完整的话
        #     if len(sentences) >= 2 and sentences[0].strip() and sentences[1].strip():
        #         response = sentences[0] + "。" + sentences[1] + "。"
        #     elif sentences[0].strip():
        #         response = sentences[0] + "。"

        # 确保回复不会太长
        if len(response) > 200:
            response = response[:200] + "..."

        # 如果回复为空或太短，提供默认回复
        if len(response.strip()) < 3:
            response = "嬛嬛明白了。"

        return response

    def _update_conversation_history(self, user_input: str, assistant_response: str):
        """更新对话历史"""
        self.conversation_history.append(
            {"user": user_input, "assistant": assistant_response}
        )

        # 限制历史记录长度
        max_history = config.system.max_conversation_history
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history = []
        voice_logger.info("对话历史已清空")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history.copy()

    def set_system_prompt(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt
        voice_logger.info(f"系统提示词已更新: {prompt[:50]}...")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "base_model_path": self.base_model_path,
            "lora_model_path": self.lora_model_path,
            "device": str(self.model.device) if self.model else "unknown",
            "conversation_length": len(self.conversation_history),
            "max_new_tokens": config.model.max_new_tokens,
            "temperature": config.model.temperature,
            "top_p": config.model.top_p,
        }

    def __del__(self):
        """清理资源"""
        try:
            if hasattr(self, "model") and self.model is not None:
                del self.model
            if hasattr(self, "tokenizer") and self.tokenizer is not None:
                del self.tokenizer
        except:
            pass
