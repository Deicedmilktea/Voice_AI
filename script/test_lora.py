from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel

base_model = "../model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"
lora_model = "../output_max/lora_model"

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# 量化配置（4bit，更省显存）
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 加载基础模型（自动分配 GPU/CPU，节省显存）
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=quant_config,
    trust_remote_code=True,
).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_model)

# prompt = "嬛嬛你怎么了，朕替你打抱不平！"
# prompt = "嬛嬛你好好打扮，朕带你出去玩耍！"
# prompt = "嬛嬛，你快出牌啊，我等到花都谢了。"

# messages = [
#     {
#         "role": "system",
#         "content": "假设你是皇帝身边的女人--甄嬛。你现在正在和朋友们一起打麻将，并且在一起聊天，心情很好。",
#     },
#     {"role": "user", "content": prompt},
# ]

messages = [
    {
        "role": "system",
        "content": "你需要学习甄嬛的说话风格：说话时语气温婉，言辞含蓄，常带比喻或文雅辞藻，但保持简短精炼，每次回答不超过两句话。",
    },
    {"role": "user", "content": "嬛嬛，你今日看起来心情不错啊。"},
    {
        "role": "assistant",
        "content": "心情好坏，不过如天边云卷云舒，姐姐一句关切，倒让我心头暖了几分。",
    },
    {"role": "user", "content": "嬛嬛，你快出牌啊，我等到花都谢了。"},
    {"role": "assistant", "content": "哪有那么快谢去的花儿，姐姐稍安，嬛嬛这就奉上。"},
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

messages.append({"role": "user", "content": "嬛嬛，今日运气不佳，我们改日再聚吧！"})

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 用 chat_template 生成输入
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True,  # 显式padding
).to(model.device)

# 手动创建 attention_mask（非 pad_token 位置为1）
attention_mask = (inputs != tokenizer.pad_token_id).long()

# 生成
with torch.no_grad():
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

response = tokenizer.decode(outputs[0][inputs.shape[-1] :], skip_special_tokens=True)

# print("皇上：", prompt)
print("嬛嬛：", response)
