import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------------
# 1. 加载模型和Tokenizer
# -------------------------------
model_path = "../model/Qwen3-4B-Instruct-2507/Qwen/Qwen3-4B-Instruct-2507"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 启用 4bit 量化
    bnb_4bit_compute_dtype=torch.float16,  # 计算时用半精度
    bnb_4bit_use_double_quant=True,  # 双重量化，进一步省显存
    bnb_4bit_quant_type="nf4",  # 最优的量化类型
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,  # 使用量化配置
    device_map="auto",  # 自动分配显存
)

# -------------------------------
# 2. LoRA 配置
# -------------------------------
lora_config = LoraConfig(
    r=8,  # LoRA 内部秩
    lora_alpha=16,  # 学习率放大系数
    target_modules=["q_proj", "v_proj"],  # 调整目标层
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# -------------------------------
# 3. 加载数据集
# -------------------------------
dataset = load_dataset(
    "json", data_files={"train": "../dataset/text/huanhuan_max.json"}
)


# 合并 instruction 和 output
def preprocess(example):
    text = f"Instruction: {example['instruction']}\nInput: {example.get('input', '')}\nOutput: {example['output']}"
    tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset["train"].map(preprocess, batched=False)

# -------------------------------
# 4. DataCollator
# -------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt"
)

# -------------------------------
# 5. 训练参数
# -------------------------------
training_args = TrainingArguments(
    output_dir="../output_max",
    per_device_train_batch_size=2,  # 你的显存决定大小
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,  # 用半精度训练节省显存
)

# -------------------------------
# 6. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# -------------------------------
# 7. 开始训练
# -------------------------------
trainer.train()

# 保存 LoRA 模型
model.save_pretrained("../output_max/lora_model")
