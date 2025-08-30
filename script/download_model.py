import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download(
    "Qwen/Qwen3-4B-Instruct-2507",
    cache_dir="../models/Qwen3-4B",
    revision="master",
)

model_dir = snapshot_download(
    "AI-ModelScope/XTTS-v2",
    cache_dir="../models/xtts-v2",
    revision="master",
)
