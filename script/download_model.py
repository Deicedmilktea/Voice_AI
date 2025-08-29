import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os

model_dir = snapshot_download(
    "Qwen/Qwen3-4B-Instruct-2507",
    cache_dir="../model/Qwen3-4B-Instruct-2507",
    revision="master",
)
