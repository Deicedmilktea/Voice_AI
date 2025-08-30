# import sounddevice as sd
# import torch

# # 查询所有设备
# devices = sd.query_devices()

# # 打印所有设备
# print(devices)

# # if torch.cuda.is_available():
# #     print("CUDA is available")
# # else:
# #     print("CUDA is not available")

from TTS.api import TTS
import os
from TTS.tts.configs.xtts_config import XttsConfig
import torch

local_path = "./models/xtts-v2/AI-ModelScope/XTTS-v2"

# 检查必要文件
config_path = os.path.join(local_path, "config.json")
model_path = os.path.join(local_path, "model.pth")

# # 使用 safe_globals 允许 XttsConfig 被反序列化
# with torch.serialization.safe_globals([XttsConfig]):
#     tts = TTS(
#         config_path=config_path,
#         model_path=model_path,
#         progress_bar=True,
#     )
# print("模型加载成功！")
# # else:
# #     print("config.json 不存在，请检查路径")

model = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    gpu=True,
)

print(model.speakers)
