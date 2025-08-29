import sounddevice as sd

# 查询所有设备
devices = sd.query_devices()

# 打印所有设备
print(devices)
