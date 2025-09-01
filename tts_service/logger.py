#!/usr/bin/env python3
"""
TTS服务日志配置
"""

import os
import sys
from loguru import logger

# 创建日志目录
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# 配置日志
logger.remove()  # 移除默认处理器

# 控制台输出
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# 文件输出
logger.add(
    os.path.join(log_dir, "tts_service_{time:YYYY-MM-DD}.log"),
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    compression="zip",
)

# 导出TTS日志器
tts_logger = logger.bind(name="TTS_SERVICE")
