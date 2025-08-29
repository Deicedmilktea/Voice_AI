import os
import sys
from datetime import datetime
from loguru import logger
from config.config import config


def setup_logger(name: str = "voice_ai") -> object:
    """设置日志配置"""

    # 移除默认的控制台输出
    logger.remove()

    # 确保日志目录存在
    os.makedirs(config.system.log_dir, exist_ok=True)

    # 日志文件路径
    log_file = os.path.join(
        config.system.log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    )

    # 控制台输出格式
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # 文件输出格式
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

    # 添加控制台输出
    logger.add(
        sys.stdout,
        format=console_format,
        level=config.system.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # 添加文件输出
    logger.add(
        log_file,
        format=file_format,
        level=config.system.log_level,
        rotation="00:00",  # 每天轮转
        retention="7 days",  # 保留7天
        compression="zip",  # 压缩旧日志
        backtrace=True,
        diagnose=True,
    )

    return logger


# 全局日志实例
voice_logger = setup_logger()
