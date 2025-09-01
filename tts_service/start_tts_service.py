#!/usr/bin/env python3
"""
TTS服务启动脚本
"""

import os
import sys
import argparse
import uvicorn
from config import TTSConfig


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="启动TTS API服务")
    parser.add_argument("--host", default="127.0.0.1", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8888, help="服务端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--reload", action="store_true", help="自动重载")

    args = parser.parse_args()

    # 更新配置
    config = TTSConfig()
    config.host = args.host
    config.port = args.port
    config.debug = args.debug

    print(f"🚀 启动TTS API服务...")
    print(f"📍 地址: http://{config.host}:{config.port}")
    print(f"📖 API文档: http://{config.host}:{config.port}/docs")
    print(f"🔧 调试模式: {'开启' if config.debug else '关闭'}")
    print("-" * 50)

    try:
        # 启动服务
        uvicorn.run(
            "app:app",
            host=config.host,
            port=config.port,
            reload=args.reload or config.debug,
            log_level="debug" if config.debug else "info",
            access_log=True,
        )
    except KeyboardInterrupt:
        print("\n👋 TTS服务已停止")
    except Exception as e:
        print(f"❌ 启动TTS服务失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
