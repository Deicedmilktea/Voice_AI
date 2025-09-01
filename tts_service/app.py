#!/usr/bin/env python3
"""
TTS API服务 - 独立的Text-to-Speech微服务
使用FastAPI提供RESTful API接口
"""

import os
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

from tts_processor import TTSProcessor
from config import TTSConfig
from logger import tts_logger

# 初始化FastAPI应用
app = FastAPI(
    title="TTS Service API", description="文字转语音微服务API", version="1.0.0"
)

# 全局TTS处理器
tts_processor: Optional[TTSProcessor] = None
config = TTSConfig()


class TTSRequest(BaseModel):
    """TTS请求模型"""

    text: str
    reference_audio: Optional[str] = None
    language: Optional[str] = None
    speaker: Optional[str] = None
    output_format: str = "wav"


class TTSResponse(BaseModel):
    """TTS响应模型"""

    success: bool
    message: str
    audio_url: Optional[str] = None
    task_id: Optional[str] = None
    duration: Optional[float] = None


class TTSStatus(BaseModel):
    """TTS任务状态"""

    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float = 0.0
    result: Optional[str] = None
    error: Optional[str] = None


# 任务状态存储（生产环境应使用Redis等）
task_status: Dict[str, TTSStatus] = {}


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化TTS处理器"""
    global tts_processor
    try:
        tts_logger.info("正在启动TTS服务...")
        tts_processor = TTSProcessor()
        tts_logger.info("TTS服务启动成功")
    except Exception as e:
        tts_logger.error(f"TTS服务启动失败: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    global tts_processor
    try:
        tts_logger.info("正在关闭TTS服务...")
        if tts_processor:
            del tts_processor
        tts_logger.info("TTS服务已关闭")
    except Exception as e:
        tts_logger.error(f"关闭TTS服务时出错: {e}")


def cleanup_old_files():
    """清理过期的音频文件"""
    try:
        output_dir = Path(config.output_dir)
        if output_dir.exists():
            current_time = time.time()
            for file_path in output_dir.glob("*.wav"):
                # 删除超过1分钟的文件
                if current_time - file_path.stat().st_mtime > 60:
                    file_path.unlink()
                    tts_logger.debug(f"清理过期文件: {file_path}")
    except Exception as e:
        tts_logger.error(f"清理文件失败: {e}")


@app.get("/")
async def root():
    """根路径"""
    return {"message": "TTS Service API", "status": "running"}


@app.get("/health")
async def health_check():
    """健康检查"""
    if tts_processor is None:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    return {
        "status": "healthy",
        "model_info": tts_processor.get_model_info(),
        "timestamp": time.time(),
    }


@app.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """同步语音合成接口"""
    if tts_processor is None:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    try:
        start_time = time.time()
        tts_logger.info(f"开始合成语音: {request.text[:50]}...")

        # 验证输入
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if len(request.text) > 1000:
            raise HTTPException(
                status_code=400, detail="Text too long (max 1000 characters)"
            )

        # 生成唯一文件名
        task_id = str(uuid.uuid4())
        output_filename = f"{task_id}.{request.output_format}"
        output_path = os.path.join(config.output_dir, output_filename)

        # 确保输出目录存在
        os.makedirs(config.output_dir, exist_ok=True)

        # 执行TTS合成
        if request.reference_audio:
            # 使用参考音频进行声音克隆
            result_path = tts_processor.synthesize_speech_with_reference(
                text=request.text,
                reference_audio=request.reference_audio,
                output_path=output_path,
            )
        else:
            # 标准TTS合成
            result_path = tts_processor.synthesize_speech(
                text=request.text, output_path=output_path
            )

        if not result_path or not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="Speech synthesis failed")

        duration = time.time() - start_time

        # 添加清理任务
        background_tasks.add_task(cleanup_old_files)

        # 返回音频文件URL
        audio_url = f"/tts/audio/{output_filename}"

        tts_logger.info(f"语音合成完成，耗时: {duration:.2f}秒")

        return TTSResponse(
            success=True,
            message="Speech synthesis completed",
            audio_url=audio_url,
            task_id=task_id,
            duration=duration,
        )

    except HTTPException:
        raise
    except Exception as e:
        tts_logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/tts/synthesize_async", response_model=TTSResponse)
async def synthesize_speech_async(
    request: TTSRequest, background_tasks: BackgroundTasks
):
    """异步语音合成接口"""
    if tts_processor is None:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    try:
        # 验证输入
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if len(request.text) > 1000:
            raise HTTPException(
                status_code=400, detail="Text too long (max 1000 characters)"
            )

        # 生成任务ID
        task_id = str(uuid.uuid4())

        # 初始化任务状态
        task_status[task_id] = TTSStatus(
            task_id=task_id, status="pending", progress=0.0
        )

        # 添加后台任务
        background_tasks.add_task(process_tts_async, task_id, request)

        tts_logger.info(f"异步TTS任务已创建: {task_id}")

        return TTSResponse(success=True, message="TTS task created", task_id=task_id)

    except Exception as e:
        tts_logger.error(f"创建异步TTS任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


async def process_tts_async(task_id: str, request: TTSRequest):
    """异步处理TTS任务"""
    try:
        # 更新状态为处理中
        task_status[task_id].status = "processing"
        task_status[task_id].progress = 0.1

        # 生成输出文件路径
        output_filename = f"{task_id}.{request.output_format}"
        output_path = os.path.join(config.output_dir, output_filename)

        # 确保输出目录存在
        os.makedirs(config.output_dir, exist_ok=True)

        # 更新进度
        task_status[task_id].progress = 0.3

        # 执行TTS合成
        if request.reference_audio:
            result_path = tts_processor.synthesize_speech_with_reference(
                text=request.text,
                reference_audio=request.reference_audio,
                output_path=output_path,
            )
        else:
            result_path = tts_processor.synthesize_speech(
                text=request.text, output_path=output_path
            )

        # 更新进度
        task_status[task_id].progress = 0.9

        if result_path and os.path.exists(result_path):
            # 任务完成
            audio_url = f"/tts/audio/{output_filename}"
            task_status[task_id].status = "completed"
            task_status[task_id].progress = 1.0
            task_status[task_id].result = audio_url
            tts_logger.info(f"异步TTS任务完成: {task_id}")
        else:
            # 任务失败
            task_status[task_id].status = "failed"
            task_status[task_id].error = "Speech synthesis failed"
            tts_logger.error(f"异步TTS任务失败: {task_id}")

    except Exception as e:
        # 任务失败
        task_status[task_id].status = "failed"
        task_status[task_id].error = str(e)
        tts_logger.error(f"异步TTS任务异常: {task_id}, {e}")


@app.get("/tts/status/{task_id}", response_model=TTSStatus)
async def get_task_status(task_id: str):
    """获取异步任务状态"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_status[task_id]


@app.get("/tts/audio/{filename}")
async def get_audio_file(filename: str):
    """获取音频文件"""
    file_path = os.path.join(config.output_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(file_path, media_type="audio/wav", filename=filename)


@app.delete("/tts/audio/{filename}")
async def delete_audio_file(filename: str):
    """删除音频文件"""
    file_path = os.path.join(config.output_dir, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        os.remove(file_path)
        return {"message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.get("/tts/models/info")
async def get_model_info():
    """获取模型信息"""
    if tts_processor is None:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    return tts_processor.get_model_info()


@app.post("/tts/config/speaker")
async def set_speaker(speaker: str):
    """设置说话人"""
    if tts_processor is None:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    try:
        tts_processor.set_speaker(speaker)
        return {"message": f"Speaker set to {speaker}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set speaker: {str(e)}")


@app.post("/tts/config/language")
async def set_language(language: str):
    """设置语言"""
    if tts_processor is None:
        raise HTTPException(status_code=503, detail="TTS processor not initialized")

    try:
        tts_processor.set_language(language)
        return {"message": f"Language set to {language}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to set language: {str(e)}")


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info",
    )
