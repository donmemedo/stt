"""_summary_
"""
import time
from dataclasses import dataclass
from typing import Annotated

import uvicorn
import whisper
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from khayyam import JalaliDatetime as jd
from transformers import (
    WhisperPreTrainedModel,
    WhisperForConditionalGeneration,
)

from src.config import settings
# from routers.subuser import subuser
from src.logger import logger

app = FastAPI(
    version=settings.VERSION,
    title=settings.SWAGGER_TITLE,
    docs_url="/docs",
    redoc_url="/redocs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_events():
    logger.info(f"Time of Startup:{jd.now().isoformat()}")
    loaders()
    logger.info(f"Ready for Your Questions:{jd.now().isoformat()}")


@dataclass
class Question:
    question: str


@app.post("/wresponse1", tags=["Whisper"])
async def create_file(file: Annotated[bytes | None, File()] = None):
    response = {}
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        AUDIO = file
        for typo in ["cuda", "cpu"]:
            for size in ["base", "small", "medium", "large-v1", "large-v2", "large-v3"]:
                start = int(1000 * time.time())
                try:
                    model = whisper.load_model(name=size, device=typo)
                except:
                    break
                transcript = model.transcribe(AUDIO)
                end = int(1000 * time.time())
                response = f"inference time for {typo} in {size}-Model is {end - start} milliseconds: \n {transcript['text']}"
                logger.info(
                    f"inference time for {typo} in {size}-Model is {end - start} milliseconds: \n {transcript['text']}"
                )
        result = {
            "Transcript": response,
            "File Size": len(file),
            "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }
        return JSONResponse(status_code=200, content=result)


@app.post("/wresponse2", tags=["Whisper"])
async def create_upload_file(file: UploadFile | None = None):
    response = {}
    if not file:
        return JSONResponse(status_code=400, content={"message": "No upload file sent"})
    else:
        AUDIO = file
        for typo in ["cuda", "cpu"]:
            for size in ["base", "small", "medium", "large-v1", "large-v2", "large-v3"]:
                start = int(1000 * time.time())
                try:
                    model = whisper.load_model(name=size, device=typo)
                except:
                    break
                transcript = model.transcribe(AUDIO)
                end = int(1000 * time.time())
                response = f"inference time for {typo} in {size}-Model is {end - start} milliseconds: \n {transcript['text']}"
                logger.info(
                    f"inference time for {typo} in {size}-Model is {end - start} milliseconds: \n {transcript['text']}"
                )
        result = {
            "Transcript": response,
            "File Name": file.filename,
            "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }
        return JSONResponse(status_code=200, content=result)


@app.post("/mresponse1", tags=["Man"])
async def create_file(file: Annotated[bytes | None, File()] = None):
    response = {}
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        AUDIO = file
        for typo in ["cuda", "cpu"]:
            start = int(1000 * time.time())
            try:
                model = WhisperPreTrainedModel.from_pretrained(
                    settings.MODEL, device_map=typo
                )
            except:
                break
            transcript = model.transcribe(AUDIO)
            end = int(1000 * time.time())
            response = f"inference time for {typo} in {settings.MODEL}-Model is {end - start} milliseconds: \n {transcript['text']}"
            logger.info(
                f"inference time for {typo} in {settings.MODEL}-Model is {end - start} milliseconds: \n {transcript['text']}"
            )
        result = {
            "Transcript": response,
            "File Size": len(file),
            "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }
        return JSONResponse(status_code=200, content=result)


@app.post("/mresponse2", tags=["Man"])
async def create_upload_file(file: UploadFile | None = None):
    response = {}
    if not file:
        return JSONResponse(status_code=400, content={"message": "No upload file sent"})
    else:
        AUDIO = file
        for typo in ["cuda", "cpu"]:
            start = int(1000 * time.time())
            try:
                model = WhisperPreTrainedModel.from_pretrained(
                    settings.MODEL, device_map=typo
                )
            except:
                break
            transcript = model.transcribe(AUDIO)
            end = int(1000 * time.time())
            response = f"inference time for {typo} in {settings.MODEL}-Model is {end - start} milliseconds: \n {transcript['text']}"
            logger.info(
                f"inference time for {typo} in {settings.MODEL}-Model is {end - start} milliseconds: \n {transcript['text']}"
            )
        result = {
            "Transcript": response,
            "File Name": file.filename,
            "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }
        return JSONResponse(status_code=200, content=result)


@app.post("/mmresponse1", tags=["Maan"])
async def create_file(file: Annotated[bytes | None, File()] = None):
    response = {}
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        AUDIO = file
        for typo in ["cuda", "cpu"]:
            start = int(1000 * time.time())
            try:
                model = WhisperForConditionalGeneration.from_pretrained(
                    settings.MODEL, device_map=typo
                )
            except:
                break
            transcript = model.transcribe(AUDIO)
            end = int(1000 * time.time())
            response = f"inference time for {typo} in {settings.MODEL}-Model is {end - start} milliseconds: \n {transcript['text']}"
            logger.info(
                f"inference time for {typo} in {settings.MODEL}-Model is {end - start} milliseconds: \n {transcript['text']}"
            )
        result = {
            "Transcript": response,
            "File Size": len(file),
            "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }
        return JSONResponse(status_code=200, content=result)


@app.post("/mmresponse2", tags=["Maan"])
async def create_upload_file(file: UploadFile | None = None):
    response = {}
    if not file:
        return JSONResponse(status_code=400, content={"message": "No upload file sent"})
    else:
        AUDIO = file
        for typo in ["cuda", "cpu"]:
            start = int(1000 * time.time())
            try:
                model = WhisperForConditionalGeneration.from_pretrained(
                    settings.MODEL, device_map=typo
                )
            except:
                break
            transcript = model.transcribe(AUDIO)
            end = int(1000 * time.time())
            response = f"inference time for {typo} in {settings.MODEL}-Model is {end - start} milliseconds: \n {transcript['text']}"
            logger.info(
                f"inference time for {typo} in {settings.MODEL}-Model is {end - start} milliseconds: \n {transcript['text']}"
            )
        result = {
            "Transcript": response,
            "File Name": file.filename,
            "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        }
        return JSONResponse(status_code=200, content=result)


@app.get("/ip-getter", tags=["Default"])
async def read_root(request: Request):
    client_host = request.client.host
    client_scope = request.scope["client"]
    logger.info(f"client host is {client_host}")
    logger.info(f"client scope is {client_scope}")

    return {"client_host": client_host, "client_scope": client_scope}


@app.get("/test", tags=["Default"])
def background_loader():
    for size in ["base", "small", "medium", "large-v1", "large-v2", "large-v3"]:
        try:
            model = whisper.load_model(name=size)
        except:
            pass
    model = WhisperPreTrainedModel.from_pretrained(settings.MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(settings.MODEL)
    logger.info("Models are loaded in Background.")


def loaders():
    for size in ["base", "small", "medium", "large-v1", "large-v2", "large-v3"]:
        try:
            model = whisper.load_model(name=size)
        except:
            pass
    model = WhisperPreTrainedModel.from_pretrained(settings.MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(settings.MODEL)
    logger.info("Models are loaded in Background.")


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=80)
