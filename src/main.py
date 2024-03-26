"""_summary_
"""
import time
import os
from dataclasses import dataclass
from typing import Annotated
import numpy as np
import uvicorn
import whisper
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from khayyam import JalaliDatetime as jd
from transformers import (
    WhisperPreTrainedModel,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    pipeline
)

from src.config import settings, whispers
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


@app.post("/response", tags=["Whisper"])
async def create_upload_file(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime('%Y-%m-%d')
            directory = f'./uploads/{date_today}'

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime('%H%M%S%f')
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            response = {}
            transcriber = whispers[0]

            AUDIO = np.frombuffer(input_voice, np.int8).flatten().astype(np.float32) / 32768.0
            start = int(1000 * time.time())
            transcript = transcriber(AUDIO)
            end = int(1000 * time.time())
            response = f"inference time is {end - start} milliseconds: \n {transcript['text']}"
            logger.info(
                f"inference time is {end - start} milliseconds: \n {transcript['text']}"
            )
            result = {
                "Transcript": transcript['text'],
                "Transcript Log": response,
                "File Size": len(AUDIO),
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})


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
    tokenizer = WhisperTokenizer.from_pretrained(
        settings.MODEL, language="Persian", task="transcribe"
    )
    # tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small",task="transcribe")

    try:
        transcriber = pipeline(
        model=settings.MODEL,
        tokenizer=tokenizer,
        device="cuda",
        use_fast=False,
        )
    except:
        transcriber = pipeline(
            model=settings.MODEL,
            tokenizer=tokenizer,
            device="cpu",
            use_fast=False,
        )

    logger.info("Models are loaded in Background.")
    return model, transcriber


if __name__ == "__main__":
    uvicorn.run(app="main:app", host="0.0.0.0", port=80)
