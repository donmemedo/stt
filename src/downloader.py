"""_summary_
"""
import datetime
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from khayyam import JalaliDatetime as jd
import asyncio
from src.config import settings
from dataclasses import dataclass

# from routers.subuser import subuser
from src.logger import logger
import time
from transformers import AutoModel, AutoTokenizer, pipeline, WhisperModel,WhisperPreTrainedModel,WhisperForConditionalGeneration
import torch
import whisper



def background_loader():
    for size in ["base", "small", "medium", "large-v1", "large-v2", "large-v3"]:
        try:
            model = whisper.load_model(name=size)
        except:
            pass
    model = WhisperPreTrainedModel.from_pretrained(settings.MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(settings.MODEL)
    logger.info("Models are loaded in Background.")


background_loader()
