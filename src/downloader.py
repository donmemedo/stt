"""_summary_
"""
import whisper
from transformers import (
    WhisperPreTrainedModel,
    WhisperForConditionalGeneration,
)

from src.config import settings
# from routers.subuser import subuser
from src.logger import logger


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
