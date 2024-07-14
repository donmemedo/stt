"""_summary_
"""
import whisper
from transformers import (
    WhisperPreTrainedModel,
    WhisperForConditionalGeneration,
    AutoModelForAudioClassification
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
    AutoModelForAudioClassification.from_pretrained("KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru", trust_remote_code=True)
    AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", trust_remote_code=True)
    logger.info("Models are loaded in Background.")


background_loader()
