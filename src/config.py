"""_summary_

Returns:
    _type_: _description_
"""
import whisper
from pydantic_settings import BaseSettings
from transformers import (
    WhisperPreTrainedModel,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    pipeline,
)


class Settings(BaseSettings):
    """_summary_

    Args:
        BaseSettings (_type_): _description_
    """

    API_PREFIX: str = ""
    DOCS_URL: str = ""

    OPENAPI_URL: str = ""
    ORIGINS: str = "*"
    ROOT_PATH: str = ""
    SWAGGER_TITLE: str = "Speech To Text"
    VERSION: str = "1.1.0"

    APPLICATION_ID: str = "d7f48c21-2a19-4bdb-ace8-48928bff0eb5"
    # GRPC_IP: str = "172.24.65.20"
    # GRPC_PORT: int = 9035
    SPLUNK_HOST: str = "172.24.65.206"
    SPLUNK_PORT: int = 5141
    SPLUNK_INDEX: str = "dev"

    DATE_STRING: str = "%Y-%m-%d"
    FASTAPI_DOCS: str = "/docs"
    FASTAPI_REDOC: str = "/redoc"
    MODEL: str = "makhataei/Whisper-Small-Ctejarat"


def loaders():
    for size in ["base", "small", "medium", "large-v1", "large-v2", "large-v3"]:
        try:
            model = whisper.load_model(name=size)
        except:
            pass
    model_wpt = WhisperPreTrainedModel.from_pretrained(settings.MODEL)
    model_wcg = WhisperForConditionalGeneration.from_pretrained(settings.MODEL)
    tokenizer = WhisperTokenizer.from_pretrained(
        settings.MODEL, language="Persian", task="transcribe"
    )
    transcriber = pipeline(
        model=settings.MODEL,
        tokenizer=tokenizer,
        device="cpu",
        use_fast=False,
    )
    # ToDo: Uncomment these on GPU Server
    # try:
    #     transcriber = pipeline(
    #     model=settings.MODEL,
    #     tokenizer=tokenizer,
    #     device="cuda",
    #     use_fast=False,
    #     )
    # except:
    #     transcriber = pipeline(
    #         model=settings.MODEL,
    #         tokenizer=tokenizer,
    #         device="cpu",
    #         use_fast=False,
    #     )
    return transcriber, model_wpt, model_wcg


settings = Settings()
whispers = loaders()
