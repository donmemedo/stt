"""_summary_

Returns:
    _type_: _description_
"""
from pydantic_settings import BaseSettings


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
    SWAGGER_TITLE: str = "ChatBotBAK"
    VERSION: str = "0.0.1"

    APPLICATION_ID: str = "d7f48c21-2a19-4bdb-ace8-48928bff0eb5"
    # GRPC_IP: str = "172.24.65.20"
    # GRPC_PORT: int = 9035
    SPLUNK_HOST: str = "172.24.65.206"
    SPLUNK_PORT: int = 5141
    SPLUNK_INDEX: str = "dev"

    DATE_STRING: str = "%Y-%m-%d"
    FASTAPI_DOCS: str = "/docs"
    FASTAPI_REDOC: str = "/redoc"
    MODEL: str = "makhataei/Whisper-Small-Common-Voice"


settings = Settings()
