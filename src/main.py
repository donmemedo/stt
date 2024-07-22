"""_summary_
"""
import os
import time

import numpy as np
import requests
import uvicorn
import whisper
from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from khayyam import JalaliDatetime as jd
from transformers import (
    WhisperPreTrainedModel,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    pipeline,
)

from src.config import settings, whispers
from src.logger import logger
from src.mute.mute_finder import percentage_mute_finder
from src.utils.inference import prediction, prediction121, prediction201, predicto
from src.utils.pred import predict_ser, predict2_ser

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


# app.add_middleware(
#     TrustedHostMiddleware, allowed_hosts = settings.ORIGINS.split(","),
# )


@app.on_event("startup")
async def startup_events():
    logger.info(f"Time of Startup:{jd.now().isoformat()}")
    loaders()
    logger.info(f"Ready for Your Questions:{jd.now().isoformat()}")


@app.post("/stt/response", tags=["STT"])
async def speech_to_text(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./stt_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            response = {}
            transcriber = whispers[0]

            AUDIO = (
                    np.frombuffer(input_voice, np.int8).flatten().astype(np.float32)
                    / 32768.0
            )
            start = int(1000 * time.time())
            transcript = transcriber(AUDIO)
            end = int(1000 * time.time())
            response = (
                f"inference time is {end - start} milliseconds: \n {transcript['text']}"
            )
            logger.info(
                f"inference time is {end - start} milliseconds: \n {transcript['text']}"
            )
            result = {
                "Transcript": transcript["text"],
                "Transcript Log": response,
                "File Size": len(AUDIO),
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

            # if not os.path.exists(directory):
            #     os.makedirs(directory)


@app.post("/ser/response", tags=["SER"])
async def speech_emotion_recognition(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./ser_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            response = {}
            AUDIO = (
                    np.frombuffer(input_voice, np.int8).flatten().astype(np.float32)
                    / 32768.0
            )
            start = int(1000 * time.time())
            emotion = prediction(image_bytes=input_voice)
            end = int(1000 * time.time())
            response = f"inference time is {end - start} milliseconds: \n {emotion}"
            logger.info(f"inference time is {end - start} milliseconds: \n {emotion}")
            result = {
                "Transcript": emotion,
                "Transcript Log": response,
                "File Size": len(AUDIO),
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/ser/response121", tags=["TEMP"])
# @app.post("/ser/response121", tags=["SER"])
async def speech_emotion_recognition121(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./ser_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            response = {}
            AUDIO = (
                    np.frombuffer(input_voice, np.int8).flatten().astype(np.float32)
                    / 32768.0
            )
            start = int(1000 * time.time())
            emotion = prediction121(image_bytes=input_voice)
            end = int(1000 * time.time())
            response = f"inference time is {end - start} milliseconds: \n {emotion}"
            logger.info(f"inference time is {end - start} milliseconds: \n {emotion}")
            result = {
                "Transcript": emotion,
                "Transcript Log": response,
                "File Size": len(AUDIO),
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/ser/response201", tags=["TEMP"])
# @app.post("/ser/response201", tags=["SER"])
async def speech_emotion_recognition201(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./ser_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            response = {}
            AUDIO = (
                    np.frombuffer(input_voice, np.int8).flatten().astype(np.float32)
                    / 32768.0
            )
            start = int(1000 * time.time())
            emotion = prediction201(image_bytes=input_voice)
            end = int(1000 * time.time())
            response = f"inference time is {end - start} milliseconds: \n {emotion}"
            logger.info(f"inference time is {end - start} milliseconds: \n {emotion}")
            result = {
                "Transcript": emotion,
                "Transcript Log": response,
                "File Size": len(AUDIO),
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/ser/responseAll", tags=["TEMP"])
# @app.post("/ser/responseAll", tags=["SER"])
async def speech_emotion_recognitionAll(file: UploadFile, model_type):
    if model_type == '121' or model_type == '201':
        pass
    else:
        return JSONResponse(status_code=400, content={"message": "You MUST choose 121 or 201"})
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./ser_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            response = {}
            AUDIO = (
                    np.frombuffer(input_voice, np.int8).flatten().astype(np.float32)
                    / 32768.0
            )
            start = int(1000 * time.time())
            emotion = predicto(model_type=model_type, image_bytes=input_voice)
            end = int(1000 * time.time())
            response = f"inference time is {end - start} milliseconds: \n {emotion}"
            logger.info(f"inference time is {end - start} milliseconds: \n {emotion}")
            result = {
                "Transcript": emotion,
                "Transcript Log": response,
                "File Size": len(AUDIO),
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/ser/KELONMYOSA", tags=["SER"])
async def kelonmyosa(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./ser_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            start = int(1000 * time.time())
            response = predict_ser(file_location)
            pred = {"label": None, "score": 0}
            for zico in response:
                if zico["score"] > pred["score"]:
                    pred["label"] = zico["label"]
                    pred["score"] = zico["score"]
                zico["score"] = int(zico["score"] * 10000) / 100

            logger.info(response)
            end = int(1000 * time.time())
            logger.info(f"inference time is {end - start} milliseconds: \n {pred['label']}")
            result = {
                "Label": pred['label'],
                "Score": int(pred['score'] * 10000) / 100,
                "Transcript Log": response,
                "Inference Log": f"{end - start} milliseconds",
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/ser/ehcalabres", tags=["SER"])
async def ehcalabres(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./ser_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            start = int(1000 * time.time())
            response = predict2_ser(file_location)
            pred = {"label": None, "score": 0}
            for zico in response:
                if zico["score"] > pred["score"]:
                    pred["label"] = zico["label"]
                    pred["score"] = zico["score"]
                zico["score"] = int(zico["score"] * 10000) / 100

            logger.info(response)
            end = int(1000 * time.time())
            logger.info(f"inference time is {end - start} milliseconds: \n {pred['label']}")
            result = {
                "Label": pred['label'],
                "Score": int(pred['score'] * 10000) / 100,
                "Transcript Log": response,
                "Inference Log": f"{end - start} milliseconds",
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/ser/Emotion", tags=["TEMP"])
# @app.post("/ser/Emotion", tags=["SER"])
async def adultchild(file: UploadFile):
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            API_URL = "https://api-inference.huggingface.co/models/anantoj/wav2vec2-adult-child-cls"
            headers = {"Authorization": "Bearer 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"}
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./ser_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            response = requests.post(API_URL, headers=headers, data=input_voice).json()
            start = int(1000 * time.time())
            end = int(1000 * time.time())
            result = {
                "Score": int(response[0]['score'] * 100),
                "Transcript Log": response,
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=200, content=result)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/mute/finder", tags=["MUTE"])
async def mute_finder(file: UploadFile, mo_db_sensitivity: float, mute_time_sensitivity: float,tone_db_sensitivity: float, tone_time_sensitivity: float,
                      overlap_time_sensitivity: float):
    try:
        float(mo_db_sensitivity)
        float(mute_time_sensitivity)
        float(tone_db_sensitivity)
        float(tone_time_sensitivity)
        float(overlap_time_sensitivity)
    except:
        return JSONResponse(status_code=400, content={"message": "You MUST choose Number for sensitivity"})
    if float(mo_db_sensitivity) > 0:
        pass
    else:
        return JSONResponse(status_code=400, content={"message": "You MUST choose correct Decibel sensitivity"})
    if float(mute_time_sensitivity) > 0:
        pass
    else:
        return JSONResponse(status_code=400, content={"message": "You MUST choose correct Mute Time sensitivity"})
    if float(tone_db_sensitivity) > 0:
        pass
    else:
        return JSONResponse(status_code=400, content={"message": "You MUST choose correct Decibel sensitivity"})
    if float(tone_time_sensitivity) > 0:
        pass
    else:
        return JSONResponse(status_code=400, content={"message": "You MUST choose correct Mute Time sensitivity"})
    if float(overlap_time_sensitivity) > 0:
        pass
    else:
        return JSONResponse(status_code=400, content={"message": "You MUST choose correct Overlap Time sensitivity"})
    if not file:
        return JSONResponse(status_code=400, content={"message": "No file sent"})
    else:
        try:
            input_voice = file.file.read()
            # Create a directory named with today's date
            date_today = jd.now().strftime("%Y-%m-%d")
            directory = f"./mute_uploads/{date_today}"

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the file with the date and time included in the filename
            date_time_now = jd.now().strftime("%H%M%S%f")
            # if file.filename.split('.')[1] != 'wav':
            #     return JSONResponse(status_code=412, content={"message": "You must send a WAV file."})
            filename = f"{date_time_now}_{file.filename}"
            file_location = f"{directory}/{filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(input_voice)
            response = {}
            AUDIO = (
                    np.frombuffer(input_voice, np.int8).flatten().astype(np.float32)
                    / 32768.0
            )
            start = int(1000 * time.time())
            percentage = percentage_mute_finder(file_location, mo_db_sensitivity, mute_time_sensitivity,
                                                tone_db_sensitivity, tone_time_sensitivity, overlap_time_sensitivity)
            end = int(1000 * time.time())
            response = f"inference time is {end - start} milliseconds when Mute Decibel Sensitivity is {mo_db_sensitivity}, Mute Time Sensitivity is {mute_time_sensitivity},Tone Decibel Sensitivity is {tone_db_sensitivity}, Tone Time Sensitivity is {tone_time_sensitivity}, and Overlap Time Sensitivity is {overlap_time_sensitivity}: \n {percentage}"
            logger.info(
                f"inference time is {end - start} milliseconds when Mute Decibel Sensitivity is {mo_db_sensitivity}, Mute Time Sensitivity is {mute_time_sensitivity}, Tone Decibel Sensitivity is {tone_db_sensitivity}, Tone Time Sensitivity is {tone_time_sensitivity}, and Overlap Time Sensitivity is {overlap_time_sensitivity}: \n {percentage}")
            result = {
                "Mute Percentage": percentage[1],
                "List of Mute times": percentage[2],
                "Mute times More Than Sensitivity": percentage[3],
                "List of Mute times More Than Sensitivity": percentage[4],
                "List of Overlap times More Than Sensitivity": percentage[5],
                "Tone Percentage": percentage[6],
                "List of Tone times": percentage[7],
                "Tone times More Than Sensitivity": percentage[8],
                "List of Tone times More Than Sensitivity": percentage[9],
                "Transcript Log": response,
                "File Size": len(AUDIO),
                "timeGenerated": jd.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
            return JSONResponse(status_code=percentage[0], content=result)
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
    # uvicorn.run(app="main:app", host="37.114.196.45", port=9005)
