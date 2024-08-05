from transformers import pipeline, WhisperTokenizer

# tokenizer = WhisperTokenizer.from_pretrained(
#     "openai/whisper-small", language="Persian", task="transcribe"
# )
tokenizer = WhisperTokenizer.from_pretrained(
    "makhataei/Whisper-Small-Ctejarat", language="Persian", task="transcribe"
)
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small",task="transcribe")
import whisper

transcriber = pipeline(
    model="makhataei/Whisper-Small-Ctejarat",  # "makhataei/Whisper-Small-Common-Voice",
    tokenizer=tokenizer,
    device="cuda",
    use_fast=False,
)
# transcriber = AutomaticSpeechRecognitionPipeline("makhataei/Whisper-Small-Common-Voice")
print("4556868.wav:\n")
print(
    transcriber("/media/makhataei/Backups/Projects/STT/test/100 Calls/4556868.wav")[
        "text"
    ]
)
print("4557220.wav:\n")
print(
    transcriber("/media/makhataei/Backups/Projects/STT/test/100 Calls/4557220.wav")[
        "text"
    ]
)
print("4527484.wav:\n")
print(
    transcriber("/media/makhataei/Backups/Projects/STT/test/100 Calls/4527484.wav")[
        "text"
    ]
)

for model in ["base", "small", "medium", "large-v2", "large-v3"]:
    print(f"\t\t{model}\n\n:")
    # tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{model}", language="persian", task="transcribe")
    # try:
    #     transcriber = pipeline(model=f"openai/whisper-{model}", tokenizer=tokenizer, device='cuda',
    #                            use_fast=False,task="automatic-speech-recognition")
    # except:
    #     transcriber = pipeline(model=f"openai/whisper-{model}", tokenizer=tokenizer, device='cpu',
    #                            use_fast=False,task="automatic-speech-recognition")

    # transcriber = AutomaticSpeechRecognitionPipeline("makhataei/Whisper-Small-Common-Voice")
    # print("4556868.wav:\n")
    # print(transcriber("/media/makhataei/Backups/Projects/STT/test/100 Calls/4556868.wav")['text'])
    # print("4557220.wav:\n")
    # print(transcriber("/media/makhataei/Backups/Projects/STT/test/100 Calls/4557220.wav")['text'])
    # print("4527484.wav:\n")
    # print(transcriber("/media/makhataei/Backups/Projects/STT/test/100 Calls/4527484.wav")['text'])
    try:
        model = whisper.load_model(model, device="cuda")
    except:
        model = whisper.load_model(model, device="cpu")
    print("4556868.wav:\n")
    print(
        model.transcribe(
            "/media/makhataei/Backups/Projects/STT/test/100 Calls/4556868.wav"
        )["text"]
    )
    print("4557220.wav:\n")
    print(
        model.transcribe(
            "/media/makhataei/Backups/Projects/STT/test/100 Calls/4557220.wav"
        )["text"]
    )
    print("4527484.wav:\n")
    print(
        model.transcribe(
            "/media/makhataei/Backups/Projects/STT/test/100 Calls/4527484.wav"
        )["text"]
    )
