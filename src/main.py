from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    use_auth_token="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")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cuda"))

# apply pretrained pipeline
diarization = pipeline("/home/makhataei/Projects/STT/test/4527484.wav")
print("----------------------------------4527484.wav----------------------------------------")
# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
diarization = pipeline("/home/makhataei/Projects/STT/test/4527674.wav")
print("----------------------------------4527674.wav----------------------------------------")
# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
diarization = pipeline("/home/makhataei/Projects/STT/test/4528104.wav")
print("----------------------------------4528104.wav----------------------------------------")
# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
diarization = pipeline("/home/makhataei/Projects/STT/test/4556868.wav")
print("----------------------------------4556868.wav----------------------------------------")
# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
diarization = pipeline("/home/makhataei/Projects/STT/test/4557220.wav")
print("----------------------------------4557220.wav----------------------------------------")
# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
# # start=0.2s stop=1.5s speaker_0
# # start=1.8s stop=3.9s speaker_1
# # start=4.2s stop=5.7s speaker_0
# # ...
#
#
# import whisper
#
# model = whisper.load_model("base")
# result = model.transcribe("audio.mp3")
# print(result["text"])