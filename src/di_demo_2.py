import torch
from deep_speaker import DeepSpeakerModel
from feature_extraction import extract_features

speech_file = "../test/1/4527484.wav"

# Load the pre-trained DeepSpeaker model
model = DeepSpeakerModel()
model.load_state_dict(torch.load('deepspeaker_checkpoint.pth'))
model.eval()

# Extract features from the input audio
audio_file_path = speech_file# 'path/to/your/audiofile.wav'
features = extract_features(audio_file_path)

# Pass the features through the model to obtain speaker labels
with torch.no_grad():
    labels = model(features)
print(labels)
# Process the labels to obtain model speaker( difeaturesar)ization

# Print the information
# speaker_speakerlabels = labels
# process_printlabels(labels)
