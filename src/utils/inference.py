from src.utils.commons import get_model, get_tensor, get_models

class_names = [
    "angry",
    # "calm",
    # "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]
model = get_model()
model121 = get_models('/home/makhataei/Projects/CallCanterQC/models/SER_densenet123.pt',121)
model201 = get_models('/home/makhataei/Projects/CallCanterQC/models/SER_densenet203.pt',201)


def prediction(image_bytes):
    tensor = get_tensor(image_bytes)
    outputs = model(tensor)
    _, prediction = outputs.max(1)
    category = prediction.item()
    emotion = class_names[category]

    return emotion
def prediction201(image_bytes):
    tensor = get_tensor(image_bytes)
    outputs = model201(tensor)
    _, prediction = outputs.max(1)
    category = prediction.item()
    emotion = class_names[category]

    return emotion
def prediction121(image_bytes):
    tensor = get_tensor(image_bytes)
    outputs = model121(tensor)
    _, prediction = outputs.max(1)
    category = prediction.item()
    emotion = class_names[category]

    return emotion
