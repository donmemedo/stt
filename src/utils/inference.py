from src.utils.commons import get_model, get_tensor, get_models

class_names = [
    "angry",
    "happy",
    "neutral",
    "sad",
    "surprised",
    "fearful",
    # "calm",
    # "disgust",
]
model = get_model()
model121 = get_models('models/SER_densenet121.pt',121)
model201 = get_models('models/SER_densenet201.pt',201)


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
def predicto(model_type,image_bytes):
    tensor = get_tensor(image_bytes)
    if model_type == '121':
        outputs = model121(tensor)
    elif model_type == '201':
        outputs = model201(tensor)
    _, prediction = outputs.max(1)
    category = prediction.item()
    emotion = class_names[category]

    return emotion
