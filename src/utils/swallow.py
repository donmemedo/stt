import numpy as np
from scipy.signal import butter, lfilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import librosa


# Load accelerometer data
def load_data(file_path):
    data = np.loadtxt(file_path)
    return data


# Filter data to remove noise
def filter_data(data, low_pass=2, high_pass=20, sampling_rate=2250):
    nyq = 0.5 * sampling_rate
    low = low_pass / nyq
    high = high_pass / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)
    return filtered_data


# Extract features from filtered data
def extract_features(filtered_data, window_size=200, overlap=0.5):
    features = []
    for i in range(0, len(filtered_data), int(window_size * (1 - overlap))):
        window = filtered_data[i:i + window_size]
        mean = np.mean(window)
        std = np.std(window)
        features.append([mean, std])
    return np.array(features)


# Train a random forest classifier to detect swallows
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # np.nan_to_num(X_train)
    model.fit(X_train, y_train)
    return model


# Detect swallows using the trained model

data, sr = librosa.load('/media/makhataei/Backups/555/happy.wav', sr=2500)
filtered_data = filter_data(data)
features = extract_features(filtered_data)
# labels = np.zeros(len(features))  # assume no swallows initially
d_normal = 10
labels = np.random.randint(0, d_normal + 1, len(features))  # assume no swallows initially
labels = np.floor(labels / d_normal)
model = train_model(features, labels)


def detect_swallow(path):
    data, sr = librosa.load(path, sr=2500)
    filtered_data = filter_data(data)
    features = extract_features(filtered_data)
    predictions = model.predict(features)
    swallow_times = []
    for i in range(len(predictions)):
        if predictions[i]:
            swallow_times.append(int(i * 4) / 100)

    # predictions = detect_swallow(model, features)

    return swallow_times

# detect_swallow('/media/makhataei/Backups/555/angry3.wav')
