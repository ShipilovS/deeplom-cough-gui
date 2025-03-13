import numpy as np
import librosa
import os, re
from tensorflow.keras.models import load_model

LABELS = ['VGMU_Coswata_Cutt', 'VGMU_Noise', 'VGMU_Voises_Cutt']

# Загрузка модели
model = load_model('cough_detection_model-37_13_03.h5')

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean


# # или так
test_file_path = './VGMU_Coswata_Cutt/8cough-heavy.wav' 
test_file_path = './VGMU_Noise/1-977-A-39.wav'
test_file_path = './voiseoutput_1.wav'
# test_file_path = '/home/sshipilov/Загрузки/Telegram Desktop/data_deeplom/VGMU_Coswata_Cutt/8cough-heavy.wav'
features = extract_features(test_file_path)
features = features.reshape(1, 1, 13, 1) 
predictions = model.predict(features)
predicted_class = np.argmax(predictions)
print(f"Предсказанный класс: {predicted_class} ({LABELS[predicted_class]})")

folder_path = '.'  # Текущая директория

# Найти все файлы output-frames_*
audio_files = [f for f in os.listdir(folder_path) if f.startswith('output-frames_') and f.endswith('.wav')]

def extract_number(filename):
    match = re.search(r'_(\d+)\.wav', filename)
    if match:
        return int(match.group(1))
    return float('inf') 


audio_files.sort(key=extract_number)

for file_name in audio_files:
    file_path = os.path.join(folder_path, file_name)

    try:
        features = extract_features(file_path)
        features = features.reshape(1, 1, 13, 1)
        predictions = model.predict(features)
        predicted_class = np.argmax(predictions)
        print(f"Файл: {file_name} - Предсказанный класс: {predicted_class} ({LABELS[predicted_class]})")
    except Exception as e:
        print(f"Ошибка обработки файла {file_name}: {e}")