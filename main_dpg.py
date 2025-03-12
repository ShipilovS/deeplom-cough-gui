import dearpygui.dearpygui as dpg
import numpy as np
import sounddevice as sd
import threading
import scipy.signal as signal
import librosa
from librosa.feature import mfcc
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write

LABELS = ['Cough', 'noise', 'voices']


class AudioRecorder:
    def __init__(self):
        print(f"Загрузка модели")
        self.model = load_model('cough_detection_model-37.h5')

        self.is_recording = False
        self.filename = 'output.wav'
        self.sample_rate = 44100
        self.record_duration = 1  # 1 сек мб
        self.chunk_size = 2048
        self.counter = 1
        self.frames = []
        self.fft_data = np.zeros(self.chunk_size // 2)  # Инициализация для графика
        self.prediction_text = "Waiting..."
        self.recording_enabled = True

    def start_recording(self):
        if self.recording_enabled:
            self.is_recording = True
            self.frames = []  # Очистить frames перед началом новой записи
            self.thread = threading.Thread(target=self.record)
            self.thread.start()
            dpg.configure_item("record_button", enabled=False)  # Disable the button
            dpg.configure_item("stop_button", enabled=True)  # Enable the button
        else:
            print("Запись отключена, сначала остановите текущую.")

    def stop_recording(self):
        self.is_recording = False
        print("* done recording")
        self.save_audio()
        dpg.configure_item("record_button", enabled=True)  # Enable the button
        dpg.configure_item("stop_button", enabled=False)  # Enable the button

    def record(self):
        print("* recording")
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            print('Записываю')
            while self.is_recording:
                data = stream.read(self.chunk_size)[0]
                self.update_plot(data)
                self.frames.append(data)
                self.make_prediction(data)
                # self.counter += 1
        print("Запись остановлена.") # Вывод после цикла записи

    def update_plot(self, data):
        """Обновляет данные графика FFT в Dear PyGui."""
        try:
            # Вычисляем новый размер для FFT
            fft_size = self.chunk_size

            # Преобразование в массив NumPy и вычисление FFT
            fft_data = np.abs(np.fft.fft(data.flatten()))[:fft_size]

            # Проверка на нулевой максимум перед нормализацией
            if np.max(fft_data) == 0:
                print("Предупреждение: Максимум FFT равен 0. Невозможно нормализовать.")
                self.fft_data = np.zeros_like(fft_data)  # Заполнить нулями, чтобы избежать ошибок
            else:
                self.fft_data = fft_data / np.max(fft_data)  # Нормализация

            # Преобразование в список и обновление данных графика Dear PyGui
            dpg.set_value("fft_series", list(self.fft_data))

        except Exception as e:
            print(f"Ошибка при обновлении графика: {e}")

    def make_prediction(self, data):
        audio_segment = data.flatten()
        try:
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13,
                                         n_fft=int(self.chunk_size))
            mfccs = mfccs[:, 0].reshape(1, 1, 13, 1)

            prediction = self.model.predict(mfccs)
            predicted_label = LABELS[np.argmax(prediction)]
            self.prediction_text = f"{predicted_label}"
            print(predicted_label)
            dpg.set_value("prediction_label", self.prediction_text)

        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            self.prediction_text = f"Ошибка предсказания {e}"
            dpg.set_value("prediction_label", self.prediction_text)

    def save_audio(self):
         if len(self.frames) > 0:
            output_filename = "output-frames.wav"
            audio_data = np.concatenate(self.frames).flatten()

            audio_data = (audio_data * 32767).astype(np.int16)

            write(output_filename, self.sample_rate, audio_data)
            print(f"Запись сохранена в {output_filename}")

    def save_audio_data(self, data):
        output_filename = f"output-frames_{self.counter}.wav"
        audio_data = data
        audio_data = (audio_data * 32767).astype(np.int16)

        write(output_filename, self.sample_rate, audio_data)
        print(f"Запись сохранена как {output_filename}\n")


# Инициализация Dear PyGui
dpg.create_context()

# Создание объекта AudioRecorder
recorder = AudioRecorder()

with dpg.window(label="Audio Recorder", width=800, height=600):
    dpg.add_text("Уровень звука")

    # Создание данных для графика (инициализация нулями)
    x_data = list(np.linspace(0, recorder.sample_rate / 2, recorder.chunk_size // 2))
    y_data = list(np.zeros(recorder.chunk_size // 2))

    with dpg.plot(label="FFT", height=300, width=700):
        dpg.add_plot_axis(dpg.mvXAxis, label="Частота (Гц)")
        dpg.add_plot_axis(dpg.mvYAxis, label="Амплитуда", tag="y_axis")
        print(x_data)
        print(y_data)
        dpg.add_line_series(x_data, y_data, label="FFT Data", tag="fft_series", parent="y_axis")

    dpg.add_button(label="Record", callback=recorder.start_recording, tag="record_button")
    dpg.add_button(label="Stop", callback=recorder.stop_recording, tag="stop_button", enabled=False)

    # Label для отображения предсказаний
    dpg.add_text(recorder.prediction_text, tag="prediction_label")

dpg.create_viewport(title='Audio Recorder', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()

# Запуск главного цикла Dear PyGui
dpg.start_dearpygui()

# Очистка ресурсов Dear PyGui
dpg.destroy_context()