from tensorflow.keras.models import load_model
import dearpygui.dearpygui as dpg
import sounddevice as sd
import numpy as np
import threading
import queue
import librosa
from scipy.io.wavfile import write
import soundfile as sf

LABELS = ['Cough', 'Noise', 'Voices']

class AudioPlotter:
    def __init__(self):
        self.sample_rate = 44100  # Частота дискретизации
        self.audio_data = np.zeros((0,))  # Массив для хранения аудиоданных
        self.data_queue = queue.Queue()  # Очередь для передачи данных между потоками
        self.recording = False  # Флаг записи
        self.display_interval = 1
        self.counter = 1
        # self.chunk_size = 1024
        # self.chunk_size = self.sample_rate
        self.chunk_size = int(self.sample_rate * self.display_interval)
        print(self.chunk_size)
        
        # Создание потока для захвата аудиосигнала
        self.stream = sd.InputStream(samplerate=self.sample_rate,
                                      channels=1,
                                      callback=self.audio_callback,
                                      blocksize=self.chunk_size)
        self.model = load_model('cough_detection_model-37_13_03.h5')
        
        # Инициализация интерфейса
        self.setup_gui()
        
        # Запуск потока для обновления графика
        self.update_thread = threading.Thread(target=self.update_plot, daemon=True)
        self.update_thread.start()
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.recording:
            self.audio_data = np.append(self.audio_data, indata.flatten())  # Сохраняем данные в массив
            self.data_queue.put(indata.flatten())  # Помещаем данные в очередь
            self.make_prediction(indata)  # Выполняем предсказание
            # раскомментить, если нужно зписать аудио по секунде
            # self.counter += 1
            # self.save_audio_data(indata)

    def update_plot(self):
        while True:
            if not self.data_queue.empty():
                audio_data = self.data_queue.get()
                # Обновляем график
                dpg.set_value("audio_plot_y", audio_data)

    def setup_gui(self):
        dpg.create_context()

        with dpg.window(label="Audio Plot", width=800, height=600):
            dpg.add_button(label="Start Recording", callback=self.start_recording)
            dpg.add_button(label="Stop Recording", callback=self.stop_recording)
            dpg.add_button(label="Save to File", callback=self.save_to_file)
            
            # Метка для отображения предсказания
            dpg.add_text(label="Prediction:", tag="prediction_label")

            # Создаем график
            with dpg.plot(label="Microphone Input", height=-1, width=-1):
                dpg.add_plot_legend()  # Добавляем легенду внутри графика
                x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="Samples")
                y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude")

                # Создаем линию для графика
                dpg.add_line_series(x=np.arange(self.chunk_size), y=np.zeros(self.chunk_size), label="Audio Signal", tag="audio_plot_y", parent=y_axis)

        dpg.create_viewport(title='Audio Plotter', width=800, height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def save_audio_data(self, data):
        output_filename = "output-frames.wav"
        audio_data = data
        
        audio_data = (audio_data * 32767).astype(np.int16)

        write(f"output-frames_{self.counter}.wav", self.sample_rate, audio_data)
        print(f"Запись сохранена как output-frames_{self.counter}\n")

    def start_recording(self):
        print('Start recording')
        self.recording = True  # Устанавливаем флаг записи
        self.audio_data = np.zeros((0,))  # Сбрасываем массив аудиоданных
        self.stream.start()  # Запускаем поток

    def stop_recording(self):
        print('Stop recording')
        self.recording = False  # Сбрасываем флаг записи
        self.stream.stop()  # Останавливаем поток

    def make_prediction(self, data):
        audio_segment = data.flatten()
        print(len(audio_segment))
        try:
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13) # n_fft=int(self.chunk_size)

            # audio, sample_rate = librosa.load('output-frames_12.wav', sr=None)
            # print(sample_rate)
            # print(self.sample_rate)
            # mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

            mfccs_mean = np.mean(mfccs.T, axis=0)
            mfccs_reshaped = mfccs_mean.reshape(1, 1, 13, 1)
            prediction = self.model.predict(mfccs_reshaped)
            # print(mfccs_reshaped)
            predicted_label = LABELS[np.argmax(prediction)]
            print(predicted_label)
            dpg.set_value("prediction_label", f"Prediction: {predicted_label}")

        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
            dpg.set_value("prediction_label", f"Ошибка предсказания: {e}")

    def save_to_file(self):
        if self.audio_data.size > 0:
            sf.write('output.wav', self.audio_data, self.sample_rate)  # Сохраняем аудио в файл
            print("Audio saved to output.wav")
        else:
            print("No audio data to save.")

if __name__ == "__main__":
    plotter = AudioPlotter()
