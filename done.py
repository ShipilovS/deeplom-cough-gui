
from tensorflow.keras.models import load_model
import dearpygui.dearpygui as dpg
import sounddevice as sd
import numpy as np
import threading
import queue, librosa
import soundfile as sf
LABELS = ['Cough', 'noise', 'voices']

class AudioPlotter:
    def __init__(self):
        self.sample_rate = 44100  # Частота дискретизации
        self.chunk_size = 1024     # Размер блока
        self.audio_data = np.zeros((0,))  # Массив для хранения аудиоданных
        self.data_queue = queue.Queue()  # Очередь для передачи данных между потоками
        self.recording = False  # Флаг записи
        
        # Создание потока для захвата аудиосигнала
        self.stream = sd.InputStream(samplerate=self.sample_rate,
                                      channels=1,
                                      callback=self.audio_callback,
                                      blocksize=self.chunk_size)
        self.model = load_model('cough_detection_model-37.h5')
        
        # Инициализация интерфейса
        self.setup_gui()
        
        # Запуск потока для обновления графика
        self.update_thread = threading.Thread(target=self.update_plot, daemon=True)
        self.update_thread.start()
        self.recording_en = True
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.recording:
            self.audio_data = np.append(self.audio_data, indata.flatten())  # Сохраняем данные в массив
        self.data_queue.put(indata.flatten())  # Помещаем данные в очередь

    def update_plot(self):
        while True:
            if not self.data_queue.empty():
                audio_data = self.data_queue.get()
                print(audio_data)
                # Обновляем график
                dpg.set_value("audio_plot_y", audio_data)

            
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


    def setup_gui(self):
        dpg.create_context()

        with dpg.window(label="Audio Plot", width=800, height=600):
            dpg.add_button(label="Start Recording", callback=self.start_recording)
            dpg.add_button(label="Stop Recording", callback=self.stop_recording)
            dpg.add_button(label="Save to File", callback=self.save_to_file)

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

    def start_recording(self):
        print('start')
        self.recording = True  # Устанавливаем флаг записи
        self.audio_data = np.zeros((0,))  # Сбрасываем массив аудиоданных
        self.stream.start()  # Запускаем поток

    def stop_recording(self):
        self.is_recording = False
        print("* done recording")
        self.save_audio()
        dpg.configure_item("record_button", enabled=True)  # Enable the button
        dpg.configure_item("stop_button", enabled=False)  # Enable the button
        
    def stop_recording(self):
        print('Stop')
        self.recording = False  # Сбрасываем флаг записи
        self.stream.stop()  # Останавливаем поток

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

    def save_to_file(self):
        if self.audio_data.size > 0:
            sf.write('output.wav', self.audio_data, self.sample_rate)  # Сохраняем аудио в файл
            print("Audio saved to output.wav")
        else:
            print("No audio data to save.")

if __name__ == "__main__":
    plotter = AudioPlotter()
