# python3 -m pip install numpy
# python3 main4.py
from keras.models import load_model
import tkinter as tk
import numpy as np
import wave
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import librosa, signal
import time
# model = load_model('cough_detection_model-37.h5')
from scipy.io.wavfile import write


LABELS = ['Кашель', 'Нет кашля', 'Шум']  
LABELS = ['VGMU_Coswata_Cutt', 'VGMU_Noise', 'VGMU_Voises_Cutt']
# LABELS = ['VGMU_Coswata_Cutt', 'VGMU_Noise', 'VGMU_Voises_Cutt']


class AudioRecorder:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Recorder")

        print(f"Загрузка модели")

        self.model = load_model('cough_detection_model-37.h5')

        self.is_recording = False
        self.filename = 'output.wav'
        self.sample_rate = 44100
        self.record_duration = 0.5
        self.chunk_size = 1024
        self.frames = []

        # Кнопки
        self.record_button = tk.Button(master, text="Записать", command=self.start_recording)
        self.record_button.pack(pady=10)

        self.stop_button = tk.Button(master, text="Стоп", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        # График
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, self.sample_rate / 2)
        self.ax.set_title("Уровень звука")
        self.ax.set_xlabel("Частота (Гц)")
        self.ax.set_ylabel("Амплитуда")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.fft_data = []
        self.fft_line, = self.ax.plot([], [], lw=2, color='red')

        signal.signal(signal.SIGINT, self.handle_exit)

    def handle_exit(self, signum, frame):
        print("Получен сигнал завершения. Остановка записи...")
        self.stop_recording()
        self.master.quit()

    def start_recording(self):
        self.is_recording = True
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        self.thread = threading.Thread(target=self.record)
        self.thread.start()

    def record(self):
        print("* recording")
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
            print('Записываю')
            while self.is_recording:
                # data = stream.read(self.chunk_size)[0]
                data = stream.read(int(self.sample_rate * self.record_duration))[0]
                self.update_plot(data)
                self.frames.append(data)
                self.make_prediction(data)

    def update_plot(self, data):
        # Преобразование в массив NumPy и вычисление FFT
        fft_data = np.abs(np.fft.fft(data.flatten()))[:1024]
        self.fft_data = fft_data / np.max(fft_data)  # Нормализация

        # Обновление графика
        self.line.set_xdata(np.linspace(0, self.sample_rate / 2, len(self.fft_data)))
        self.line.set_ydata(self.fft_data)
        self.ax.draw_artist(self.line)
        self.canvas.draw_idle()

    def stop_recording(self):
        print("* done recording")
        self.is_recording = False
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_audio()

    def save_audio(self):
        output_filename = "output-frames.wav"
        audio_data = np.concatenate(self.frames).flatten() 
        
        audio_data = (audio_data * 32767).astype(np.int16)

        write(output_filename, self.sample_rate, audio_data)
        print(f"Запись сохранена в {output_filename}")

    def make_prediction(self, data):
        audio_segment = data.flatten()

        mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13, n_fft=self.chunk_size)
        
        mfccs = mfccs[:, 0].reshape(1, 1, 13, 1)

        prediction = self.model.predict(mfccs)
        predicted_label = LABELS[np.argmax(prediction)] 

        print(f"Предсказание: {predicted_label}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()
