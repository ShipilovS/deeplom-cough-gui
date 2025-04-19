from keras.models import load_model
import tkinter as tk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import tensorflow as tf
from scipy.io.wavfile import write
import signal

LABELS = ['Кашель', 'Шум', 'Голоса']

class AudioRecorder:
    def __init__(self, master):
        self.master = master
        self.master.title("Audio Recorder")

        print("Загрузка модели")
        self.model = load_model('cough_detection_model_tfmfcc.h5')

        # Параметры для извлечения признаков (должны соответствовать обучению модели)
        self.sample_rate = 44100
        self.n_mfcc = 25
        self.frame_length = 1024
        self.frame_step = 512
        self.num_mel_bins = 40
        self.lower_edge_hertz = 0.0
        self.upper_edge_hertz = 8000.0

        self.is_recording = False
        self.record_duration = 1  # 1 секунда
        self.chunk_size = self.sample_rate
        self.counter = 1
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
                data = stream.read(int(self.sample_rate * self.record_duration))[0]
                self.update_plot(data)
                self.frames.append(data)
                self.make_prediction(data)
                self.counter += 1
                self.save_audio_data(data)

    def update_plot(self, data):
        fft_data = np.abs(np.fft.fft(data.flatten()))[:1024]
        self.fft_data = fft_data / np.max(fft_data)  # Нормализация

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

    def save_audio_data(self, data):
        audio_data = (data * 32767).astype(np.int16)
        write(f"output-frames_{self.counter}.wav", self.sample_rate, audio_data)
        print(f"Запись сохранена как output-frames_{self.counter}\n")

    def extract_features_tf(self, audio):
        # Преобразуем numpy массив в тензор TensorFlow
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        # Нормализация
        audio = audio - tf.math.reduce_mean(audio)
        
        # STFT
        stft = tf.signal.stft(audio, 
                             frame_length=self.frame_length, 
                             frame_step=self.frame_step)
        spectrogram = tf.abs(stft)
        
        # Mel-спектрограмма
        num_spectrogram_bins = stft.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.lower_edge_hertz,
            upper_edge_hertz=self.upper_edge_hertz)
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        
        # Логарифмирование
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        
        # MFCC
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.n_mfcc]  # Берем только первые n_mfcc коэффициентов
        
        # Усреднение по времени
        mfccs_mean = tf.math.reduce_mean(mfccs, axis=0)
        
        return mfccs_mean.numpy()

    def make_prediction(self, data):
        try:
            # Извлекаем признаки тем же способом, что и при обучении
            features = self.extract_features_tf(data.flatten())
            
            # Подготавливаем данные для модели (как при обучении)
            features_reshaped = features.reshape(1, 1, self.n_mfcc, 1)
            
            # Делаем предсказание
            prediction = self.model.predict(features_reshaped)
            predicted_label = LABELS[np.argmax(prediction)]
            print(f"Предсказание: {predicted_label} (вероятности: {prediction[0]})")
            
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorder(root)
    root.mainloop()