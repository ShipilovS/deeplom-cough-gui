import dearpygui.dearpygui as dpg
import numpy as np
import sounddevice as sd
import threading
import time

class AudioVisualizer:
    def __init__(self):
        self.sample_rate = 44100
        self.chunk_size = 2048
        self.is_running = False  # Start with recording off
        self.data = np.array([]) # Инициализация как пустой массив NumPy
        self.x_data = list(np.arange(self.chunk_size))  # Create x_data
        self.stream = None  # Initialize stream to None

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.data = indata[:, 0] # Присваиваем новые данные массиву data

    def start_recording(self):
        if not self.is_running:  # Prevent multiple streams
            try:
                print(sd.query_devices())
                self.stream = sd.InputStream(samplerate=self.sample_rate,
                                           channels=1,
                                           callback=self.audio_callback,
                                           blocksize=self.chunk_size)

                self.stream.start()
                self.is_running = True
                print("Audio stream started")
                dpg.configure_item("start_button", enabled=False)  # Disable start button
                dpg.configure_item("stop_button", enabled=True)  # Enable stop button
            except Exception as e:
                print(f"Error starting audio stream: {e}")

    def stop_recording(self):
        if self.is_running:
            try:
                self.stream.stop()
                self.stream.close()
                self.is_running = False
                print("Audio stream stopped")
                dpg.configure_item("start_button", enabled=True)  # Enable start button
                dpg.configure_item("stop_button", enabled=False)  # Disable stop button
            except Exception as e:
                print(f"Error stopping audio stream: {e}")

    def update_plot(self):
        try:
            dpg.set_value("audio_series", list(self.data))
        except Exception as e:
            print(f"Error updating plot: {e}")

    def update_loop(self):
        while self.is_running:
            self.update_plot()
            time.sleep(0.01)

# Initialize Dear PyGui
dpg.create_context()

# Create the visualizer object
visualizer = AudioVisualizer()

# Create Dear PyGui window
with dpg.window(label="Microphone Audio", width=800, height=600):
    dpg.add_text("Audio from Microphone")

    dpg.add_button(label="Start Recording", callback=visualizer.start_recording, tag="start_button")
    dpg.add_button(label="Stop Recording", callback=visualizer.stop_recording, tag="stop_button", enabled=False)  # Initially disabled

    with dpg.plot(label="Audio Waveform", height=300, width=700):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time")
        dpg.add_plot_axis(dpg.mvYAxis, label="Amplitude", tag="y_axis")

        # Создаем график с нулевыми данными (пустой график)
        initial_data = []  # Пустой список
        dpg.add_line_series(visualizer.x_data[:0], initial_data, label="Audio Data", tag="audio_series", parent="y_axis")

# Start the update loop in a separate thread
def run_update_loop():
    visualizer.update_loop()

update_thread = threading.Thread(target=run_update_loop)
update_thread.daemon = True
update_thread.start()

# Dear PyGui settings
dpg.create_viewport(title='Microphone Audio', width=800, height=600)
dpg.setup_dearpygui()
dpg.show_viewport()

# Start Dear PyGui
dpg.start_dearpygui()

# Dear PyGui cleanup
dpg.destroy_context()