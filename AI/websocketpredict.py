import asyncio
import websockets
import pyaudio
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import threading
import queue
import paho.mqtt.publish as publish
import datetime
import time

model = load_model("final_model3.h5")

SR = 44100
TRACK_DURATION = 10  # measured in seconds
samples_per_segment = int(SR / 10)
hop_length = 512
num_mfcc_vectors_per_segment = 9  # Set to match training setup
num_mfcc = 20
labels = ["Tenang", "Berisik"]
interval = 2
time_last = datetime.datetime.now()
prediction_result = 0
last_prediction = 0
# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SR,
                output=True)

# Thread-safe queue to pass audio data
audio_queue = queue.Queue()

def play_audio():
    global last_prediction
    global prediction_result
    global time_last
    while True:
        if prediction_result != last_prediction:
            if prediction_result == 1:
                print("Berisik")
                publish.single("/sic5/kelompok13/prediction", "Berisik", hostname="broker.hivemq.com")
                time.sleep(2)
            else:
                print("Tenang")
                publish.single("/sic5/kelompok13/prediction", "Tenang", hostname="broker.hivemq.com")
            last_prediction = prediction_result
        else:
            time_now = datetime.datetime.now()
            if (time_now - time_last).total_seconds() > interval:
                if prediction_result == 1:
                    print("Berisik")
                    publish.single("/sic5/kelompok13/prediction", "Berisik", hostname="broker.hivemq.com")
                else:
                    print("Tenang")
                    publish.single("/sic5/kelompok13/prediction", "Tenang", hostname="broker.hivemq.com")
                time_last = time_now


        data = audio_queue.get()
        if data is None:  # Stop signal
            break
        # stream.write(data)

def process_audio():
    global prediction_result
    data_audio = np.zeros(samples_per_segment, dtype=np.float32)
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:  # Stop signal
            break

        audio_float = audio_chunk.astype(np.float32) / 32768.0
        data_audio = np.concatenate((data_audio[len(audio_float):], audio_float))

        if len(data_audio) == samples_per_segment:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=data_audio, sr=SR, n_mfcc=num_mfcc, n_fft=2048, hop_length=hop_length)
            mfcc = mfcc.T
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

            # Ensure mfcc shape is (9, 20) to match model input
            if mfcc.shape[0] >= num_mfcc_vectors_per_segment:
                mfcc = mfcc[:num_mfcc_vectors_per_segment, :]  # Trim to match shape
                mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
                
                # Print the shape for debugging
                

                # Predict
                prediction = model.predict(mfcc)
                prediction_result = np.argmax(prediction)
                if np.argmax(prediction) == 1:
                    print(f"MFCC shape: {mfcc.shape}")
                    print(f"Prediction: {prediction}")

                    # Print the predicted label
                    print(labels[np.argmax(prediction)])

async def listen_to_server():
    uri = "ws://103.84.207.23:8888"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello Server!")
        
        while True:
            try:
                message = await websocket.recv()

                if isinstance(message, str):
                    message = message.encode('utf-8')

                if len(message) % 2 != 0:
                    print("Received incomplete data. Skipping...")
                    continue

                data = np.frombuffer(message, dtype=np.int16)
                audio_queue.put(data)
                
            except websockets.ConnectionClosed:
                print("Connection closed")
                break
            except ValueError as e:
                print(f"ValueError: {e}")
                continue
                
        audio_queue.put(None)  # Send stop signal to processing thread
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    play_thread = threading.Thread(target=play_audio)
    process_thread = threading.Thread(target=process_audio)

    play_thread.start()
    process_thread.start()

    asyncio.run(listen_to_server())

    play_thread.join()
    process_thread.join()
