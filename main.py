import time
import json
import base64
import keyboard
import io
import wave
import threading
import numpy as np
import sounddevice as sd
import speech_recognition as sr

from openai import OpenAI
from sk import my_sk
from datetime import datetime


#Datetime
def getTime():
    return datetime.now().strftime("%H:%M:%S")


#Audio devices
MIC_INPUT_DEVICE = None
TTS_OUTPUT_DEVICE = 7
AUDIO_SAMPLE_RATE = 48000
TTS_SAMPLE_RATE = 24000

#Audio buffers
recording_chunks = []
recording_lock = threading.Lock()
tts_buffer = np.zeros(0, dtype=np.float32)
tts_lock = threading.Lock()

#Mic calibration
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False

#Audio mixer
def audio_callback(indata, outdata, frames, time_info, status):
    global tts_buffer

    #Continous microphone passthrough
    outdata[:] = indata


    #Record microphone will F8 is held
    if keyboard.is_pressed("F8"):
        with recording_lock:
            recording_chunks.append(indata.copy())


    #Mix Mandarin TTS with microphone input
    with tts_lock:
        count = min(frames, len(tts_buffer))

        if count > 0:
            outdata[:count, 0] += tts_buffer[:count]
            tts_buffer = tts_buffer[count:]


    #Prevent audio clipping **REASSESS
    np.clip(outdata, -1.0, 1.0, out=outdata)

#Connect to OpenAI
client = OpenAI(api_key=my_sk)

#Speech to text and translation (English to Mandarin)
def get_input(audio_chunks):
    try:
        if not audio_chunks:
            print(getTime(), ": No audio recorded.")
            return "", ""


        #Combine shared input (TTS and mic stream)
        audio = np.concatenate(audio_chunks, axis=0).flatten()


        #Convert float32 audio to 16bit
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)


        #Convert audio to WAV
        audio_data = sr.AudioData(
            audio.tobytes(),
            AUDIO_SAMPLE_RATE,
            2
        )


        #Convert audio to WAV data
        wav_data = audio_data.get_wav_data()
        encoded_audio = base64.b64encode(wav_data).decode("utf-8")

        #Send audio for transcription and translation
        start = time.perf_counter()

        response = client.chat.completions.create(
            model="gpt-audio-1.5",
            modalities=["text"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Transcribe the spoken English accurately. "
                        "Translate it into natural conversational Mandarin "
                        "using Simplified Chinese characters. "
                        "Return only valid JSON with string fields "
                        "named english and mandarin."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ]
        )

        #Read transcription and translation
        content = response.choices[0].message.content

        if not content:
            print(getTime(), ": OpenAI returned an empty response.")
            return "", ""

        #Parse the returned JSON
        result = json.loads(content)

        english_text = result["english"].strip()
        mandarin_text = result["mandarin"].strip()

        print(getTime(), ": EN:", english_text)
        print(getTime(), ": CN:", mandarin_text)
        print(
            f"Combined transcription and translation took "
            f"{time.perf_counter() - start:.2f}s"
        )

        return english_text, mandarin_text

    #Exception handling
    except Exception as e:
        print(getTime(), ": Combined audio request error:", e)
        return "", ""


#Text to speech (Mandarin)
def speak_mandarin(text):
    try:
        start = time.perf_counter()

        #Generate Mandarin speech synchronously
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions=(
                "Speak Mandarin naturally with standard pronunciation."
            ),
            response_format="pcm"
        )

        #Read the returned PCM audio
        audio_bytes = response.content
        audio = np.frombuffer(audio_bytes, dtype=np.int16
        ).astype(np.float32) / 32768.0

        if len(audio) == 0:
            print(getTime(), ": TTS returned empty audio.")
            return

        #Resample from 24 kHz to 48 kHz
        old_positions = np.arange(len(audio))
        new_length = int(
            len(audio) * AUDIO_SAMPLE_RATE / TTS_SAMPLE_RATE
        )

        audio = np.interp(
            np.arange(new_length) * TTS_SAMPLE_RATE / AUDIO_SAMPLE_RATE,
            old_positions,
            audio
        ).astype(np.float32)

        #Queue Mandarin audio for the mixer
        global tts_buffer

        with tts_lock:
            tts_buffer = np.concatenate((tts_buffer, audio))


        print(
            f"Speech generation and playback took "
            f"{time.perf_counter() - start:.2f}s"
        )

    #Exception handling
    except Exception as e:
        print(getTime(), ": TTS error:", repr(e))


#Microphone calibration
try:
    print(getTime(), ": Checking microphone...")


    with sr.Microphone(device_index=MIC_INPUT_DEVICE) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)


    print(getTime(), ": Microphone calibration completed.")


except Exception as e:
    print(getTime(), ": Microphone setup error:", e)
    raise


#Start audio stream
try:
    audio_stream = sd.Stream(
        device=(MIC_INPUT_DEVICE, TTS_OUTPUT_DEVICE),
        samplerate=AUDIO_SAMPLE_RATE,
        channels=(1, 1),
        dtype="float32",
        callback=audio_callback
    )


    audio_stream.start()
    print(getTime(), ": Audio stream started.")


except Exception as e:
    print(getTime(), ": Audio stream error:", repr(e))
    raise


#Main translation loop
while True:
    try:
        #Clear previous recording
        with recording_lock:
            recording_chunks.clear()


        #Wait for F8 to be pressed
        keyboard.wait("F8")


        print(getTime(), "Recording... release F8 to translate.")


        #Wait until F8 is released before accepting another recording
        while keyboard.is_pressed("F8"):
            time.sleep(0.01)


        #Copy recorded audio
        with recording_lock:
            audio_chunks = recording_chunks.copy()
            recording_chunks.clear()


        #Get transcription and translation together
        recognized_text, translated_text = get_input(audio_chunks)


        if not recognized_text or not translated_text:
            continue


        #Generate Mandarin speech
        speak_mandarin(translated_text)


    #Exception handling
    except KeyboardInterrupt:
        print(getTime(), "Reunion stopped.")
        break

    except Exception as e:
        print(
            getTime(),
            ": Error during translation cycle:",
            repr(e)
        )