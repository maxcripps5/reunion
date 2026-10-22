import time
import json
import base64
import keyboard
import numpy as np
import sounddevice as sd
import speech_recognition as sr

from openai import OpenAI
from sk import my_sk
from datetime import datetime


#Datetime
def getTime():
    return datetime.now().strftime("%H:%M:%S")

#VB-Cable output device
TTS_OUTPUT_DEVICE = 7
TTS_SAMPLE_RATE = 24000

#Connect to OpenAI
client = OpenAI(api_key=my_sk)


#Mic calibration
recognizer = sr.Recognizer()
recognizer.dynamic_energy_threshold = False


#Speech to text and translation (English to Mandarin)
def get_input():
    try:
        audio_chunks = []

        with sr.Microphone() as source:
            print(getTime(), ": Hold F8 to speak...")

            #Record while F8 is held down
            while keyboard.is_pressed("F8"):
                audio_chunks.append(
                    recognizer.record(source, duration=0.2)
                )

        if not audio_chunks:
            print(getTime(), ": No audio recorded.")
            return "", ""

        #Combine recorded microphone audio
        audio = sr.AudioData(
            b"".join(chunk.frame_data for chunk in audio_chunks),
            audio_chunks[0].sample_rate,
            audio_chunks[0].sample_width
        )

        #Convert audio to WAV data
        wav_data = audio.get_wav_data()
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
        audio = np.frombuffer(audio_bytes, dtype=np.int16)

        if len(audio) == 0:
            print(getTime(), ": TTS returned empty audio.")
            return

        #Play Mandarin audio and wait until playback finishes
        sd.play(audio, 
            samplerate=TTS_SAMPLE_RATE,
            device=TTS_OUTPUT_DEVICE
        )
        sd.wait()

        print(
            f"Speech generation and playback took "
            f"{time.perf_counter() - start:.2f}s"
        )

    #Exception handling
    except Exception as e:
        print(getTime(), ": TTS error:", repr(e))


#Microphone calibration
try:
    with sr.Microphone() as source:
        print(getTime(), ": Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=0.3)

    print(
        getTime(),
        ": Reunion ready. Hold F8 to speak; release to translate."
    )

except Exception as e:
    print(getTime(), ": Microphone setup error:", e)
    raise


#Main translation loop
while True:
    try:
        #Wait for F8 to be pressed
        keyboard.wait("F8")
        time.sleep(0.1)

        #Get transcription and translation together
        recognized_text, translated_text = get_input()

        if not recognized_text or not translated_text:
            continue

        #Convert Mandarin text to speech
        speak_mandarin(translated_text)

        #Wait until F8 is released before accepting another recording
        while keyboard.is_pressed("F8"):
            time.sleep(0.05)

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