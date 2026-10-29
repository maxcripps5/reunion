#SPEECH - TTS
import time
import numpy as np
import sounddevice as sd

import config
import audio

from utils import getTime


#Text to speech (Mandarin)
def speak_mandarin(text):
    try:
        start = time.perf_counter()

        #Generate Mandarin speech
        response = config.client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak Mandarin naturally with standard pronunciation.",
            response_format="pcm",
            speed=1.25
        )

        #Read returned PCM audio
        audio_bytes = response.content

        audio_data = (
            np.frombuffer(
                audio_bytes,
                dtype=np.int16
            ).astype(np.float32)
            / 32768.0
        )

        if len(audio_data) == 0:
            print(getTime(), ": TTS returned empty audio.")
            return

        #Resample from 24 kHz to 48 kHz
        old_positions = np.arange(len(audio_data))

        new_length = int(
            len(audio_data)
            * config.AUDIO_SAMPLE_RATE
            / config.TTS_SAMPLE_RATE
        )

        audio_data = np.interp(
            np.arange(new_length)
            * config.TTS_SAMPLE_RATE
            / config.AUDIO_SAMPLE_RATE,
            old_positions,
            audio_data
        ).astype(np.float32)

        #Queue Mandarin audio for the mixer
        with audio.tts_lock:
            audio.tts_buffer = np.concatenate(
                (
                    audio.tts_buffer,
                    audio_data
                )
            )

        print(
            f"Speech generation and playback took "
            f"{time.perf_counter() - start:.2f}s"
        )

    #Exception handling
    except Exception as e:
        print(getTime(), ": TTS error:", repr(e))


#Text to speech (English)
def speak_english(text):
    try:
        response = config.client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak English clearly and naturally.",
            response_format="pcm",
            speed=1.25
        )

        audio_data = (
            np.frombuffer(
                response.content,
                dtype=np.int16
            ).astype(np.float32)
            / 32768.0
        )

        if len(audio_data) == 0:
            return

        old_positions = np.arange(len(audio_data))

        new_length = int(
            len(audio_data)
            * config.AUDIO_SAMPLE_RATE
            / config.TTS_SAMPLE_RATE
        )

        audio_data = np.interp(
            np.arange(new_length)
            * config.TTS_SAMPLE_RATE
            / config.AUDIO_SAMPLE_RATE,
            old_positions,
            audio_data
        ).astype(np.float32)

        sd.play(
            audio_data,
            samplerate=config.AUDIO_SAMPLE_RATE,
            device=None
        )

        sd.wait()

    except Exception as e:
        print(
            getTime(),
            ": English TTS error:",
            repr(e)
        )