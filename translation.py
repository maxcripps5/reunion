#TRANSLATE - Transcription & Translation using OpenAI
import time
import json
import base64
import io
import wave
import queue
import numpy as np
import jieba
import re

from pypinyin import lazy_pinyin, Style

#Project files
import config
import audio
from utils import getTime


#Speech to text and translation (English to Mandarin)
def get_input(audio_chunks):
    try:
        if not audio_chunks:
            print(
                getTime(),
                ": No audio recorded. Skipping transcription."
            )
            return "", ""

        #Combine recorded microphone audio
        audio_data = np.concatenate(
            audio_chunks,
            axis=0
        ).flatten()

        #Skip recordings that are too short
        duration = (
            len(audio_data) / config.AUDIO_SAMPLE_RATE
        )

        if duration < 0.35:
            print(
                getTime(),
                ": Recording is too short. Skipping transcription."
            )
            return "", ""

        #Skip audio that is effectively silent
        rms = float(
            np.sqrt(np.mean(audio_data ** 2))
        )

        if rms < config.SILENCE_VOL_THRESHOLD:
            print(
                getTime(),
                ": Recording is silent. Skipping transcription."
            )
            return "", ""

        #Convert float32 audio to 16-bit PCM
        audio_data = np.clip(
            audio_data,
            -1.0,
            1.0
        )

        audio_data = (
            audio_data * 32767
        ).astype(np.int16)

        #Convert audio to WAV format
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(config.AUDIO_SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())

        wav_data = wav_buffer.getvalue()

        encoded_audio = base64.b64encode(
            wav_data
        ).decode("utf-8")

        #Send audio for transcription and translation
        start = time.perf_counter()

        response = config.client.chat.completions.create(
            model="gpt-audio-1.5",
            modalities=["text"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Accurately transcribe the spoken English "
                        "in the audio. Only include words that are "
                        "clearly audible. Do not guess, invent, "
                        "complete, or infer missing words. Do not "
                        "interpret silence, background noise, or "
                        "unclear sounds as speech. If no clear speech "
                        "is audible, return an empty string for both "
                        "english and mandarin. "
                        "Translate the accurately transcribed English "
                        "into natural conversational Mandarin using "
                        "Simplified Chinese characters. Do not add "
                        "meaning that is not present in the English "
                        "transcription. "
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
            print(
                getTime(),
                ": OpenAI returned an empty response."
            )
            return "", ""

        #Parse the returned JSON
        result = json.loads(content)

        english_text = result.get("english", "").strip()
        mandarin_text = result.get("mandarin", "").strip()

        #Skip empty transcription results
        if not english_text or not mandarin_text:
            # print(
            #     getTime(),
            #     ": No usable speech detected. Skipping playback."
            # )
            return "", ""

        print(getTime(), ": YOU:", english_text)
        #print(getTime(), ": CN:", mandarin_text)

        print(
            f"Combined transcription and translation took "
            f"{time.perf_counter() - start:.2f}s"
        )

        return english_text, mandarin_text

    #Exception handling
    except Exception as e:
        print(
            getTime(),
            ": Combined audio request error:",
            repr(e)
        )
        return "", ""


#Transcribe incoming Mandarin and translate to English
def process_incoming_audio():
    while not audio.shutdown_event.is_set():
        try:
            item = audio.incoming_audio_queue.get(
                timeout=0.5
            )
        except queue.Empty:
            continue

        if item is None:
            audio.incoming_audio_queue.task_done()
            break

        audio_data, sample_rate = item

        try:
            #Skip incoming audio that is too short
            duration = (
                len(audio_data) / sample_rate
            )

            if duration < 0.35:
                print(
                    getTime(),
                    ": Incoming audio is too short. Skipping transcription."
                )
                continue

            #Skip incoming audio that is effectively silent
            rms = float(
                np.sqrt(np.mean(audio_data ** 2))
            )

            if rms < config.SILENCE_VOL_THRESHOLD:
                print(
                    getTime(),
                    ": Incoming audio is silent. Skipping transcription."
                )
                continue

            wav_data = audio.create_wav(
                audio_data,
                sample_rate
            )

            encoded_audio = base64.b64encode(
                wav_data
            ).decode("utf-8")

            #Send incoming audio for transcription and translation
            start = time.perf_counter()

            response = config.client.chat.completions.create(
                model="gpt-audio-1.5",
                modalities=["text"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Listen for clearly audible spoken Mandarin. "
                            "Transcribe only words that are clearly "
                            "spoken using Simplified Chinese characters. "
                            "Do not guess, invent, complete, or infer "
                            "missing words. Do not interpret silence, "
                            "background noise, or unclear sounds as "
                            "speech. If no clear Mandarin speech "
                            "is audible, return an empty string for both "
                            "mandarin and english. "
                            "Translate the accurate Mandarin "
                            "transcription into natural English. "
                            "Do not add meaning that is not present "
                            "in the Mandarin transcription. "
                            "Return only valid JSON with string fields "
                            "named mandarin and english."
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
                print(
                    getTime(),
                    ": Empty incoming translation response."
                )
                continue

            #Parse the returned JSON
            result = json.loads(content)

            mandarin_text = result.get("mandarin", "").strip()
            english_text = result.get("english", "").strip()

            #Skip empty transcription results
            if not mandarin_text or not english_text:
                # print(
                #     getTime(),
                #     ": No usable incoming speech detected. Skipping translation."
                # )
                continue

            #Convert Mandarin transcription to Pinyin and use Jieba to keep the words intact
            words = jieba.cut(mandarin_text)

            mandarin_pinyin = " ".join(
                "".join(lazy_pinyin(word, style=Style.TONE))
                for word in words
            )

            #Remove spaces before punctuation
            mandarin_pinyin = re.sub(r"\s+([，。！？；：])", r"\1", mandarin_pinyin)

            if english_text:
                print(
                    getTime(),
                    ": CALLER:",
                    english_text
                )

                print(
                    getTime(),
                    ": PINYIN:",
                    mandarin_pinyin
                )

                #Disabled feature - English TTS
                #speech.speak_english(english_text)

            #Log incoming translation processing time
            print(
                f"Incoming transcription and translation took "
                f"{time.perf_counter() - start:.2f}s"
            )

        except Exception as e:
            print(
                getTime(),
                ": Incoming translation error:",
                repr(e)
            )

        finally:
            audio.incoming_audio_queue.task_done()