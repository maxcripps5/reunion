
import time
import json
import base64
import io
import wave
import keyboard
import threading
import numpy as np
import sounddevice as sd
import soundcard as sc
import queue
import re
import warnings

from openai import OpenAI
from sk import my_sk
from datetime import datetime

#Suppress SoundCard errors prompts (temporary)
warnings.filterwarnings(
    "ignore",
    message="data discontinuity in recording",
    category=RuntimeWarning
)

#Datetime
def getTime():
    return datetime.now().strftime("%H:%M:%S")


#Audio devices
MIC_INPUT_DEVICE = None
TTS_OUTPUT_DEVICE = None
AUDIO_SAMPLE_RATE = 48000
TTS_SAMPLE_RATE = 24000

#Audio buffers
recording_active = False
recording_chunks = []
recording_lock = threading.Lock()
tts_buffer = np.zeros(0, dtype=np.float32)
tts_lock = threading.Lock()

#Incoming audio
INCOMING_SPEECH_THRESHOLD = 0.035
INCOMING_SILENCE_SECONDS = 0.75
INCOMING_MAX_SEGMENT_SECONDS = 10

incoming_audio_queue = queue.Queue(maxsize=5)
shutdown_event = threading.Event()

#Silence audio detection
SILENCE_VOL_THRESHOLD = 0.02


#Connect to OpenAI
client = OpenAI(api_key=my_sk)


#Automatically find the VB-CABLE playback device (which feeds its virtual microphone)
def find_virtual_microphone_output():
    devices = sd.query_devices()
    candidates = []

    for index, device in enumerate(devices):
        name = device["name"].lower()
        if device["max_output_channels"] <= 0:
            continue

        #VB-CABLE's playback side is named CABLE Input on Windows
        if "cable input" in name or ("vb-audio" in name and "cable" in name):
            candidates.append((index, device["name"]))

    if not candidates:
        available = "\n".join(
            f"{i}: {device['name']} (outputs: {device['max_output_channels']})"
            for i, device in enumerate(devices)
        )
        raise RuntimeError(
            "Could not automatically find VB-CABLE's playback device. "
            "Make sure VB-CABLE is installed and enabled. Available devices:\n"
            + available
        )

    index, name = candidates[0]
    print(getTime(), f": Automatically detected virtual microphone output: {name} (device {index})")
    return index


#Audio mixer and microphone recording
def audio_callback(indata, outdata, frames, time_info, status):
    global tts_buffer

    #Continuous microphone passthrough
    outdata[:] = indata

    #Record microphone while F8 is held
    if recording_active:
        with recording_lock:
            recording_chunks.append(indata.copy())

    #Mix Mandarin TTS with microphone input
    with tts_lock:
        count = min(frames, len(tts_buffer))

        if count > 0:
            outdata[:count, 0] += tts_buffer[:count]
            tts_buffer = tts_buffer[count:]

    #Prevent audio clipping
    np.clip(outdata, -1.0, 1.0, out=outdata)


#Speech to text and translation (English to Mandarin)
def get_input(audio_chunks):
    try:
        if not audio_chunks:
            print(getTime(), ": No audio recorded. Skipping transcription.")
            return "", ""

        #Combine recorded microphone audio
        audio = np.concatenate(audio_chunks, axis=0).flatten()

        #Skip audio that is effectively silent
        rms = float(np.sqrt(np.mean(audio ** 2)))

        if rms < SILENCE_VOL_THRESHOLD:
            print(getTime(), ": Recording is silent. Skipping transcription.")
            return "", ""

        #Convert float32 audio to 16-bit PCM
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)

        #Convert audio to WAV format
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(AUDIO_SAMPLE_RATE)
            wav_file.writeframes(audio.tobytes())

        wav_data = wav_buffer.getvalue()
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

        #Skip empty transcription results
        if not english_text or not mandarin_text:
            print(getTime(), ": No usable speech detected. Skipping playback.")
            return "", ""

        print(getTime(), ": EN:", english_text)
        print(getTime(), ": CN:", mandarin_text)
        print(
            f"Combined transcription and translation took "
            f"{time.perf_counter() - start:.2f}s"
        )

        return english_text, mandarin_text

    #Exception handling
    except Exception as e:
        print(getTime(), ": Combined audio request error:", repr(e))
        return "", ""


#Text to speech (Mandarin)
def speak_mandarin(text):
    try:
        start = time.perf_counter()

        #Generate Mandarin speech
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak Mandarin naturally with standard pronunciation.",
            response_format="pcm",
            speed=1.25
        )

        #Read returned PCM audio
        audio_bytes = response.content
        audio = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            / 32768.0
        )

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

    #Exception handlin
    except Exception as e:
        print(getTime(), ": TTS error:", repr(e))



#Text to speech (English)
def speak_english(text):
    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak English clearly and naturally.",
            response_format="pcm",
            speed=1.25
        )

        audio = (
            np.frombuffer(response.content, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        if len(audio) == 0:
            return

        old_positions = np.arange(len(audio))
        new_length = int(
            len(audio) * AUDIO_SAMPLE_RATE / TTS_SAMPLE_RATE
        )

        audio = np.interp(
            np.arange(new_length) * TTS_SAMPLE_RATE / AUDIO_SAMPLE_RATE,
            old_positions,
            audio
        ).astype(np.float32)

        sd.play(
            audio,
            samplerate=AUDIO_SAMPLE_RATE,
            device=None
        )
        sd.wait()

    except Exception as e:
        print(getTime(), ": English TTS error:", repr(e))



#Find the loopback microphone for the default playback device
def find_loopback_device():
    speaker = sc.default_speaker()

    if speaker is None:
        raise RuntimeError("No default playback device found.")

    speaker_name = speaker.name

    def normalize(name):
        return re.sub(r"[^a-z0-9]", "", name.lower())

    normalized_speaker = normalize(speaker_name)

    microphones = sc.all_microphones(include_loopback=True)

    loopbacks = [
        mic for mic in microphones
        if getattr(mic, "isloopback", False)
    ]

    #Try an exact device name match first
    for mic in loopbacks:
        if normalize(mic.name) == normalized_speaker:
            print(getTime(), ": Incoming audio device:", mic.name)
            return mic

    #Try a partial name match
    for mic in loopbacks:
        normalized_name = normalize(mic.name)

        if (
            normalized_speaker in normalized_name
            or normalized_name in normalized_speaker
        ):
            print(getTime(), ": Incoming audio device:", mic.name)
            return mic

    available = ", ".join(mic.name for mic in loopbacks)

    raise RuntimeError(
        f"Could not match '{speaker_name}' to a loopback device. "
        f"Available loopback devices: {available}"
    )



#Record incoming audio 
def incoming_audio_worker():
    try:
        device = find_loopback_device()
        sample_rate = int(device.samplerate) if hasattr(device, "samplerate") else 48000
        block_size = max(1, int(sample_rate * 0.25))

        print(getTime(), ": Listening for incoming speech...")

        with device.recorder(
            samplerate=sample_rate,
            channels=2
        ) as recorder:
            audio_buffer = []
            silence_duration = 0.0
            speech_detected = False

            while not shutdown_event.is_set():
                audio = recorder.record(
                    numframes=block_size
                )

                if audio.size == 0:
                    continue

                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)

                rms = float(np.sqrt(np.mean(audio ** 2)))
                duration = len(audio) / sample_rate

                if rms >= INCOMING_SPEECH_THRESHOLD:
                    speech_detected = True
                    silence_duration = 0.0
                    audio_buffer.append(audio.copy())

                elif speech_detected:
                    audio_buffer.append(audio.copy())
                    silence_duration += duration

                buffered_seconds = sum(
                    len(chunk) for chunk in audio_buffer
                ) / sample_rate

                if speech_detected and (
                    silence_duration >= INCOMING_SILENCE_SECONDS
                    or buffered_seconds >= INCOMING_MAX_SEGMENT_SECONDS
                ):
                    segment = np.concatenate(audio_buffer)

                    try:
                        incoming_audio_queue.put(
                            (segment, sample_rate),
                            timeout=1
                        )
                    except queue.Full:
                        print(getTime(), ": Incoming queue full; segment skipped.")

                    audio_buffer = []
                    silence_duration = 0.0
                    speech_detected = False

    except Exception as e:
        print(getTime(), ": Incoming recorder error:", repr(e))


#Convert incoming audio to WAV format
def create_wav(audio, sample_rate):
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767).astype(np.int16)

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

    return wav_buffer.getvalue()


#Transcribe incoming Mandarin and translate to English
def process_incoming_audio():
    while not shutdown_event.is_set():
        try:
            item = incoming_audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        if item is None:
            break

        audio, sample_rate = item

        try:
            wav_data = create_wav(audio, sample_rate)
            encoded_audio = base64.b64encode(wav_data).decode("utf-8")

            response = client.chat.completions.create(
                model="gpt-audio-1.5",
                modalities=["text"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "If the spoken language is Mandarin, transcribe it using Simplified Chinese "
                            "Translate it into natural English. "
                            "Return only valid JSON with string fields named mandarin and english. "
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

            content = response.choices[0].message.content

            if not content:
                print(getTime(), ": Empty incoming translation response.")
                continue

            result = json.loads(content)
            mandarin_text = result["mandarin"].strip()
            english_text = result["english"].strip()

            if mandarin_text:
                print(getTime(), ": Received CN:", mandarin_text)

            if english_text:
                print(getTime(), ": English translation:", english_text)
                #speak_english(english_text)


        except Exception as e:
            print(getTime(), ": Incoming translation error:", repr(e))

        finally:
            incoming_audio_queue.task_done()


#Start audio stream
try:
    #Send microphone passthrough and Mandarin TTS to the virtual microphone.
    #The computer's default speaker/headphone output remains unchanged.
    TTS_OUTPUT_DEVICE = find_virtual_microphone_output()

    audio_stream = sd.Stream(
        device=(MIC_INPUT_DEVICE, TTS_OUTPUT_DEVICE),
        samplerate=AUDIO_SAMPLE_RATE,
        channels=(1, 1),
        dtype="float32",
        callback=audio_callback
    )

    audio_stream.start()
    print(getTime(), ": Audio stream started. Hold F8 to speak.")


    #Start incoming audio processing
    threading.Thread(
        target=incoming_audio_worker,
        daemon=True
    ).start()


    threading.Thread(
        target=process_incoming_audio,
        daemon=True
    ).start()


except Exception as e:
    print(getTime(), ": Audio stream error:", repr(e))
    raise


#Main translation loop
try:
    while True:
        #Clear previous recording
        with recording_lock:
            recording_chunks.clear()

        #Wait for F8 to be pressed
        keyboard.wait("F8")

        with recording_lock:
            recording_chunks.clear()

        recording_active = True
        print(getTime(), ": Recording... release F8 to translate.")

        #Record until F8 is released
        while keyboard.is_pressed("F8"):
            time.sleep(0.01)

        recording_active = False

        #Copy recorded audio
        with recording_lock:
            audio_chunks = recording_chunks.copy()
            recording_chunks.clear()

        #Transcribe and translate
        recognized_text, translated_text = get_input(audio_chunks)

        if not recognized_text or not translated_text:
            continue

        #Generate Mandarin speech
        speak_mandarin(translated_text)     

except KeyboardInterrupt:
    print(getTime(), ": Reunion stopped.")

finally:
    shutdown_event.set()
    #Stop the audio stream
    recording_active = False
    audio_stream.stop()
    audio_stream.close()
