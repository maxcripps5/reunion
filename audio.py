#Audi
import io
import wave
import queue
import threading
import re
import numpy as np
import sounddevice as sd
import soundcard as sc

#Project files
from utils import getTime

#Microphone capture, loopback capture, audio callbacks, buffers and device routing

#Audio sample rate
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
            #print(getTime(), ": Incoming audio device:", mic.name)
            return mic

    #Try a partial name match
    for mic in loopbacks:
        normalized_name = normalize(mic.name)

        if (
            normalized_speaker in normalized_name
            or normalized_name in normalized_speaker
        ):
            #print(getTime(), ": Incoming audio device:", mic.name)
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

        device_name = device.name.removeprefix("Loopback Speakers (").rsplit(") (", 1)[0]

        print(getTime(), ": Listening for incoming speech with", device_name)

        with device.recorder(
            samplerate=sample_rate,
            channels=2
        ) as recorder:
            audio_buffer = []
            silence_duration = 0.0
            buffered_seconds = 0.0
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
                    buffered_seconds += duration

                elif speech_detected:
                    audio_buffer.append(audio.copy())
                    buffered_seconds += duration
                    silence_duration += duration

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
                    buffered_seconds = 0.0
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
