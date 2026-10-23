
import time
import numpy as np
import sounddevice as sd
import soundcard as sc

#Detect the default Windows playback device
output_index = sd.default.device[1]

if output_index is None or output_index < 0:
    raise RuntimeError("No default playback device found.")

output = sd.query_devices(output_index)
output_name = output["name"]
sample_rate = int(output["default_samplerate"])

print(f"Playback device: {output_name}")
print(f"Detected sample rate: {sample_rate} Hz")

#Find the loopback microphone matching the playback device
loopbacks = [
    mic for mic in sc.all_microphones(include_loopback=True)
    if mic.isloopback
]

def normalize(name):
    return name.lower().replace("(loopback)", "").strip()

matching = [
    mic for mic in loopbacks
    if normalize(output_name) == normalize(mic.name)
    or normalize(output_name) in normalize(mic.name)
    or normalize(mic.name) in normalize(output_name)
]

if not matching:
    print("\nAvailable loopback devices:")
    for mic in loopbacks:
        print(f"- {mic.name}")
    raise RuntimeError(
        "Could not match the default playback device to a loopback device."
    )

loopback = matching[0]
print(f"Loopback device: {loopback.name}")
print("Play audio through your headphones. Press Ctrl+C to stop.")

#Measure captured playback volume
try:
    with loopback.recorder(samplerate=sample_rate, channels=2) as recorder:
        while True:
            audio = recorder.record(
                numframes=int(sample_rate * 0.25)
            )

            if audio.size:
                level = float(
                    np.sqrt(np.mean(np.square(audio.astype(np.float32))))
                )
                print(f"\rAudio level: {level:.5f}", end="", flush=True)

            time.sleep(0.01)

except KeyboardInterrupt:
    print("\nTest stopped.")

except Exception as error:
    print(f"\nLoopback capture failed: {error}")
