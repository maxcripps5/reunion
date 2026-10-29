#MAIN - Main project file for Reunion
import threading
import sounddevice as sd

import audio
import config
import translation
import hotkey
from utils import getTime


#Start audio stream
try:
    #Send microphone passthrough and Mandarin TTS to the virtual microphone
    config.TTS_OUTPUT_DEVICE = audio.find_virtual_microphone_output()

    audio_stream = sd.Stream(
        device=(
            config.MIC_INPUT_DEVICE,
            config.TTS_OUTPUT_DEVICE
        ),
        samplerate=config.AUDIO_SAMPLE_RATE,
        channels=(1, 1),
        dtype="float32",
        callback=audio.audio_callback
    )

    audio_stream.start()

    print(
        getTime(),
        ": Audio stream started. Hold F8 to speak."
    )

    #Start incoming audio processing
    threading.Thread(                                   
        target=audio.incoming_audio_worker,
        daemon=True
    ).start()

    #Start incoming translation processing
    threading.Thread(
        target=translation.process_incoming_audio,
        daemon=True
    ).start()

except Exception as e:
    print(
        getTime(),
        ": Audio stream error:",
        repr(e)
    )
    raise


#Run the F8 translation loop
try:
    hotkey.run_hotkey_loop()

except KeyboardInterrupt:
    print(getTime(), ": Reunion stopped.")

finally:
    audio.shutdown_event.set()

    #Stop the audio stream
    audio.recording_active = False

    audio_stream.stop()
    audio_stream.close()