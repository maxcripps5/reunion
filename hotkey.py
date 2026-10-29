#HOTKEY - Hold F8 to speak English
import time
import keyboard

#Project files
import audio
import translation
import speech
from utils import getTime


#Main translation loop
def run_hotkey_loop():
    try:
        while True:
            #Clear previous recording
            with audio.recording_lock:
                audio.recording_chunks.clear()

            #Wait for F8 to be pressed
            keyboard.wait("F8")

            with audio.recording_lock:
                audio.recording_chunks.clear()

            audio.recording_active = True

            print(
                getTime(),
                ": Recording... release F8 to translate."
            )

            #Record until F8 is released
            while keyboard.is_pressed("F8"):
                time.sleep(0.01)

            audio.recording_active = False

            #Copy recorded audio
            with audio.recording_lock:
                audio_chunks = audio.recording_chunks.copy()
                audio.recording_chunks.clear()

            #Transcribe and translate
            recognized_text, translated_text = (
                translation.get_input(audio_chunks)
            )

            if not recognized_text or not translated_text:
                continue

            #Generate Mandarin speech
            speech.speak_mandarin(translated_text)

    except KeyboardInterrupt:
        print(getTime(), ": Reunion stopped.")