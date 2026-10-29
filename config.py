#CONFIG - Vars that will be changeable in the GUI
import warnings

#Project files
from openai import OpenAI
from sk import my_sk

#Suppress SoundCard errors prompts (temporary)
warnings.filterwarnings(
    "ignore",
    message="data discontinuity in recording",
    category=RuntimeWarning
)

#OpenAI
client = OpenAI(
    api_key=my_sk,
    timeout=60.0,
    max_retries=1
)

#Audio devices
MIC_INPUT_DEVICE = None
TTS_OUTPUT_DEVICE = 7

#Audio sample rates
AUDIO_SAMPLE_RATE = 48000
TTS_SAMPLE_RATE = 24000

#Silence audio detection
SILENCE_VOL_THRESHOLD = 0.02

#Incoming audio detection
INCOMING_SPEECH_THRESHOLD = 0.035
INCOMING_SILENCE_SECONDS = 0.75
INCOMING_MAX_SEGMENT_SECONDS = 10

#Incoming audio queue
INCOMING_QUEUE_MAXSIZE = 5