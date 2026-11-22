"""Settings read from the environment, plus fixed business rules. Imported by every other module."""
import os

# The project root (one level above this package): index.html and the default database live there.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
# RESTAURANT_DB_PATH lets tests use a throwaway database.
DB_PATH = os.environ.get("RESTAURANT_DB_PATH", os.path.join(BASE_DIR, "restaurant_agent.db"))
CSV_PATH = os.path.join(BASE_DIR, "restaurants_1000.csv")

# --- LLM (LM Studio, OpenAI-compatible API) ---
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1235/v1")
LMSTUDIO_ROOT = LLM_BASE_URL.rstrip("/").removesuffix("/v1")
# Models offered in the UI as (LM Studio model key, display label). The first one is the default.
# Only one fits in GPU memory, so switching unloads the other one (see lmstudio.py).
ALLOWED_MODELS = [
    ("gemma-4-26b-a4b-it", "Gemma 4 26B"),
    ("qwen3.8-27b", "Qwen 3.8 27B"),
]
MODEL_NAME = os.environ.get("LLM_MODEL", ALLOWED_MODELS[0][0])
LOAD_CONTEXT_LENGTH = int(os.environ.get("LLM_CONTEXT_LENGTH", "16384"))
MODEL_LOAD_TIMEOUT = 600
LLM_TIMEOUT_SECONDS = 300  # Qwen 3.8 27B is about 4x slower than Gemma on this hardware
MAX_TOOL_ROUNDS = 6
# Older turns are dropped (whole user turns only, so tool-call pairs stay complete).
MAX_HISTORY_TURNS = 12

# --- Reservation rules ---
SITTING_MINUTES = 120          # how long a table is blocked by one reservation
LAST_SEATING_BEFORE_CLOSE = 60 # minutes: the last booking must start at least this long before closing
MAX_DAYS_AHEAD = 60
MIN_GUESTS, MAX_GUESTS = 1, 20 # bigger groups go to the events team (a human)
SLOT_STEP_MINUTES = 30         # bookings start on :00 or :30

# --- Voice ---
# Speech-to-text runs locally with faster-whisper on the CPU (the GPU is reserved for the LLM).
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
MAX_AUDIO_BYTES = 15 * 1024 * 1024
# Languages the agent supports. Whisper's language detection is restricted to these, because short
# Bulgarian clips are otherwise often labelled as Russian or Macedonian.
SUPPORTED_LANGUAGES = ("en", "bg", "de", "es", "fr", "it")
PRIMARY_LANGUAGE = "en"
WHISPER_PROMPTS = {
    "en": "Restaurant booking: table for two, reservation, tonight, tomorrow, allergens, gluten, menu, vegetarian.",
    "bg": "Резервация в ресторант: маса за двама, резервация, довечера, утре, алергени, глутен, меню, вегетарианско. "
          "Ястия: телешко варено, пилешка супа, шкембе чорба, боб чорба, таратор, мусака, сарми, шопска салата, "
          "кебапчета, баница, каварма.",
    "de": "Restaurantreservierung: Tisch für zwei, heute Abend, morgen, Allergene, Gluten, Speisekarte, vegetarisch.",
    "es": "Reserva de restaurante: mesa para dos, esta noche, mañana, alérgenos, gluten, menú, vegetariano.",
    "fr": "Réservation de restaurant : table pour deux, ce soir, demain, allergènes, gluten, menu, végétarien.",
    "it": "Prenotazione ristorante: tavolo per due, stasera, domani, allergeni, glutine, menù, vegetariano.",
}
# Text-to-speech uses edge-tts (Microsoft's online neural voices), so reply text leaves the machine.
TTS_VOICES = {
    "male": {"en": "en-US-AndrewNeural", "bg": "bg-BG-BorislavNeural", "de": "de-DE-ConradNeural",
             "es": "es-ES-AlvaroNeural", "fr": "fr-FR-HenriNeural", "it": "it-IT-DiegoNeural"},
    "female": {"en": "en-US-AvaNeural", "bg": "bg-BG-KalinaNeural", "de": "de-DE-KatjaNeural",
               "es": "es-ES-ElviraNeural", "fr": "fr-FR-DeniseNeural", "it": "it-IT-ElsaNeural"},
}
DEFAULT_VOICE_GENDER = "female"
MAX_TTS_CHARS = 1200
TTS_ATTEMPTS = 3
TTS_ATTEMPT_TIMEOUT = 15
