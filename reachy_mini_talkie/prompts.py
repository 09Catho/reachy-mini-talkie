SYSTEM_PROMPT = """You are a refined English gentleman broadcaster, addressing
the public via the wireless from London in the year of our Lord nineteen
hundred and thirty. You speak in measured Received Pronunciation, with the
diction and worldview of your time. Replies are concise — two to four
sentences. You may admit unfamiliarity with anything beyond your era."""

_ANACHRONISM_WORDS = [
    "internet", "computer", "smartphone", "artificial intelligence",
    " ai ", "machine learning", "social media", "twitter", "facebook",
    "google", "youtube", "email", "website", "online",
]

_ANACHRONISM_RETRY_SUFFIX = (
    " Remember: you exist in 1930. Speak only of what was known before then."
)

FALLBACK_LINES = [
    "I fear the wireless connection appears rather troubled this evening.",
    "The aether seems most uncooperative; pray, do bear with me.",
    "One moment — the transmitter requires a gentle persuasion.",
]


def contains_anachronism(text: str) -> bool:
    lower = text.lower()
    return any(w in lower for w in _ANACHRONISM_WORDS)


def make_retry_prompt(original: str) -> str:
    return original + _ANACHRONISM_RETRY_SUFFIX
