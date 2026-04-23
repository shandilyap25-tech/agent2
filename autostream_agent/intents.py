from __future__ import annotations

import re
from string import capwords

INTENT_GREETING = "casual_greeting"
INTENT_PRODUCT = "product_or_pricing_inquiry"
INTENT_HIGH = "high_intent_lead"

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

PLATFORM_ALIASES = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "twitch": "Twitch",
    "linkedin": "LinkedIn",
    "facebook": "Facebook",
    "twitter": "X/Twitter",
    "x": "X/Twitter",
    "podcast": "Podcast",
    "spotify": "Spotify",
}

PRODUCT_KEYWORDS = {
    "pricing",
    "price",
    "cost",
    "plan",
    "plans",
    "feature",
    "features",
    "refund",
    "refunds",
    "support",
    "caption",
    "captions",
    "4k",
    "720p",
    "video",
    "videos",
    "resolution",
    "autostream",
    "month",
}

GREETING_KEYWORDS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}

HIGH_INTENT_PHRASES = (
    "i want to try",
    "ready to sign up",
    "sign me up",
    "get started",
    "start trial",
    "start a trial",
    "i want the pro plan",
    "i want pro",
    "i want to subscribe",
    "upgrade me",
    "sounds good i want",
    "sounds good, i want",
    "i am ready to buy",
    "let's do it",
)

STOP_NAME_WORDS = {
    "and",
    "or",
    "youtube",
    "instagram",
    "tiktok",
    "twitter",
    "facebook",
    "linkedin",
    "podcast",
    "spotify",
    "pro",
    "basic",
    "pricing",
    "plan",
    "plans",
    "email",
}


def contains_product_keywords(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in PRODUCT_KEYWORDS)


def detect_intent(text: str, collecting_lead: bool = False) -> str:
    normalized = " ".join(text.lower().split())

    if collecting_lead:
        return INTENT_HIGH

    if any(phrase in normalized for phrase in HIGH_INTENT_PHRASES):
        return INTENT_HIGH

    if contains_product_keywords(normalized):
        return INTENT_PRODUCT

    if any(greeting in normalized for greeting in GREETING_KEYWORDS):
        return INTENT_GREETING

    if normalized.endswith("?"):
        return INTENT_PRODUCT

    return INTENT_GREETING


def extract_lead_details(text: str, existing: dict[str, str] | None = None) -> dict[str, str]:
    lead_info = dict(existing or {})
    lead_info.setdefault("name", "")
    lead_info.setdefault("email", "")
    lead_info.setdefault("platform", "")

    email = _extract_email(text)
    if email:
        lead_info["email"] = email

    platform = _extract_platform(text)
    if platform:
        lead_info["platform"] = platform

    name = _extract_name(text)
    if name:
        lead_info["name"] = name

    return lead_info


def get_missing_lead_fields(lead_info: dict[str, str]) -> list[str]:
    required_fields = ("name", "email", "platform")
    return [field for field in required_fields if not lead_info.get(field, "").strip()]


def _extract_email(text: str) -> str:
    match = EMAIL_PATTERN.search(text)
    return match.group(0).strip() if match else ""


def _extract_platform(text: str) -> str:
    normalized = text.lower()
    for alias, canonical in PLATFORM_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            return canonical
    return ""


def _extract_name(text: str) -> str:
    patterns = (
        r"\bmy name is\s+([A-Za-z][A-Za-z .'-]{1,60})",
        r"\bname is\s+([A-Za-z][A-Za-z .'-]{1,60})",
        r"\bi am\s+([A-Za-z][A-Za-z .'-]{1,60})",
        r"\bi'm\s+([A-Za-z][A-Za-z .'-]{1,60})",
        r"\bthis is\s+([A-Za-z][A-Za-z .'-]{1,60})",
        r"\bname:\s*([A-Za-z][A-Za-z .'-]{1,60})",
    )

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = _clean_name(match.group(1))
            if candidate:
                return candidate

    stripped = text.strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,60}", stripped):
        candidate = _clean_name(stripped)
        if candidate:
            return candidate

    return ""


def _clean_name(value: str) -> str:
    candidate = re.split(r"[,;\n]", value, maxsplit=1)[0].strip()
    candidate = re.sub(r"\s+", " ", candidate)

    if not candidate or EMAIL_PATTERN.search(candidate):
        return ""

    words = candidate.replace("-", " ").replace("'", " ").split()
    if not 1 <= len(words) <= 4:
        return ""

    lowered_words = {word.lower() for word in words}
    if lowered_words & STOP_NAME_WORDS:
        return ""

    if any(char.isdigit() for char in candidate):
        return ""

    return capwords(candidate)

