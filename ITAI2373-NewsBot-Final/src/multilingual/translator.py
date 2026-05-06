"""
Translation via deep-translator (Google Translate backend — no API key required).
"""

from deep_translator import GoogleTranslator

from config.settings import TRANSLATION_TARGET_LANG
from src.multilingual.language_detector import detect_language


def translate_to_english(text: str) -> str:
    """Translate text to English.

    Args:
        text: Source-language text.

    Returns:
        English translation string, or original text if translation fails.
    """
    try:
        return GoogleTranslator(source='auto', target=TRANSLATION_TARGET_LANG).translate(str(text))
    except Exception as exc:
        return f'[Translation error: {exc}]'


def translate_and_detect(text: str) -> dict:
    """Detect source language and translate to English.

    Args:
        text: Source text in any supported language.

    Returns:
        Dict with 'source_lang', 'translated', and 'error' keys.
    """
    source_lang = detect_language(text)
    if source_lang == 'en':
        return {'source_lang': 'en', 'translated': text, 'error': None}
    try:
        translated = GoogleTranslator(source='auto', target=TRANSLATION_TARGET_LANG).translate(str(text))
        return {'source_lang': source_lang, 'translated': translated, 'error': None}
    except Exception as exc:
        return {'source_lang': source_lang, 'translated': text, 'error': str(exc)}


def batch_translate(texts) -> list[dict]:
    """Translate a list of texts.

    Args:
        texts: Iterable of strings.

    Returns:
        List of dicts from translate_and_detect().
    """
    return [translate_and_detect(t) for t in texts]
