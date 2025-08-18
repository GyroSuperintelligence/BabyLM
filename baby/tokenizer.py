from openai_harmony import load_harmony_encoding, HarmonyEncodingName

def get_tokenizer():
    """Get the standard Harmony encoding for consistency across the codebase."""
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
