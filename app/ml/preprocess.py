import re


def clean_resume(text: str) -> str:
    """
    Clean resume text before prediction or training.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove phone numbers
    text = re.sub(r"\+?\d[\d\s()-]{7,}\d", " ", text)

    # Remove hashtags
    text = re.sub(r"#\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\S+", " ", text)

    # Remove special characters
    text = re.sub(r"[^a-z0-9 ]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
