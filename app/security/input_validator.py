import re

MAX_RESUME_LENGTH = 20_000


def validate_resume_input(resume_text: str) -> None:
    """
    Validate resume text before sending it to the ML model.

    Raises:
        ValueError: If the resume input is invalid or suspicious.
    """

    if resume_text is None:
        raise ValueError("Resume text is required.")

    if not isinstance(resume_text, str):
        raise ValueError("Resume input must be text.")

    cleaned_text = resume_text.strip()

    if not cleaned_text:
        raise ValueError("Resume text cannot be empty.")

    if len(cleaned_text) > MAX_RESUME_LENGTH:
        raise ValueError(
            f"Resume text exceeds the maximum allowed length of "
            f"{MAX_RESUME_LENGTH} characters."
        )

    suspicious_patterns = {
        "HTML or script content": r"<\s*(script|iframe|object|embed|html|body)",
        "SQL injection pattern": (
            r"\b(drop\s+table|delete\s+from|insert\s+into|"
            r"union\s+select|alter\s+table)\b"
        ),
        "Private key content": r"-----BEGIN .*PRIVATE KEY-----",
        "API key or token": (r"\b(api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]"),
        "Password or credential": (r"\b(password|passwd|pwd|username)\s*[:=]\s*\S+"),
    }

    for issue_name, pattern in suspicious_patterns.items():
        if re.search(pattern, cleaned_text, flags=re.IGNORECASE):
            raise ValueError(
                f"Resume input rejected because it contains "
                f"suspicious {issue_name.lower()}."
            )

    special_character_count = sum(
        1 for char in cleaned_text if not char.isalnum() and not char.isspace()
    )

    special_character_ratio = special_character_count / len(cleaned_text)

    if special_character_ratio > 0.30:
        raise ValueError("Resume input contains too many special characters.")

    if re.search(r"(.)\1{20,}", cleaned_text):
        raise ValueError("Resume input contains excessively repeated characters.")
