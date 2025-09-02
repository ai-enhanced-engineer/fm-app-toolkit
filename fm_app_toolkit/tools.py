"""Simple tools for demonstrating LlamaIndex testing patterns."""


def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide numbers, raises ValueError if divisor is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


def word_count(text: str) -> int:
    """Count the number of words in text."""
    return len(text.split())
