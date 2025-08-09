"""Simple tools for demonstrating LlamaIndex testing patterns."""


def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b
    """
    return a * b


def divide(a: float, b: float) -> float:
    """Divide one number by another.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        The quotient of a divided by b

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def reverse_string(text: str) -> str:
    """Reverse a string.

    Args:
        text: The string to reverse

    Returns:
        The reversed string
    """
    return text[::-1]


def word_count(text: str) -> int:
    """Count the number of words in a text.

    Args:
        text: The text to count words in

    Returns:
        The number of words
    """
    return len(text.split())
