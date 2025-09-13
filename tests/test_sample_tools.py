"""Unit tests for sample tools.

These tests verify the functionality of each sample tool, including
error handling and logging behavior.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from src.agents.llamaindex.sample_tools import (
    calculate,
    create_reminder,
    flip_coin,
    get_current_time,
    get_joke,
    get_random_fact,
    get_weather,
    roll_dice,
    search_web,
)


def test__get_current__time() -> None:
    """Test that get_current_time returns a properly formatted timestamp."""
    result = get_current_time()

    # Should start with "The current time is"
    assert result.startswith("The current time is")

    # Should contain UTC
    assert "UTC" in result

    # Should be parseable as a datetime
    time_str = result.replace("The current time is ", "")
    datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S UTC")


def test__calculate_simple__operations() -> None:
    """Test basic calculations."""
    assert calculate("2 + 2") == "2 + 2 = 4"
    assert calculate("10 - 5") == "10 - 5 = 5"
    assert calculate("3 * 4") == "3 * 4 = 12"
    assert calculate("15 / 3") == "15 / 3 = 5.0"
    assert calculate("(2 + 3) * 4") == "(2 + 3) * 4 = 20"


def test__calculate_with__decimals() -> None:
    """Test calculations with decimal numbers."""
    assert calculate("3.14 * 2") == "3.14 * 2 = 6.28"
    assert calculate("10.5 / 2.5") == "10.5 / 2.5 = 4.2"


def test__calculate_invalid__characters() -> None:
    """Test that calculate rejects invalid expressions."""
    result = calculate("2 + 2; print('hack')")
    assert result.startswith("Error: Could not calculate")

    result = calculate("import os")
    assert result.startswith("Error: Could not calculate")


def test__calculate_division__by_zero() -> None:
    """Test that calculate handles division by zero."""
    result = calculate("10 / 0")
    assert result.startswith("Error: Could not calculate")


def test__calculate_malformed__expression() -> None:
    """Test that calculate handles malformed expressions."""
    # Note: "2 + + 3" is actually valid Python (unary plus), so it evaluates to 5
    # Let's test with a truly malformed expression
    result = calculate("2 +* 3")
    assert result.startswith("Error: Could not calculate")


@patch("random.choice")
@patch("random.randint")
def test__get__weather(mock_randint: MagicMock, mock_choice: MagicMock) -> None:
    """Test weather retrieval with mocked random values."""
    mock_choice.return_value = "sunny"
    mock_randint.return_value = 75

    result = get_weather("San Francisco")
    assert result == "Weather in San Francisco: 75°F and sunny"

    result = get_weather("New York")
    assert result == "Weather in New York: 75°F and sunny"


def test__search__web() -> None:
    """Test web search returns mock results."""
    result = search_web("Python programming")

    assert "Found information about Python programming" in result
    assert "Here's a relevant article on Python programming" in result

    # Should return 2 results
    lines = result.split("\n")
    assert len(lines) == 2


@patch("random.randint")
def test__create__reminder(mock_randint: MagicMock) -> None:
    """Test reminder creation with mocked ID."""
    mock_randint.return_value = 456

    result = create_reminder("Team meeting", "2024-01-15 14:00")
    assert result == "✓ Reminder created: 'Team meeting' at 2024-01-15 14:00 (ID: REM-456)"


@patch("random.choice")
def test__get_random__fact(mock_choice: MagicMock) -> None:
    """Test random fact retrieval."""
    expected_fact = "Python was named after Monty Python, not the snake."
    mock_choice.return_value = expected_fact

    result = get_random_fact()
    assert result == f"Fun fact: {expected_fact}"


def test__get_random__fact_contains_valid_facts() -> None:
    """Test that get_random_fact returns one of the predefined facts."""
    valid_facts = [
        "Python was named after Monty Python, not the snake.",
        "The first computer bug was an actual moth found in a computer.",
        "The @ symbol was chosen for email because it was rarely used.",
        "The first website is still online at http://info.cern.ch",
        "Git was created by Linus Torvalds in just 10 days.",
    ]

    # Run multiple times to ensure it's returning from the valid set
    for _ in range(10):
        result = get_random_fact()
        fact_text = result.replace("Fun fact: ", "")
        assert fact_text in valid_facts


@patch("random.randint")
def test__roll_dice__default(mock_randint: MagicMock) -> None:
    """Test rolling a default 6-sided dice."""
    mock_randint.return_value = 4

    result = roll_dice()
    assert result == "Rolled a 6-sided dice: 4"


@patch("random.randint")
def test__roll_dice__custom_sides(mock_randint: MagicMock) -> None:
    """Test rolling dice with custom number of sides."""
    mock_randint.return_value = 15

    result = roll_dice(20)
    assert result == "Rolled a 20-sided dice: 15"


def test__roll_dice__invalid_sides() -> None:
    """Test that roll_dice rejects invalid number of sides."""
    assert roll_dice(1) == "Error: Dice must have at least 2 sides"
    assert roll_dice(0) == "Error: Dice must have at least 2 sides"
    assert roll_dice(-5) == "Error: Dice must have at least 2 sides"


@patch("random.choice")
def test__flip_coin__heads(mock_choice: MagicMock) -> None:
    """Test coin flip returning heads."""
    mock_choice.return_value = "Heads"

    result = flip_coin()
    assert result == "Coin flip: Heads"


@patch("random.choice")
def test__flip_coin__tails(mock_choice: MagicMock) -> None:
    """Test coin flip returning tails."""
    mock_choice.return_value = "Tails"

    result = flip_coin()
    assert result == "Coin flip: Tails"


def test__flip_coin__valid_results() -> None:
    """Test that flip_coin only returns Heads or Tails."""
    valid_results = ["Coin flip: Heads", "Coin flip: Tails"]

    # Run multiple times to check both outcomes are possible
    for _ in range(20):
        result = flip_coin()
        assert result in valid_results


@patch("random.choice")
def test__get__joke(mock_choice: MagicMock) -> None:
    """Test joke retrieval."""
    expected_joke = "Why do programmers prefer dark mode? Because light attracts bugs!"
    mock_choice.return_value = expected_joke

    result = get_joke()
    assert result == expected_joke


def test__get_joke__valid_jokes() -> None:
    """Test that get_joke returns one of the predefined jokes."""
    valid_jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "Why do Java developers wear glasses? Because they don't C#!",
        "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
        "Why do Python programmers prefer snake_case? Because they can't C CamelCase!",
        "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
    ]

    # Run multiple times to ensure it's returning from the valid set
    for _ in range(10):
        result = get_joke()
        assert result in valid_jokes


def test__all_tools__return_strings() -> None:
    """Test that all tools return string values as expected."""
    # This is important for the agent's text-based interface
    assert isinstance(get_current_time(), str)
    assert isinstance(calculate("1 + 1"), str)
    assert isinstance(get_weather("Test City"), str)
    assert isinstance(search_web("test query"), str)
    assert isinstance(create_reminder("test", "now"), str)
    assert isinstance(get_random_fact(), str)
    assert isinstance(roll_dice(), str)
    assert isinstance(flip_coin(), str)
    assert isinstance(get_joke(), str)
