"""Sample tools for demonstrating the SimpleReActAgent capabilities.

These tools provide various utilities that can be used to test and showcase
the agent's ability to reason about and use different types of tools.
Note: Some tools return mock data and are intended for demonstration purposes.
"""

import random
from datetime import datetime, timezone

from ai_test_lab.logging import get_logger

# Module-level logger
logger = get_logger(__name__)


def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        Current UTC timestamp in a readable format
    """
    current_time = datetime.now(timezone.utc)
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    logger.info("Retrieved current time", timestamp=formatted_time)
    return f"The current time is {formatted_time}"


def calculate(expression: str) -> str:
    """Perform a simple calculation.

    Args:
        expression: Mathematical expression (e.g., "2 + 2")

    Returns:
        The result of the calculation
    """
    try:
        # Only allow basic math characters for safety
        allowed = set("0123456789+-*/()., ")
        if all(c in allowed for c in expression):
            result = eval(expression, {"__builtins__": {}})
            logger.info("Calculated expression", expression=expression, result=result)
            return f"{expression} = {result}"
        else:
            return "Error: Please use only numbers and basic operators (+, -, *, /)"
    except Exception as e:
        logger.error("Calculation failed", expression=expression, error=str(e))
        return f"Error: Could not calculate {expression}"


def get_weather(location: str) -> str:
    """Get weather for a location (mock data for demo).

    Args:
        location: City name

    Returns:
        Mock weather information
    """
    # Simple mock data - in production, this would call a real weather API
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
    condition = random.choice(conditions)
    temp = random.randint(60, 85)

    logger.info("Retrieved weather", location=location, temp=temp, condition=condition)
    return f"Weather in {location}: {temp}°F and {condition}"


def search_web(query: str) -> str:
    """Search for information (mock results for demo).

    Args:
        query: Search query

    Returns:
        Mock search results
    """
    # Simple mock results - in production, this would use a real search API
    results = [
        f"Found information about {query}",
        f"Here's a relevant article on {query}",
        f"Top result for {query} from Wikipedia",
    ]

    logger.info("Searched for query", query=query)
    return "\n".join(results[:2])


def create_reminder(title: str, time: str) -> str:
    """Create a reminder (mock for demo).

    Args:
        title: What to remember
        time: When to remind

    Returns:
        Confirmation message
    """
    reminder_id = f"REM-{random.randint(100, 999)}"
    logger.info("Created reminder", id=reminder_id, title=title, time=time)
    return f"✓ Reminder created: '{title}' at {time} (ID: {reminder_id})"


def get_random_fact() -> str:
    """Get a random interesting fact.

    Returns:
        A random fact
    """
    facts = [
        "Python was named after Monty Python, not the snake.",
        "The first computer bug was an actual moth found in a computer.",
        "The @ symbol was chosen for email because it was rarely used.",
        "The first website is still online at http://info.cern.ch",
        "Git was created by Linus Torvalds in just 10 days.",
    ]

    fact = random.choice(facts)
    logger.info("Retrieved random fact")
    return f"Fun fact: {fact}"


def roll_dice(sides: int = 6) -> str:
    """Roll a dice with specified number of sides.

    Args:
        sides: Number of sides on the dice (default: 6)

    Returns:
        The dice roll result
    """
    if sides < 2:
        return "Error: Dice must have at least 2 sides"

    result = random.randint(1, sides)
    logger.info("Rolled dice", sides=sides, result=result)
    return f"Rolled a {sides}-sided dice: {result}"


def flip_coin() -> str:
    """Flip a coin.

    Returns:
        Heads or Tails
    """
    result = random.choice(["Heads", "Tails"])
    logger.info("Flipped coin", result=result)
    return f"Coin flip: {result}"


def get_joke() -> str:
    """Get a programming joke.

    Returns:
        A programming-related joke
    """
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "Why do Java developers wear glasses? Because they don't C#!",
        "How many programmers does it take to change a light bulb? None, that's a hardware problem!",
        "Why do Python programmers prefer snake_case? Because they can't C CamelCase!",
        "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
    ]

    joke = random.choice(jokes)
    logger.info("Retrieved joke")
    return joke