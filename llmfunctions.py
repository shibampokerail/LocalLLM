from datetime import datetime
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedules a meeting with specified attendees at a given date and time. Use today's date if the user doesn't specify one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attendees": {"type": "array", "items": {"type": "string"},
                                  "description": "List of people to invite."},
                    "date": {"type": "string", "description": "The date of the meeting in YYYY-MM-DD format."},
                    "time": {"type": "string", "description": "The time of the meeting in HH:MM format."},
                    "topic": {"type": "string", "description": "The subject or topic of the meeting."},
                },
                "required": ["attendees", "date", "time", "topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_weather",
            "description": "Gets the current weather for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name, e.g., 'San Francisco'."},
                    "units": {"type": "string", "enum": ["metric", "imperial"], "default": "metric"},
                },
                "required": ["city"],
            },
        },
    },
]


# 2. Function Implementations (The "How")
# These are the actual Python functions that execute the logic.
# Note: They now accept named arguments directly, which is cleaner.
def schedule_meeting(attendees, date, time, topic):
    """Schedules a meeting and returns a confirmation string."""
    try:
        dt = datetime.fromisoformat(f"{date}T{time}")
        attendee_str = ", ".join(attendees)
        return f"Meeting '{topic}' successfully scheduled for {dt.strftime('%A, %B %d at %H:%M')} with {attendee_str}."
    except (ValueError, TypeError) as e:
        return f"Error scheduling meeting: Invalid date or time format. Details: {e}"


def fetch_weather(city, units="metric"):
    """Fetches a mock weather report for a given city."""
    # In a real application, you would call a weather API here.
    temp = 22 if units == "metric" else 72
    unit_label = "°C" if units == "metric" else "°F"
    return f"🌤️ The current temperature in {city} is {temp}{unit_label}."


# Mapping tool names to their implementation functions.
FUNCTION_IMPLEMENTATIONS = {
    "schedule_meeting": schedule_meeting,
    "fetch_weather": fetch_weather,
}

