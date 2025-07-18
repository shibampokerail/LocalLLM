import json
from datetime import datetime
from pathlib import Path
import sys
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS

# --- All your existing agent code ---
MODEL_DIR = Path("./models")
MODEL_REPO = "TheBloke/Phi-3-mini-4k-instruct-GGUF"
MODEL_FILENAME = "Phi-3-mini-4k-instruct.Q4_K_M.gguf"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

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


def schedule_meeting(attendees, date, time, topic):
    try:
        dt = datetime.fromisoformat(f"{date}T{time}")
        attendee_str = ", ".join(attendees)
        return f" Meeting '{topic}' successfully scheduled for {dt.strftime('%A, %B %d at %H:%M')} with {attendee_str}."
    except (ValueError, TypeError) as e:
        return f"Error scheduling meeting: Invalid date or time format. Details: {e}"


def fetch_weather(city, units="metric"):
    temp = 22 if units == "metric" else 72
    unit_label = "°C" if units == "metric" else "°F"
    return f"🌤️ The current temperature in {city} is {temp}{unit_label}."


FUNCTION_IMPLEMENTATIONS = {
    "schedule_meeting": schedule_meeting,
    "fetch_weather": fetch_weather,
}


class FunctionCallingAgent:
    def __init__(self, model_path, tools, function_implementations):
        self.model_path = Path(model_path)
        self.tools = tools
        self.function_implementations = function_implementations
        self._ensure_model_exists()
        print("Initializing model... (This may take a few moments)")
        self.llm = Llama(
            model_path=str(self.model_path),
            chat_format="chatml",
            n_ctx=4096,
            n_threads=6,
            verbose=False,
        )
        print("Model initialized successfully.")
        self.system_prompt = self._create_system_prompt()
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def _ensure_model_exists(self):
        if not self.model_path.is_file():
            print(f"Model not found at '{self.model_path}'. Downloading...")
            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILENAME,
                    local_dir=self.model_path.parent,
                    local_dir_use_symlinks=False,
                )
                print("Download complete.")
            except Exception as e:
                print(f"An error occurred during download: {e}")
                sys.exit(1)

    def _create_system_prompt(self):
        # ... (This function is unchanged)
        prompt_parts = [
            "You are a helpful assistant that strictly follows instructions to call functions.",
            f"The current date and time is: {datetime.now().isoformat()}",
            "\n--- RULES ---",
            "1. You MUST call a tool when the user's request can be fulfilled by one of the available tools.",
            "2. When you decide to call a tool, you MUST respond in the format of a JSON object.",
            "3. The JSON object MUST contain \"name\" (the tool name) and \"arguments\" (a sub-object with parameters).",
            "4. Your response MUST ONLY contain the JSON object and nothing else. Do not add any conversational text, explanations, or apologies before or after the JSON.",
            "\n--- AVAILABLE TOOLS ---"
        ]
        for tool in self.tools:
            func = tool['function']
            prompt_parts.append(
                f"\nTool: {func['name']}\n"
                f"Description: {func['description']}\n"
                f"Parameters (JSON Schema): {json.dumps(func['parameters'], indent=2)}"
            )
        prompt_parts.append("\n--- END OF TOOLS ---")
        return "\n".join(prompt_parts)

    def chat(self, user_query):
        # This is kept mostly the same, but we will reset history for each API call
        # for a stateless API. If you want to maintain session state, this would need adjustment.
        chat_session = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        response = self.llm.create_chat_completion(
            messages=chat_session,
            tools=self.tools,
            tool_choice="auto",
            temperature=0.0,
        )
        choice = response["choices"][0]["message"]
        content = choice.get("content", "")
        final_answer = None

        try:
            tool_call_data = json.loads(content)
            if (isinstance(tool_call_data, dict) and
                    "name" in tool_call_data and
                    tool_call_data["name"] in self.function_implementations):
                print(f"Bot wants to use tool: {tool_call_data['name']}")
                fn_name = tool_call_data["name"]
                args = tool_call_data.get("arguments", {})
                function_to_call = self.function_implementations[fn_name]
                final_answer = function_to_call(**args)
            else:
                final_answer = content
        except (json.JSONDecodeError, TypeError):
            print("Bot responded directly with text.")
            final_answer = content

        print(f"Final answer to be sent: {final_answer}")
        return final_answer


# --- NEW: Flask API Implementation ---

# 1. Initialize the Flask App
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# 2. Load the Agent (This happens only once on startup)
print("🔧 Initializing Function-Calling Agent for Flask Server...")
agent = FunctionCallingAgent(
    model_path=MODEL_PATH,
    tools=TOOL_DEFINITIONS,
    function_implementations=FUNCTION_IMPLEMENTATIONS
)
print("\n✅ Agent is loaded and ready to receive requests.")


# 3. Define the chat endpoint
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Receives a user message and returns the agent's response."""
    if not request.json or 'message' not in request.json:
        return jsonify({"error": "Invalid request: 'message' key not found in JSON payload."}), 400

    user_message = request.json['message']
    print(f"\nReceived message: '{user_message}'")

    if not user_message.strip():
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        bot_response = agent.chat(user_message)
        return jsonify({"response": bot_response})
    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        return jsonify({"error": "An internal error occurred."}), 500


# 4. Run the app on a specific port (e.g., 5001) to avoid conflicts
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
