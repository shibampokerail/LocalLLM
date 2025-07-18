import json
from datetime import datetime
from pathlib import Path
import sys
from llama_cpp import Llama
from huggingface_hub import hf_hub_download


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
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of people to invite."},
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
    """An agent that can use tools to answer questions."""

    def __init__(self, model_path, tools, function_implementations):
        self.model_path = Path(model_path)
        self.tools = tools
        self.function_implementations = function_implementations
        
        self._ensure_model_exists()

        print("Initializing model... (This may take a moment)")
        self.llm = Llama(
            model_path=str(self.model_path), 
            chat_format="chatml",
            n_ctx=4096,
            n_threads=6,
            verbose=False,
        )
        print("Model initialized.")
        self.system_prompt = self._create_system_prompt()
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def _ensure_model_exists(self):
        """Checks if the model file exists, and if not, prompts the user to download it."""
        if not self.model_path.is_file():
            print(f"Model not found at '{self.model_path}'")
            
            # Ask the user for permission to download
            choice = input("Would you like to download the model? (approx. 2.2 GB) [Y/n]: ").strip().lower()
            
            if choice in ["y", "yes"]: # Default to 'yes' if they just press Enter
                print(f"Downloading '{MODEL_FILENAME}' from Hugging Face...")
                print(f"Repo: {MODEL_REPO}")

                # Create the model directory if it doesn't exist
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Use huggingface_hub to download with a progress bar
                    hf_hub_download(
                        repo_id=MODEL_REPO,
                        filename=MODEL_FILENAME,
                        local_dir=self.model_path.parent,
                        local_dir_use_symlinks=False, # Copies the file to your dir
                    )
                    print(" Download complete.")
                except Exception as e:
                    print(f" An error occurred during download: {e}")
                    print("Please try downloading the model manually and placing it in the 'models' directory.")
                    sys.exit(1) 
            else:
                print("Download declined. Exiting.")
                sys.exit(0) 

   
    def _create_system_prompt(self):
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
        self.chat_history.append({"role": "user", "content": user_query})
        response = self.llm.create_chat_completion(
            messages=self.chat_history,
            tools=self.tools,
            tool_choice="auto",
            temperature=0.0,
        )
        choice = response["choices"][0]["message"]
        self.chat_history.append(choice)
        content = choice.get("content", "")
        final_answer = None

        try:
            tool_call_data = json.loads(content)
            if (isinstance(tool_call_data, dict) and
                "name" in tool_call_data and
                tool_call_data["name"] in self.function_implementations):
                print(f"Bot wants to use a tool (detected from content): {tool_call_data['name']}")
                fn_name = tool_call_data["name"]
                args = tool_call_data.get("arguments", {})
                try:
                    print(f"  > Calling function: {fn_name} with args: {args}")
                    function_to_call = self.function_implementations[fn_name]
                    final_answer = function_to_call(**args)
                except Exception as e:
                    print(f"  > Error calling function {fn_name}: {e}")
                    final_answer = f"Error executing tool {fn_name}: {e}"
            else:
                final_answer = content
        except (json.JSONDecodeError, TypeError):
            print("Bot responded directly.")
            final_answer = content
        self.chat_history.append({"role": "assistant", "content": final_answer})
        return final_answer

def main():
    """The main function to run the chat application."""
    print("🔧 Initializing Chat with Function-Calling Phi-3 Mini...")
    
    agent = FunctionCallingAgent(
        model_path=MODEL_PATH,
        tools=TOOL_DEFINITIONS,
        function_implementations=FUNCTION_IMPLEMENTATIONS
    )

    print("\n Agent is ready! Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue

            bot_response = agent.chat(user_input)
            print("Bot:", bot_response)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()
