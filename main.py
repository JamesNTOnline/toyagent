import os 
from dotenv import load_dotenv
from google import genai
from google.genai import types
import sys

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

if len(sys.argv) > 1:
        messages = [
                types.Content(role="user", parts=[types.Part(text=sys.argv[1])]),
                ]
        res = client.models.generate_content(model="gemini-2.0-flash-001",contents=messages)
        if len(sys.argv) > 2:
            print(f"{res.text}\nUser prompt: {messages[0].parts[0].text}\nPrompt tokens: {res.usage_metadata.prompt_token_count}\nResponse tokens: {res.usage_metadata.candidates_token_count}")
else:
        print("prompt not provided")
        sys.exit(1)

def main():
    print("Hello from llm-agent!")


if __name__ == "__main__":
    main()
