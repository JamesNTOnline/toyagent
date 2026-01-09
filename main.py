import os
import argparse
from dotenv import load_dotenv
from google import genai


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise RuntimeError("GEMINI_API_KEY not found in environment. Did you create your .env file?")
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    args = parser.parse_args()

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=args.user_prompt,
        )

    # print metadata
    usage = response.usage_metadata
    if usage is None:
        raise RuntimeError("no metadata in the response")
    p_count = usage.prompt_token_count
    r_count = usage.candidates_token_count

    if p_count is None or r_count is None:
        raise RuntimeError(f"partial API response; token counts missing. usage_metadata = {usage}")
    print(f"Prompt tokens: {p_count}")
    print(f"Response tokens: {r_count}")

    # print the model's answer
    print(response.text)

if __name__ == "__main__":
    main()
