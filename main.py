import os
import argparse
from dotenv import load_dotenv
from google import genai
from google.genai import types


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key is None:
        raise RuntimeError("GEMINI_API_KEY not found in environment. Did you create your .env file?")
    client = genai.Client(api_key=api_key)
   
    # set up a parser with a prompt and verbose argument
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("user_prompt", type=str, help="User prompt")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    messages = [types.Content(role="user", parts=[types.Part(text=args.user_prompt)])]

    # verbose print - includes metadata
    if args.verbose:
        print(f"User prompt: {args.user_prompt}\n")

    generate_output(client, messages, args.verbose)



def generate_output(client, messages, verbose):     
    # get a response from the api
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=messages,
        )
    # check for usage data to display in a verbose response    
    if verbose:
        usage = res.usage_metadata
        if usage is None: # no usage data
            raise RuntimeError("no metadata in the response")
        p_count = usage.prompt_token_count
        r_count = usage.candidates_token_count
        if p_count is None or r_count is None: #partially missing usage data
            raise RuntimeError(f"partial API response; token counts missing. usage_metadata = {usage}")
        # all ok! print out the usage data
        print(f"Prompt tokens: {p_count}") 
        print(f"Response tokens: {r_count}")

    # print the model's answer
    print(f"Response: {res.text}")



if __name__ == "__main__":
    main()
