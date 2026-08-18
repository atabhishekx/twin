from agno import os
from openai import OpenAI
from context import TWIN_SYSTEM_PROMPT
from tools import tools, handle_tool_calls
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr
import os 

load_dotenv(override=True)

MODEL_NAME = "gemini-3.6-flash"

google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    print(f"Google API Key exists and begins {google_api_key[:2]}")
else:
    print("Google API Key not set (and this is optional)")

# Gemini client
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
openai = OpenAI(api_key=google_api_key, base_url=GEMINI_BASE_URL)

system = [{"role": "system", "content": TWIN_SYSTEM_PROMPT}]

def chat(message, history):
    messages = system + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        response = openai.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
    return response.choices[0].message.content

if __name__ == "__main__":
    # Optional test run before launching UI
    testmessage = [{"role": "user", "content": "What is the color of lotus?"}]
    response = openai.chat.completions.create(model=MODEL_NAME, messages=testmessage)
    print(response.choices[0].message.content)

    gr.ChatInterface(
        fn=chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
