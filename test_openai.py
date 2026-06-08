import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
try:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)
