import os
from dotenv import load_dotenv
import openai

load_dotenv()

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def extract_music_features(prompt):
    messages = [
        {"role": "system", "content": "Extract genres, mood, valence (0-1), and energy (0-1) from music-related prompts."},
        {"role": "user", "content": prompt}
    ]
    res = client.chat.completions.create(model="llama3-70b-8192", messages=messages)
    return res.choices[0].message.content
