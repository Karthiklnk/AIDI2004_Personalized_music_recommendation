import os
from dotenv import load_dotenv
import openai
import json

load_dotenv()

client = openai.OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1'
)

def extract_music_features(prompt):
    try:
        system_prompt = """You are a music expert. Extract musical features from the user's text.
        Return ONLY a JSON object without any additional text or explanation, in this format:
        {"genres": ["genre1", "genre2"], "valence": 0.5, "energy": 0.5}"""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        result = response.choices[0].message.content.strip()
        
        # Try to extract JSON if response contains markdown code blocks
        if "```" in result:
            # Extract content between code blocks
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                result = result[json_start:json_end]

        # Validate JSON format
        parsed_json = json.loads(result)
        
        # Validate structure
        if not all(key in parsed_json for key in ['genres', 'valence', 'energy']):
            raise ValueError("Missing required fields in response")
            
        return json.dumps(parsed_json, indent=2)

    except Exception as e:
        print(f"Error in extract_music_features: {str(e)}")
        print(f"Raw response: {result if 'result' in locals() else 'No response'}")
        raise


if __name__ == "__main__":
# Test the feature extraction
    test_prompts = [
    "I want happy summer beach music",
        "Sad rainy day jazz",
        "High energy workout music"
    ]
    
    print("Testing music feature extraction...")
    for prompt in test_prompts:
        try:
            print(f"\nTesting prompt: '{prompt}'")
            result = extract_music_features(prompt)
            parsed = json.loads(result)
            print(f"Extracted features: {json.dumps(parsed, indent=2)}")
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")