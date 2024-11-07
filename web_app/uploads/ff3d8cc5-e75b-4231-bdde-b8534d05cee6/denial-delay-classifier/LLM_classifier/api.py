from openai import OpenAI
import os

class APIClient():
    def __init__(self):
        self.default_model = "gpt-4"
        self.temperature = 0
        self.max_tokens = 1500

    def get_response(self, instructions, prompt, model=None, temperature=None, max_tokens=None):
        
        if not model:
            model = self.default_model
        if temperature is None:
            temperature = self.temperature
        if not max_tokens:
            max_tokens = self.max_tokens

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        messages = instructions + [{'role': 'user', 'content': prompt}]

        response_object = client.chat.completions.create(model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature)

        response = response_object.choices[0].message.content

        return response
