import requests
import json

def chat_offline(persona, job, text):
    prompt = f'''Please summarize the following text for {persona} who wants to {job}.Only provide the summary without mentioning the persona or job in your response.

Here is the text to summarize:
{text}'''
    
    response = requests.post('http://localhost:11434/api/generate', 
                           json={
                               'model': 'tinyllama',
                               'prompt': prompt,
                               'stream': False
                           })
    
    result = response.json()
    return result['response']