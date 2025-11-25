import json
import re

def parse_markdown_json(text):
    try:
        # 1. Use Regex to find the content inside the ```json ... ``` blocks
        # This looks for ```json (or just ```), captures everything inside, and ends with ```
        match = re.search(r"```(?:json)?\s*(.*)\s*```", text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
        else:
            json_str = text
            
        return json.loads(json_str)
        
    except (json.JSONDecodeError, AttributeError):
        print(f"Error parsing JSON. Raw text: {text}")
        # Return an empty dict or an error dict so the code doesn't crash
        return {"error": "Failed to parse JSON", "raw_content": text}