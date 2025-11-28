
import os
import json
import requests
from dotenv import load_dotenv
from loguru import logger


load_dotenv()

def zero_gpt_test(text):
    ZERO_GPT_API_KEY = os.getenv("ZERO_GPT_API_KEY")
    ZERO_GPT_URL = os.getenv("ZERO_GPT_URL")
    if ZERO_GPT_API_KEY and ZERO_GPT_URL:

        headers = {
            'ApiKey': ZERO_GPT_API_KEY,
        }

        payload = json.dumps({
                    "input_text": text
                    })
        
        try:
            response = requests.request("POST", ZERO_GPT_URL, headers=headers, data=payload)
            
            if response.status_code == 200:
                data = response.json()
                # GPTZero returns a probability (0 to 1)
                prob = data['data']['fakePercentage']
                return prob* 100
            else:
                logger.warning("Zero GPT API error")
                return -1
                
        except Exception as e:
            logger.error("Zero GPT connection failed")
            return -1
    else:
        if not ZERO_GPT_API_KEY:

            logger.warning("ZERO GPT API KEY not found")
        else:
            logger.warning("ZERO GPT URL not found")
        return -1