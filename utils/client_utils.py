import json
import os
import requests
import time


def get_oneke_prompt(text, schema):
    system_prompt = '<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n'
    json_instruct = {
        "instruct": f'''
        You are an expert in relationship extraction. 
        Please extract relationship triples that match the schema definition from the input. 
        Return an empty list for relationships that do not exist. 
        Please respond in the format of a JSON string.
        ''',
        "schema": schema,
        "input": text,
    }
    prompt = '[INST] ' + system_prompt + json.dumps(json_instruct, ensure_ascii=False) + ' [/INST]'
    return prompt


def send_vllm_request(prompt, model="OneKE", host="0.0.0.0", port=7000, max_retries=3, max_tokens=1024):
    """
    Send request to vLLM service endpoint and get response
    """
    url = f"http://{host}:{port}/v1/completions"
    
    payload = {
        "model": model,  # Model name from the service
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.9,
        "stop": ["<|endoftext|>", "</s>"]
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["text"].strip()
            else:
                print(f"Warning: Unexpected response format: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"All {max_retries} attempts failed for prompt: {prompt[:100]}...")
                return ""
    
    return ""


def get_kgrl_ie_config():
    [kgrl_ie_host, kgrl_ie_port] = os.environ.get("KGRL_IE_HOST", "0.0.0.0:7000").split(':')
    # kgrl_ie_port = int(os.environ.get("KGRL_IE_PORT", "7000"))
    return kgrl_ie_host, int(kgrl_ie_port)


def get_kgrl_mode_config():
    kgrl_mode = os.environ.get("KGRL_REWARD_MODE", "full")
    return kgrl_mode
