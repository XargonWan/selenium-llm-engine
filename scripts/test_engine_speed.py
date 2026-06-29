import time
import requests
import sys

def test_speed(engine="gemini", prompt="Hello, how are you?"):
    url = f"http://localhost:8000/engine/{engine}/prompt"
    payload = {"prompt": prompt}
    
    print(f"Testing engine: {engine}")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=120)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            elapsed = end_time - start_time
            print(f"SUCCESS: Received response in {elapsed:.2f} seconds")
            # print(f"Response: {data['choices'][0]['message']['content'][:100]}...")
        else:
            print(f"FAILURE: Status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    engine = sys.argv[1] if len(sys.argv) > 1 else "gemini"
    test_speed(engine)
