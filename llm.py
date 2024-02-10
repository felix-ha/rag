import time
import httpx
from ollama import Client
from prefect import task

client = Client(host='http://0.0.0.0:11434', timeout=60 * 15)

@task
def ask_llm(prompt: str, model="tinyllama", temperature=1) -> str:
    try:
        options = {"temperature": temperature, "seed": int(time.time())} 
        response = client.generate(model=model, prompt=prompt, options=options)
        answer = response['response']
        return answer
    except httpx.TimeoutException:
        return "LLM timed out..."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


if __name__ == "__main__":
    prompt = 'Why is the sky blue?'
    answer = ask_llm(prompt, temperature=5)
    print(answer)

