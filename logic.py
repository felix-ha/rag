import os
import json
from datetime import datetime
import time
from time import perf_counter 
import logging
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
import httpx
import logging
from config import Config
from pymongo import MongoClient
import logging
import random 

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        func(*args, **kwargs)
        end = perf_counter()
        logging.info(f"call of {func.__name__} took {(end-start)} sec")
    return wrapper


class State(BaseModel):
    update_time: str
    text: str


class Mongo:
    def __init__(self, local=False):
       if local:
            self.client = MongoClient('mongodb://root:example@0.0.0.0:27017/')
       else:
            self.client = MongoClient('mongodb', 27017, username='root', password='example')
       self.db = self.client["database"]
       self.collection = self.db['states']

    def insert(self, state: State):
        self.collection.insert_one(state.model_dump())


MONGO = Mongo()
#MONGO.insert(State(update_time="-", text="asdf"))

def call_llm(prompt: str, model="mistral:instruct", temperature=0.1) -> str:
    answer = ""
    url_server = "http://0.0.0.0:11434"
    route = "/api/generate"

    options = {"temperature": temperature, "seed": int(time.time())}
    data = {"model": model, "stream": False, "prompt": prompt, "options": options}

    try:
        response = httpx.post(f"{url_server}{route}", json=data, headers={"Content-Type": "application/json"}, timeout=None)
        if response.status_code == 200:
            json_response = response.json()
            answer_raw = json_response['response']
            answer = answer_raw.replace("\n", "").replace("\"", "")
        else:
            time.sleep(30)
            response = httpx.post(f"{url_server}{route}", json=data, headers={"Content-Type": "application/json"}, timeout=None)
            if response.status_code == 200:
                json_response = response.json()
                answer_raw = json_response['response']
                answer = answer_raw.replace("\n", "").replace("\"", "")
            else:
                logging.error(f"Server responded with {response.status_code}")
                answer = response.text
    except:
        logging.exception(f"Server is offline.")

    return answer


def update_state(config: Config) -> None:
    Path(config.resources_path).mkdir(parents=True, exist_ok=True)

    prompt = "Write a small welcome text for my homepage. My name is Felix, i am a Data Scientist and a love to program and want to present my work. Do not use emojis."

   # joke_phi = call_llm(prompt, model="phi")
   # joke_mistral = call_llm(prompt, model="mistral:instruct")
   # joke_llama = call_llm(prompt, model="llama2")
    answer = call_llm(prompt, model="tinyllama")
   # joke_llama_uncensored = call_llm(prompt, model="llama2-uncensored")


    state = State(update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  text=answer)

    MONGO.insert(state)

    logging.info(f'written state {state} to mongodb')

    file_name = Path(config.resources_path, config.state_file)

    with open(file_name, 'w') as f:
        f.write(state.model_dump_json())

    logging.info(f"written state to {file_name}")

    create_sound(answer)


def get_server_data(config: Config):
    state_file = Path(config.resources_path, config.state_file)

    try:
        with open(state_file, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        logging.exception(e)
        json_data = {'update_time': datetime.now(), 'text': "failed"}
        
    return json_data


class Mail(BaseModel):
    receiver: str
    subject: str
    message: str


def send_mail():
    try:
        receiver_email = "hauerf98@gmail.com"
        subject = "CronMail"
        message=f"""
        This is a test mail from a cronjob.
        """

        mail = Mail(receiver="hauerf98@gmail.com",
                    subject=subject,
                    message=message)

        path_resoucres =  Path(os.getenv("PATH_RESOURCES", "./temp"))
        emails_directory = path_resoucres.joinpath(os.getenv("EMAIL_DIRECTORY", "mails"))
        emails_directory.mkdir(parents=True, exist_ok=True)

        file_name = f"{str(int(time.time()))}.json"
        file = emails_directory.joinpath(file_name)
        with open(file, 'w') as f:
            f.write(mail.model_dump_json())

    except Exception as e:
        logging.exception(f"preparing mail failed: {e}")



from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

def get_first_n_words(input_string, n):
    words = input_string.split()
    first_n_words = ' '.join(words[:n])
    return first_n_words

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

@measure_time
def create_sound(prompt):
    try:
        temp_file = Path("static/tmp.wav")
        speech_file = Path("static/speech.wav")

        prompt = get_first_n_words(prompt, 100)
        speech = synthesiser(prompt, forward_params={"speaker_embeddings": speaker_embedding})
        sf.write(str(temp_file), speech["audio"], samplerate=speech["sampling_rate"])

        if speech_file.is_file():
            temp_file.replace(speech_file)
        else:
            temp_file.rename(speech_file)

    except Exception as e:
        logging.exception(f"text to speech failed: {e}")


if __name__ == '__main__':
    mongo = Mongo()

    #mongo.collection.drop()

    for document in mongo.collection.find():
        print(document)
