import os
from pathlib import Path
from prefect import flow, task, serve, get_run_logger
from prefect.artifacts import create_markdown_artifact
from datetime import datetime
import time

from pydantic import BaseModel
from logic import call_llm, get_first_n_words
import smtplib
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(module)s %(levelname)s - %(message)s")
file_handler = logging.StreamHandler()
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
from pymongo import MongoClient


class State(BaseModel):
    update_time: str
    text: str

    
class Mongo:
    def __init__(self, local=False):
       self.client = MongoClient('mongodb://root:example@0.0.0.0:27017/')
       self.db = self.client["database"]
       self.collection = self.db['states']

    def insert(self, state: State):
        self.collection.insert_one(state.model_dump())


MONGO = Mongo()

@task
def write_to_mongo(message: str):
    state = State(update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  text=message)

    MONGO.insert(state)


def get_first_n_words(input_string, n):
    words = input_string.split()
    first_n_words = ' '.join(words[:n])
    return first_n_words

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

path_resoucres =  Path(os.getenv("PATH_RESOURCES", "./temp"))
sound_directory = path_resoucres.joinpath(os.getenv("SOUND_DIRECTORY", "sounds"))
sound_directory.mkdir(parents=True, exist_ok=True)


@task
def create_sound(prompt):
    temp_file = sound_directory.joinpath("tmp.wav")
    speech_file = sound_directory.joinpath("speech.wav")

    prompt = get_first_n_words(prompt, 90)
    speech = synthesiser(prompt, forward_params={"speaker_embeddings": speaker_embedding})
    sf.write(str(temp_file), speech["audio"], samplerate=speech["sampling_rate"])

    if speech_file.is_file():
        temp_file.replace(speech_file)
    else:
        temp_file.rename(speech_file)




class MailClient:

    def __init__(self, server, port, user, password):
        self.server = server
        self.port = port
        self.user = user
        self.password = password

    def send(self, mail):
        with smtplib.SMTP(self.server, self.port) as server:
            server.ehlo()
            server.starttls()
            server.login(self.user, self.password)
            message = f"Subject: {mail.subject}\n\n{mail.message}"
            server.sendmail(self.user, mail.receiver, message)


def get_email_client():
    return MailClient("smtp-mail.outlook.com", 587, "felix.jobson@outlook.de", os.getenv('EMAIL_PASSWORD'))


class Mail(BaseModel):
    receiver: str
    subject: str
    message: str

MAILCLIENT = get_email_client()

@task
def send_mail():
    logger = get_run_logger()
    logger.info('this is a test log')

    receiver_email = "hauerf98@gmail.com"
    subject = "CronMail"
    message=f"""
    This is a test mail from a cronjob.
    """

    mail = Mail(receiver="hauerf98@gmail.com",
                subject=subject,
                message=message)

    MAILCLIENT.send(mail)

# TODO: Add temperature argument to functions
# TODO: Only use llm.py drop llm thing in logic.py and use it in llm heartbeat
@task()
def ask_tinydolphin(prompt: str):
    answer = call_llm(prompt, model="tinydolphin")
    return answer

@task()
def ask_tinyllama(prompt: str):
    answer = call_llm(prompt, model="tinyllama")
    return answer

@task()
def ask_phi(prompt: str):
    answer = call_llm(prompt, model="phi:chat")
    return answer

@task()
def ask_llama2(prompt: str):
    answer = call_llm(prompt, model="llama2")
    return answer

@task()
def ask_vicuna(prompt: str):
    answer = call_llm(prompt, model="vicuna")
    return answer

@task()
def ask_neural_chat(prompt: str, temperature: int = 5):
    answer = call_llm(prompt, model="neural-chat", temperature=temperature)
    return answer

@task()
def ask_mistral(prompt: str):
    answer = call_llm(prompt, model="mistral:instruct")
    return answer

@task()
def ask_mistral_openorca(prompt: str):
    answer = call_llm(prompt, model="mistral-openorca")
    return answer




@flow(log_prints=True)
def llm_heartbeat(name: str = "world", goodbye: bool = False):
    logger = get_run_logger()

    prompt = "Give me a motivating stoic quote, that helps me conquer my day and fight my fears."

    answer = ask_neural_chat(prompt) 

    logger.info('llm finished')

    create_sound(answer)

    logger.info('creating sound finished')

    write_to_mongo(answer)

    markdown_report = f"""# LLM Report

## Prompt 

{prompt}


## Answer

{answer}

"""
    create_markdown_artifact(
        key="llm-result",
        markdown=markdown_report,
        description="Result of a LLM prompt",
    )



@flow
def flow_1():
    send_mail()



from datetime import datetime
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document

#--------------old------------------
@task
def load_html(urls): 
    loader = AsyncHtmlLoader(urls)
    html = loader.load()
    return html

@task
def transform_html(html):
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["p"])
    return docs_transformed

#--------------old------------------


@task
def get_urls_from_page(url, subpage):
    logger = get_run_logger()
    response = requests.get(url + subpage)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        all_urls = [a.get('href') for a in soup.find_all('a') if a.get('href')]
        filtered_urls = [url for url in all_urls if url.startswith(subpage)]
        filtered_urls = list(set(filtered_urls))
        urls_to_read = [url + u for u in filtered_urls]
        logger.info(f"extract urls from: {url} {subpage}:")
        return urls_to_read
    else:
        logger.info(f"Failed to fetch the page. Status code: {response.status_code}")
        return [url + subpage]


@task
def parse_url_handball_em(url) -> str:
    """
    Parsing page of sportschau.de. Return a string with all texts of article.
    """
    logger = get_run_logger()
    try:
        time.sleep(2)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find(id="content")
        date = results.find_all("p", class_="metatextline")
        topline = results.find_all("span", class_="seitenkopf__topline")
        headline = results.find_all("span", class_="seitenkopf__headline--text")
        texts = results.find_all("p", class_="textabsatz m-ten m-offset-one l-eight l-offset-two columns twelve")
        html =  date + topline + headline + texts
        texts = [t.text.strip() for t in html] 
        result = ' '.join(texts)
    except Exception as e:
        result = ""
    return result

@task
def parse_url(url) -> str:
    """
    Parsing page of sportschau.de. Return a string with all texts of article.
    """
    logger = get_run_logger()
    try:
        time.sleep(2)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        #results = soup.find(id="content")
        # date = results.find_all("p", class_="metatextline")
        h1_tag = soup.find("h1", class_="seitenkopf__headline")
        return ". ".join(h1_tag.text.strip().split('\n'))
    except Exception as e:
        result = ""
    return result

@task
def split_docs(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=100, length_function=len
    )
    docs = text_splitter.split_documents(documents)
    return docs

@task
def get_context(documents, query, n_results, model_name):
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)
    db = FAISS.from_documents(documents, embedding_function)
    docs = db.similarity_search(query, k=n_results)
    context = [document.page_content for document in docs]
    return '\n\n'.join(context)


@flow(retries=2, retry_delay_seconds=120)
def rag():
    logger = get_run_logger()

    model_name = "distiluse-base-multilingual-cased-v1"
    url = "https://www.sportschau.de"
    subpage = '/handball/em'

    urls = get_urls_from_page(url, subpage)
    texts = [parse_url(url) for url in urls]
    docs = [Document(page_content=t) for t in texts]
    docs_splitted = split_docs(docs)

    current_date = datetime.now().strftime("%d.%m.%Y")
    query = f"Spielte Deutschland heute am {current_date} ein Spiel bei der Handball EM? Und wenn ja gegen wen und wie ist es augegangen?"
    context = get_context(docs_splitted, query, 3, model_name)

    prompt = f"Mit Hilfe von Ausschnitten aus Sportberichten, beantworte eine Frage. Wenn du die Antwort nicht in den Ausschnitten zu finden ist, antworte mit 'Ich kann nicht.' Hier die Ausschnitte: \n{context} \nNun beantworte damit die Frage: {query}."

    markdown_report = f"""# LLM Report

## Query 

{query}

## Answers

### tinyllama  
{ask_tinyllama(prompt)}

### phi  
{ask_phi(prompt)}

### llama2
{ask_llama2(prompt)}

### vicuna
{ask_vicuna(prompt)}   

### mistral
{ask_mistral(prompt)}   

### mistral-openorca 
{ask_mistral_openorca(prompt)}  

### neural-chat   
{ask_neural_chat(prompt)}    


## Context

{context}

## Prompt

{prompt}

## Extracted texts from articles
"""


    urls_texts = [u + ' \n ' + t + '\n\n' for u, t in zip(urls, texts)]
    urls_texts = '\n'.join(urls_texts)
    markdown_report = markdown_report + ' \n\n' + urls_texts

    create_markdown_artifact(
        key="rag-result",
        markdown=markdown_report,
        description="Result of a RAG run",
    )




from rag import get_context
from llm import ask_llm
@flow(retries=2, retry_delay_seconds=120)
def rag_llama_index():
    logger = get_run_logger()

    url = "https://www.sportschau.de"
    subpage = '/fussball/bundesliga'

    urls = get_urls_from_page(url, subpage)
    texts = [parse_url(url) for url in urls]

    current_date = datetime.now().strftime("%d.%m.%Y")
    query = f"Hat der FC Bayern München ein Spiel gewonnen?"

    nodes = get_context(query, texts, chunk_size=100,  similarity_top_k=3)

    context = [node.get_text() for node in nodes]
    context =  '\n\n'.join(context)

    prompt = f"Mit Hilfe von Ausschnitten aus Sportberichten, beantworte eine Frage. Wenn du die Antwort nicht in den Ausschnitten zu finden ist, antworte mit 'Ich kann nicht.' Hier die Ausschnitte: \n{context} \nNun beantworte damit die Frage: {query}."

    markdown_report = f"""# LLM Report

## Query 

{query}

# TODO: try stablelm2 and stablelm2 zephyr
## Answers

### tinydolphin  
{ask_llm(prompt, model="tinydolphin")}


### tinyllama  
#{ask_llm(prompt, model="tinyllama")}

### phi  
#{ask_llm(prompt, model="phi:chat")}

### llama2
#{ask_llm(prompt, model="llama2")}

### vicuna
{ask_llm(prompt, model="vicuna")}

### mistral
{ask_llm(prompt, model="mistral:instruct")}

### mistral-openorca 
{ask_llm(prompt, model="mistral-openorca")}

### neural-chat   
{ask_llm(prompt, model="neural-chat")}

## Context

{context}

## Prompt

{prompt}

## Extracted texts from articles
"""


    urls_texts = [u + ' \n ' + t + '\n\n' for u, t in zip(urls, texts)]
    urls_texts = '\n'.join(urls_texts)
    markdown_report = markdown_report + ' \n\n' + urls_texts

    create_markdown_artifact(
        key="rag-result",
        markdown=markdown_report,
        description="Result of a RAG run",
    )


if __name__ == "__main__":

    deployment_1 = flow_1.to_deployment(name="mail",
                      tags=["messages"],
                      interval=60 * 60 * 24)

    deployment_2 = llm_heartbeat.to_deployment(name="llm-server",
                      tags=["llm"],
                      parameters={"goodbye": True},
                      cron = '0 6 * * *') 

    deployment_3 = rag_llama_index.to_deployment(name="RAG pipeline",
                      tags=["rag"],
                      interval= 60 * 60 * 8)

    serve(deployment_1, deployment_2, deployment_3)

