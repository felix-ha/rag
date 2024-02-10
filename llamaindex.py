import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import os
os.environ["LLAMA_INDEX_CACHE_DIR"] = "/tmp/llama_index"
os.environ["HF_HOME"] = "/tmp/hf"
from llama_index.llms import Ollama
from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.text_splitter import SentenceSplitter


llm = Ollama(base_url ='http://0.0.0.0:11434',
             model="tinyllama",
             temperature=0, 
             additional_kwargs = {'seed': 1},
             request_timeout=60.0 * 5)


#resp = llm.complete("Who is Paul Graham?")
#print(resp)



text1 = "Dies ist ein Test. Dies ist ein zweiter Test."
text2 = "Dies ist ein Bericht über Handball. Dies über Volleyball."
text3 = "Was ist Handball?"
text4 = "Was sind die Spielregeln beim Handball?"
text5 = "Frankreich ist Gruppensieger Olympiasieger Frankreich hat sich als drittes Team nach Weltmeister Dänemark und Titelverteidiger Schweden vorzeitig für das Halbfinale bei der Handball-EM qualifiziert und die erhoffte Schützenhilfe für die deutsche Mannschaft geleistet."

texts = [text1, text2, text3, text4, text5]
documents = [Document(text=t) for t in texts if len(t) > 0]

embed_model = HuggingFaceEmbedding(
    model_name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
)

text_splitter = SentenceSplitter(
    paragraph_separator=r"\n\n", chunk_size=1_000, chunk_overlap=0
)

service_context = ServiceContext.from_defaults(
    embed_model=embed_model, llm=llm, text_splitter=text_splitter
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("Wer wurde Gruppensieger?")
print(response)




