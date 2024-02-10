from llama_index import Document, VectorStoreIndex
from llama_index.text_splitter import SentenceSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import ServiceContext
from llama_index.embeddings import HuggingFaceEmbedding



def get_context(query: str, texts: list[str], chunk_size: int, similarity_top_k: int):
    documents = [Document(text=t) for t in texts if len(t) > 0]


    embed_model = HuggingFaceEmbedding(
        model_name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer"
    )

    text_splitter = SentenceSplitter(
        paragraph_separator=r"\n\n", chunk_size=chunk_size, chunk_overlap=0
    )

    service_context = ServiceContext.from_defaults(
        embed_model=embed_model, llm=None, text_splitter=text_splitter
    )

    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, show_progress=True
    )

    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    nodes = retriever.retrieve(query)
    return nodes

if __name__ == "__main__":

    import os
    os.environ["LLAMA_INDEX_CACHE_DIR"] = "/tmp/llama_index"
    os.environ["HF_HOME"] = "/tmp/hf"


    text1 = "Dies ist ein Test. Dies ist ein zweiter Test."
    text2 = "Dies ist ein Bericht über Handball. Dies über Volleyball."
    text3 = "Was ist Handball?"
    text4 = "Was sind die Spielregeln beim Handball?"
    text5 = "Frankreich ist Gruppensieger Olympiasieger Frankreich hat sich als drittes Team nach Weltmeister Dänemark und Titelverteidiger Schweden vorzeitig für das Halbfinale bei der Handball-EM qualifiziert und die erhoffte Schützenhilfe für die deutsche Mannschaft geleistet."

    text_list = [text1, text2, text3, text4, text5]

    query = "Wer ist Gruppensieger?"
    nodes = get_context(query, text_list, chunk_size=50,  similarity_top_k=3)


    for node in nodes:
        print(node.get_text())
        print(node.get_score())
        print()


    context = [node.get_text() for node in nodes]
    context =  '\n\n'.join(context)
    print(context)

    quit()




    bullet_splitter = SentenceSplitter(
        paragraph_separator=r"\n\n", chunk_size=50, chunk_overlap=0
    )
    nodes = bullet_splitter(documents)
    slides_parser = SimpleNodeParser.from_defaults(
        text_splitter=bullet_splitter, include_prev_next_rel=True, include_metadata=True
    )
    nodes = slides_parser.get_nodes_from_documents(documents)
