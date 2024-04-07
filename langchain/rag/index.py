import os
import dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

PDF_URL = os.getenv("PDF_URL")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")


def index(load=True):
    vector_store = None
    if load:
        # Indexing: Load
        loader = PyPDFLoader(PDF_URL)
        docs = loader.load()

        # Indexing: Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        splits = text_splitter.split_documents(docs)

        # Indexing: Embed
        embedding = OpenAIEmbeddings()

        # Indexing: Store
        vector_store = Chroma.from_documents(
            documents=splits, embedding=embedding, persist_directory="./chroma_db"
        )
    else:
        vector_store = Chroma(
            persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()
        )
    return vector_store


def main(load=True):
    vector_store = index(load=load)

    # Retrieve
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    # Generate
    system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt + "\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["context"])),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    history_store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in history_store:
            history_store[session_id] = ChatMessageHistory()
        return history_store[session_id]

    rag_chain_with_history = RunnableWithMessageHistory(
        runnable=rag_chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    def contextualized_question(input: dict):
        return input["question"]

    rag_chain_with_source = RunnableParallel(
        {
            "context": contextualized_question | retriever,
            "question": lambda x: x["question"],
        }
    ).assign(answer=rag_chain_with_history)

    return rag_chain_with_source, history_store


if __name__ == "__main__":
    rag_chain_with_source, store = main(should_index=True)
    # Use rag_chain_with_source and store as needed
