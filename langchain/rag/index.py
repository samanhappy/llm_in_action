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

# Load environment variables
dotenv.load_dotenv()

# Get environment variables
PDF_URL = os.getenv("PDF_URL")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")


def index(load=True):
  """
  Index the documents for vector store.

  Parameters:
  load (bool): If True, load the documents from PDF_URL, split the documents, embed the splits, and store the embeddings. If False, load the embeddings from the persist directory.

  Returns:
  vector_store: The vector store with the embeddings.
  """
  vector_store = None
  if load:
    # Load the documents from PDF_URL
    loader = PyPDFLoader(PDF_URL)
    docs = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(docs)

    # Embed the chunks
    embedding = OpenAIEmbeddings()

    # Store the embeddings
    vector_store = Chroma.from_documents(
      documents=splits, embedding=embedding, persist_directory="./chroma_db"
    )
  else:
    # Load the embeddings from the persist directory
    vector_store = Chroma(
      persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()
    )
  return vector_store


def main(load=True):
  """
  Main function to run the chatbot.

  Parameters:
  load (bool): If True, index the documents for vector store. If False, load the vector store from the persist directory.

  Returns:
  rag_chain_with_source: The runnable chain with source.
  history_store: The history store.
  """
  # Index the documents for vector store
  vector_store = index(load=load)

  # Retrieve the most similar documents
  retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 6}
  )

  # Generate the chat prompt
  system_prompt = "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."
  prompt = ChatPromptTemplate.from_messages(
    [
      ("system", system_prompt + "\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("human", "{question}"),
    ]
  )

  # Initialize the chatbot
  llm = ChatOpenAI(model_name=OPENAI_MODEL_NAME, temperature=0)

  # Format the documents
  def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

  # Create the runnable chain
  rag_chain = (
    RunnablePassthrough.assign(
      context=(lambda x: format_docs(x["context"])),
    )
    | prompt
    | llm
    | StrOutputParser()
  )

  # Initialize the history store
  history_store = {}

  # Get the session history
  def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in history_store:
      history_store[session_id] = ChatMessageHistory()
    return history_store[session_id]

  # Create the runnable chain with message history
  rag_chain_with_history = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
  )

  # Get the contextualized question
  def contextualized_question(input: dict):
    return input["question"]

  # Create the runnable chain with source
  rag_chain_with_source = RunnableParallel(
    {
      "context": contextualized_question | retriever,
      "question": lambda x: x["question"],
    }
  ).assign(answer=rag_chain_with_history)

  return rag_chain_with_source, history_store


if __name__ == "__main__":
  # Run the main function
  rag_chain_with_source, store = main(should_index=True)
  # Use rag_chain_with_source and store as needed