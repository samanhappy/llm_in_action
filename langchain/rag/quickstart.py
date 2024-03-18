import dotenv
import bs4
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

# Indexing: Load
loader = PyPDFLoader("https://arxiv.org/pdf/2402.16480.pdf")
docs = loader.load()
print(len(docs))
print(docs[0].page_content)

# Indexing: Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
splits = text_splitter.split_documents(docs)
print(len(splits))
print(splits[0].page_content)
print(splits[1].metadata)

# Indexing: Embed
embedding = OpenAIEmbeddings()
# embedded_query = embedding.embed_query("What was the name mentioned in the conversation?")
# print(embedded_query[:5])

# Indexing: Store
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
query = "如何在开源项目中使用 ChatGPT ?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)

# # Retrieve
# retriever = vectorstore.as_retriever()

# Generate
# prompt = hub.pull("rlm/rag-prompt")
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# rag_chain.invoke("What is Task Decomposition?")

# # cleanup
# vectorstore.delete_collection()
