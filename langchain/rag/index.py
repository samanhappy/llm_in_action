import dotenv
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

# Indexing: Load
loader = PyPDFLoader("https://arxiv.org/pdf/2402.16480.pdf")
docs = loader.load()
# print(len(docs))
# print(docs[0].page_content)

# Indexing: Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
splits = text_splitter.split_documents(docs)
# print(len(splits))
# print(splits[0].page_content)
# print(splits[1].metadata)

# Indexing: Embed
embedding = OpenAIEmbeddings()
# embedded_query = embedding.embed_query("What was the name mentioned in the conversation?")
# print(embedded_query[:5])

# Indexing: Store
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
# query = "如何在开源项目中使用 ChatGPT ?"
# docs = vectorstore.similarity_search(query)
# print(docs[0].page_content)

# # Retrieve
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# query = "如何在开源项目中使用 ChatGPT ?"
# docs = retriever.invoke(query)
# print(len(docs))
# print(docs[0].page_content)

# Generate
# 创建 prompt，支持多轮对话
system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt + "\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# 使用 OpenAI 的 gpt-3.5-turbo 模型
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# 格式化文档
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# 构建 chain
rag_chain = (
    RunnablePassthrough.assign(
        context=(lambda x: format_docs(x["context"])),
    )
    | prompt
    | llm
    | StrOutputParser()
)

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


rag_chain_with_history = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


def contextualized_question(input: dict):
    return input["question"]


# 返回源文档
rag_chain_with_source = RunnableParallel(
    {
        "context": contextualized_question | retriever,
        "question": lambda x: x["question"],
    }
).assign(answer=rag_chain_with_history)

# 保存对话历史

# 以流的方式生成答案
# for chunk in rag_chain_with_source.stream("如何在开源项目中使用 ChatGPT ?"):
#     print(chunk, end="", flush=True)

# 一次性生成答案
question1 = "如何在开源项目中使用 ChatGPT ?"
answer1 = rag_chain_with_source.invoke(
    {"question": question1}, config={"configurable": {"session_id": "123"}}
)
print(answer1)

question2 = "我们刚才聊了什么?"
answer2 = rag_chain_with_source.invoke(
    {"question": question2}, config={"configurable": {"session_id": "123"}}
)
print(answer2)

print(store)

# # cleanup
# vectorstore.delete_collection()
