from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

# 创建一个列表，包含我们将使用的搜索工具
tools = [TavilySearchResults(max_results=1)]

# 初始化 OpenAI 的聊天模型，这里我们使用了 'gpt-3.5-turbo-1106' 模型
# 'temperature' 参数用于控制生成文本的创造性，值越高，结果越多样化
chat_model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# 定义一个提示模板，其中包含聊天历史的占位符
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "您是一个智能助手。您可能不需要为每个查询使用工具 - 用户可能只是想聊天！",
        ),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 创建一个聊天智能体，它结合了聊天模型和工具
agent = create_openai_tools_agent(chat_model, tools, prompt_template)

# 创建一个执行器来运行智能体
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 运行智能体并获取响应
response = executor.invoke({"messages": [HumanMessage(content="你好，你叫什么名字？")]})
print(response)

# 测试搜索功能
search_response = executor.invoke(
    {"messages": [HumanMessage(content="最近的科技新闻是什么？")]}
)
print(search_response)
