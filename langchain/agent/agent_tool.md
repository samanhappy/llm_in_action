# LangChain 智能体初体验：使用工具

在人工智能领域，Agent 智能体被定义为能够自主理解、规划决策、执行复杂任务的实体。LangChain 中的 Agent 核心思想是使用大型语言模型（LLM）作为推理引擎，动态确定要采取的操作、使用哪些工具以及按什么顺序执行。这与 Chain 不同，Chain 是被硬编码在代码中的一系列操作。在本教程中，我将介绍如何使用 LangChain 创建一个能够使用工具与外部系统（API）进行交互的聊天智能体。

## 环境设置

### 安装Python包

在开始之前，请确保你已经安装了以下 Python 包：

```bash
pip install --upgrade --quiet langchain-openai tavily-python
```

Tavily 是一个用于搜索的 API，可以帮助我们在互联网上查找信息。Tavily 为研究人员提供了 1000 次每月的免费 API 调用额度，可以在 [Tavily 官方网站](https://tavily.com/) 上注册并获取。

![tavily](img/tavily.png)

### 设置环境变量

你需要设置环境变量 `OPENAI_API_KEY` 和 `TAVILY_API_KEY`，这些密钥分别对应 OpenAI 和 Tavily 账户。推荐使用 [python-dotenv](https://pypi.org/project/python-dotenv/) 来管理环境变量。

```bash
pip install python-dotenv
```

在项目根目录下创建一个 `.env` 文件，并添加以下内容：

```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

然后在代码中加载环境变量：

```python
from dotenv import load_dotenv

load_dotenv()
```

## 智能体创建

### 初始化工具和聊天模型

```python
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

# 创建一个列表，包含我们将使用的搜索工具
tools = [TavilySearchResults(max_results=1)]

# 初始化 OpenAI 的聊天模型，这里我们使用了 'gpt-3.5-turbo-1106' 模型
# 'temperature' 参数用于控制生成文本的创造性，值越高，结果越多样化
chat_model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)
```

### 定义提示模板

为了使智能体能够进行对话，我们需要定义一个包含聊天历史占位符的提示模板：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 定义一个提示模板，其中包含聊天历史的占位符
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能助手。你可能不需要为每个查询使用工具 - 用户可能只是想聊天！"),
    MessagesPlaceholder(variable_name="messages"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
```

### 组装智能体

现在我们已经有了初始化的工具和聊天模型，以及定义好的提示模板，接下来我们将组装智能体：

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent

# 创建一个聊天智能体，它结合了聊天模型和工具
agent = create_openai_tools_agent(chat_model, tools, prompt_template)

# 创建一个执行器来运行智能体
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

## 运行和测试

### 运行智能体

设置好智能体后，我们可以尝试与其进行交互。智能体可以处理不需要查找的简单查询：

```python
from langchain_core.messages import HumanMessage

# 运行智能体并获取响应
response = executor.invoke({"messages": [HumanMessage(content="你好，你叫什么名字？")]})
print(response)
```

可以看到响应：

```console
> Entering new AgentExecutor chain...
你好！我是智能助手，很高兴为你服务。有什么我可以帮助你的吗？

> Finished chain.
{'messages': [HumanMessage(content='你好，你叫什么名字？')], 'output': '你好！我是智能助手，很高兴为你服务。有什么我可以帮助你的吗？'}
```

### 测试搜索功能

智能体也可以使用传递的搜索工具来获取最新信息（如果需要的话）：

```python
# 测试搜索功能
search_response = executor.invoke({
    "messages": [HumanMessage(content="最近的科技新闻是什么？")]
})
print(search_response)
```

可以看到响应：

```console
> Entering new AgentExecutor chain...

Invoking: `tavily_search_results_json` with `{'query': 'recent technology news'}`


[{'url': 'https://www.theverge.com/tech', 'content': 'Screensharing mistake gets Nvidia sued over alleged stolen trade secrets\nFiled under:\nThe Verge’s guide to Cyber Monday 2023\nThe best early Cyber Monday deals\nFiled under:\nThe Verge’s guide to Cyber Monday 2023\nThe second-gen Apple Pencil has dropped to a new all-time low\nAdvertiser Content\nThere’s a lengthy transcript of a 1978 recording of George Lucas, Stephen Spielberg, and Lawrence Kasdan brainstorming Indiana Jones and the Raiders of the Lost Ark that’s floated around the internet for the last 14 years or longer.\n [Proton Blog]\nPagination\nThe best Black Friday streaming deals on Max, Paramount Plus, Hulu, and more\nThe best Cyber Monday deals for some of The Verge staff’s favorite stuff\nThe best early Cyber Monday deals\nThe best wireless earbuds to buy right now\nDbrand is suing Casetify for ripping off its Teardown designs\nInstaller\n/ A weekly newsletter about the best and Verge-iest stuff you should know about.\n [Twitter]\nReviews\nThis is the keyboard that will convert you to the low-profile life\nFitbit Charge 6 review: practically a Pixel band\nGaming brands are learning the right lessons from enthusiast mechanical keyboards\nThe best smart holiday string lights\nThe entire story of Twitter / X under Elon Musk\nLook, there’s nothing wrong with taking your phone to the ceramic throne so you can relax, read, and relieve... [Indie Film Hustle®]\nFiled under:\nThe Verge’s guide to Cyber Monday 2023\nRoomba’s newest robot vacuums are up to $400 off for Cyber Monday\nMeta communications director Andy Stone is “wanted under an article of the Criminal Code of the Russian Federation,” according to Russian state news agency TASS, citing an interior ministry\xa0database. [GamesIndustry.biz]\nIt’s clearly time: all the news about the transparent tech renaissance\nThe first of three new Doctor Who episodes is about to premiere at 6:30PM GMT (1:30PM ET, and if you’re not in the UK or Ireland, you’ll find the new episodes on Disney Plus now).'}]最近的科技新闻包括Nvidia因涉嫌窃取商业机密而被起诉，以及Apple Pencil第二代的价格创下历史新低。你可以在以下链接中了解更多：[科技新闻](https://www.theverge.com/tech)

> Finished chain.
{'messages': [HumanMessage(content='最近的科技新闻是什么？')], 'output': '最近的科技新闻包括Nvidia因涉嫌窃取商业机密而被起诉，以及Apple Pencil第二代的价格创下历史新低。你可以在以下链接中了解更多：[科技新闻](https://www.theverge.com/tech)'}
```

## 完整代码

```python
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
            "你是一个智能助手。你可能不需要为每个查询使用工具 - 用户可能只是想聊天！",
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

```

## 资源链接

- [LangChain 官方文档](https://python.langchain.com/docs/use_cases/chatbots/tool_usage/)
- [Tavily 官方文档](https://tavily.com/)
