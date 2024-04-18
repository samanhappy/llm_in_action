# LangChain 中如何使用不支持 Function Calling 的模型实现工具

## 引言

本文将介绍如何使用 LangChain 框架构建一个工具调用链，以便在不支持函数调用的模型上使用工具。我将使用一个简单的数学计算工具集作为示例，展示如何通过模型的输出选择并调用相应的工具。

## 环境准备

首先，确保安装了必要的 Python 包：

```bash
pip install --upgrade --quiet langchain langchain-openai
```

接着，使用 `dotenv` 加载环境变量：

```python
import dotenv

dotenv.load_dotenv()
```

## 创建自定义工具

创建一个自定义的乘法工具：

```python
from langchain_core.tools import tool

@tool
def multiply(first: int, second: int) -> int:
    """实现两个整数的乘法运算。"""
    return first * second
```

## 工具调用示例

调用 `multiply` 工具进行计算：

```python
result = multiply.invoke({"first": 4, "second": 5})
print(result)  # 输出结果：20
```

## 构建系统提示

创建一个系统提示，明确告诉模型我有哪些工具可用，以及如何返回工具的调用信息：

```python
from langchain.tools.render import render_text_description
from langchain_core.prompts import ChatPromptTemplate

rendered_tools = render_text_description([multiply])

system_prompt = f"""你是一个可以访问以下工具集的助手。以下是每个工具的名称和描述：

{rendered_tools}

根据用户输入，返回要使用的工具的名称和输入。将你的响应以包含 'name' 和 'arguments' 键的 JSON 格式返回。"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)
```

## 添加输出解析器

使用 `JsonOutputParser` 将模型的输出解析为 JSON 格式：

```python
from langchain_core.result_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

chain = prompt | model | JsonOutputParser()

# 演示如何通过模型调用工具
result = chain.invoke({"input": "计算 13 乘以 4 的结果"})
print(result) # 输出结果：{'name': 'multiply', 'arguments': {'first': 13, 'second': 4}}
```

## 工具链的构建与调用

将模型生成的参数传递给工具链，并调用相应的工具：

```python
from operator import itemgetter

chain = prompt | model | JsonOutputParser() | itemgetter("arguments") | multiply

result = chain.invoke({"input": "计算 13 乘以 4 的结果"}
print(result)  # 输出结果：52
```

## 多工具选择与调用

如果需要从多个工具中选择，可以定义一个工具链函数来实现：

```python
# 定义一个加法工具
@tool
def add(first: int, second: int) -> int:
  """实现两个整数的加法运算。"""
  return first + second

# 定义一个指数运算工具
@tool
def exponentiate(base: int, exponent: int) -> int:
  """实现指数运算。"""
  return base**exponent

# 可用工具列表
tools = [multiply, add, exponentiate]

# 根据输入链式调用工具的函数
def tool_chain(input):
  # 将工具名称映射到工具函数
  tool_map = {tool.name: tool for tool in tools}
  # 根据输入名称选择工具
  choose_tool = tool_map[input["name"]]
  arguments = input["arguments"]
  for key, value in arguments.items():
    # 如果值是字典，递归调用tool_chain
    if isinstance(value, dict):
      arguments[key] = tool_chain(value)
  # 使用参数调用选定的工具
  return choose_tool.invoke(arguments)

chain = prompt | model | JsonOutputParser() | tool_chain

result = chain.invoke({"input": "计算 13 乘以 4 的结果的平方"})
print(result)  # 输出结果：2704
```

## 返回工具输入和输出

在上面的例子中，我们直接打印了整个工具链的最终输出结果；但有时候出于监控和调试的目的，我们可能需要将工具链的输入和输出传递给其他工具链。这时候，可以使用 `RunnablePassthrough.assign` 来返回工具的输入和输出：

```python
from langchain_core.runnables import RunnablePassthrough

# 链接提示、模型、JSON输出解析器和工具链
chain = (
    prompt | model | JsonOutputParser() | RunnablePassthrough.assign(result=tool_chain)
)

result = chain.invoke({"input": "计算 13 乘以 4 的结果的平方"})
print(result)  # 输出示例：{'name': 'exponentiate', 'arguments': {'base': {'name': 'multiply', 'arguments': {'first': 13, 'second': 4}}, 'exponent': 2}, 'result': 2704}
```

## 完整代码

```python
import dotenv
from langchain.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 加载环境变量
dotenv.load_dotenv()


# 定义一个乘法工具
@tool
def multiply(first: int, second: int) -> int:
    """实现两个整数的乘法运算。"""
    return first * second


# 定义一个加法工具
@tool
def add(first: int, second: int) -> int:
    """实现两个整数的加法运算。"""
    return first + second


# 定义一个指数运算工具
@tool
def exponentiate(base: int, exponent: int) -> int:
    """实现指数运算。"""
    return base**exponent


# 可用工具列表
tools = [multiply, add, exponentiate]


# 根据输入链式调用工具的函数
def tool_chain(input):
    # 将工具名称映射到工具函数
    tool_map = {tool.name: tool for tool in tools}
    # 根据输入名称选择工具
    choose_tool = tool_map[input["name"]]
    # 拷贝输入参数
    arguments = input["arguments"].copy()
    for key, value in arguments.items():
        # 如果值是字典，递归调用tool_chain
        if isinstance(value, dict):
            arguments[key] = tool_chain(value)
    # 使用参数调用选定的工具
    return choose_tool.invoke(arguments)


# 渲染工具的文本描述
rendered_tools = render_text_description(tools)

# 定义系统提示
system_prompt = f"""你是一个可以访问以下工具集的助手。以下是每个工具的名称和描述：

{rendered_tools}

根据用户输入，返回要使用的工具的名称和输入。将你的响应以包含 'name' 和 'arguments' 键的 JSON 格式返回。"""

# 从系统和用户消息创建聊天提示模板
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("user", "{input}")]
)

# 创建ChatOpenAI模型
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 链接提示、模型、JSON输出解析器和工具链
chain = (
    prompt | model | JsonOutputParser() | RunnablePassthrough.assign(result=tool_chain)
)

# 使用特定输入调用链并打印输出
result = chain.invoke({"input": "计算 13 乘以 4 的结果的平方"})
print(result)
```

## 结语

本文介绍了如何使用 LangChain 框架构建一个工具调用链，以便在不支持函数调用的模型上使用多个工具。希望这篇文章对你有所帮助！
