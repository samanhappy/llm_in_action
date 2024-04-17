import dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


@tool
def multiply(first: int, second: int) -> int:
    "将两个整数相乘。"
    return first * second


@tool
def add(first_int: int, second_int: int) -> int:
    "将两个整数相加。"
    return first_int + second_int


@tool
def exponentiate(base: int, exponent: int) -> int:
    "求底数的幂次方。"
    return base**exponent


tools = [multiply, add, exponentiate]

llm = ChatOpenAI(model="gpt-3.5-turbo")

llm_with_tools = llm.bind_tools(tools)

# 1，使用 chain 调用工具
chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
result = chain.invoke("6 乘以 8 等于多少?")
print("Result:", result)

# 2，使用智能体调用工具

# 获取并打印智能体提示
prompt = hub.pull("hwchase17/openai-tools-agent")
print("Agent Prompt:", prompt)

# 构建工具调用智能体
agent = create_openai_tools_agent(llm, tools, prompt)

# 创建智能体执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 使用智能体执行复杂计算
output = agent_executor.invoke(
    {"input": "3 的五次方乘以 12 加 3 的和，然后对整个结果求平方"}
)
print("Agent Output:", output)
