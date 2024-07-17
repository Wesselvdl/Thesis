from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory 
from solutions.llm import llm
from langchain.prompts import PromptTemplate
from tools.vector import kg_qa
from tools.fewshot import cypher_qa

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

tools = [
    Tool.from_function(
        name="General Chat",
        description="For handling general queries and conversations that are not specifically covered by other tools. Use this tool for random or miscellaneous questions.",
        func=llm.invoke,
        return_direct=False
    ),
    Tool.from_function(
        name="Cypher QA",
        description="Utilizes Cypher queries to retrieve all nodes related to a specific category, section, or sub-section. Use this tool when the user requests a list of IDs and descriptions or asks for an action plan. For example, 'What are all the topics related to Entire Data Centre?' or 'Write an action plan for the topics 3.2.1 and 7.1.2'.",
        func=cypher_qa,
        return_direct=False
    ),
    Tool.from_function(
        name="Vector Search Index",
        description="Performs a Vector Search to find descriptions matching the user's specific topic query. Use this tool when the user asks about a specific topic or concept. For example, 'What is said in your database about effective Free Cooling?'",
        func=kg_qa,
        return_direct=False
    ),
]

agent_prompt = PromptTemplate.from_template("""
Use the received context to answer the question at the end.

You are a Legal Assistant. You have been asked to provide information on a legal topic. You have access to a database of legal documents and can use the following tools to help you find the information you need.
                                            
TOOLS:
------

You have access to the following tools:
{tools}

- Use Vector Search when looking for similar Regulatory descriptions in your database.
- Use Cypher QA to provide information about categories and underlying nodes using Cypher queries. This tool expects a searchable Category. Expected type like `Entire Data Centre`. Format all answers in a clear list format.
- Use General Chat for general chat not covered by other tools.

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent_executor.invoke({"input": prompt})

    return response['output']