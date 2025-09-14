from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

import requests
import random


load_dotenv()

llm=ChatGoogleGenerativeAI(model='gemini-1.5-flash')



# tools
search_tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add (+), sub (-), mul (*), div (/)
    """
    try:
        if operation in ['add', '+']:
            result = first_num + second_num
        elif operation in ['sub', '-']:
            result = first_num - second_num
        elif operation in ['mul', '*']:
            result = first_num * second_num
        elif operation in ['div', '/']:
            if second_num == 0:
                return {'error': "division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {'error': f'unsupported operation: {operation}'}
        
        return {
            "first_num": first_num,
            "operation": operation,
            "second_num": second_num,
            "result": result
        }
    except Exception as e:
        return {'error': str(e)}

tools=[search_tool, calculator]

# make the llm tool aware
llm_with_tools=llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Chat nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call"""
    message=state['messages']
    response=llm_with_tools.invoke(message)
    return {'messages':[response]}

tool_node=ToolNode(tools)

# graph structure
graph=StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)


graph.add_edge(START, 'chat_node')

# if the llm asked for a tool, go to toolnode else finish
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot=graph.compile()
while True:
    user_input = input("Enter your query (type 'exit' to quit): ")
    
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chatbot. Goodbye!")
        break
    print(user_input)
    out = chatbot.invoke({'messages': [HumanMessage(content=user_input)]})
    print(out['messages'][-1].content)
    print("\n**********************************\n")