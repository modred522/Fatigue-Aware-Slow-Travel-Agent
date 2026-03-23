from typing import Annotated
from langchain_community.chat_models import ChatTongyi
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

load_dotenv()
llm = ChatTongyi(model="qwen-turbo")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

while True:
    user_input = input("你：")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("再见！")
        break
    result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    print("AI：", result["messages"][-1].content)