from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langgraph.graph import END, START, StateGraph

from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()


# System message
sys_msg = SystemMessage(content="You are a helpful assistant.")

llm = ChatOpenAI(model="gpt-4o")

# NODES

# Not the right way to do HIL, just for curiousity
def human_feedback(state: MessagesState):
   inp = input("Human Input: ")
   return {"messages": [HumanMessage(inp)]} 

# Assistant node
def assistant(state: MessagesState):
   return {"messages": [llm.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("human_feedback", human_feedback)

# Define edges: these determine the control flow
builder.add_edge(START, "human_feedback")
builder.add_edge("human_feedback", "assistant")
builder.add_edge("assistant", END)

memory = MemorySaver()
# graph = builder.compile(interrupt_before=["human_feedback"], checkpointer=memory)

graph = builder.compile(checkpointer=memory)
print(graph.get_graph().draw_mermaid())

if __name__ == '__main__':
    thread = {"configurable": {"thread_id": "1"}}

    for event in graph.stream({'messages': []}, thread, stream_mode="values"):
        if event['messages']:
            event['messages'][-1].pretty_print()

