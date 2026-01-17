from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
# from langgraph.prebuilt import add_messages
from langgraph.graph.message import add_messages
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode


class SupportState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # saves conversation history
    should_escalate: bool
    issue_type: str


@tool
def get_order_status(order_id:str):
    """Check the Order status and expected delivery date
    
    Args:
        order_id: The ID of the order
    """
    return {"order_id": order_id, "status": "shipped","expected_delivery": "2026-01-20"}

@tool
def create_ticket(issue:str, priority:str):
    """Create a ticket for the issue
    
    Args:
        issue: The issue description
        priority: The priority of the issue
    """
    return {"ticket_id": "TKT1123", "status": "open", "priority": priority}

tools = [get_order_status, create_ticket]

llm_openai = ChatOpenAI(model="gpt-4.1-nano", temperature=0.4).bind_tools(tools)

def agent_node(state: SupportState):
    """ The Agent Node is responsible for handling the conversation and the tools calls"""

    messages = state["messages"]
    response = llm_openai.invoke(messages)
    return {"messages": [response]}

def should_continue_node(state: SupportState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return {"should_continue": True}
    else:
        return {"should_continue": False}


workflow = StateGraph(SupportState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("should_continue", should_continue_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue_node,
{True: "tools", False: END})

workflow.add_edge("tools", "agent")
workflow.add_edge("agent", END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="support_agent_with_tools.png")
result = app.invoke({"messages": [HumanMessage(content="I want to check the order status for order ID 123456")], "should_escalate": False, "issue_type": "order_status"})
print("*************************************************")
print("Result: ", result)
print("*************************************************")