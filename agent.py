import os
import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv, find_dotenv
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END
from tools import calculate_walking_distance 

# Load environment variables
_ = load_dotenv(find_dotenv(), override=True)

# Initialize the LLM (The Brain)
llm = ChatTongyi(model="qwen-turbo")

# 1. Define the complete TravelState
class TravelState(TypedDict):
    destination: str
    # Use operator.add to append new spots to the list instead of overwriting
    itinerary: Annotated[List[str], operator.add] 
    # Use operator.add to accumulate the total walking distance
    cumulative_distance: Annotated[int, operator.add] 
    current_location: str 
    needs_rest: bool 

# 2. Node A: The Planner
def select_next_spot(state: TravelState):
    print("\n--- [Node: Planner] Thinking about the next destination ---")
    
    # If the itinerary is empty, pick the first scenic spot
    if not state["itinerary"]:
        prompt = f"The user is planning a slow travel trip to {state['destination']}. Recommend ONE famous starting scenic spot. Output only the exact name of the spot, nothing else."
        response = llm.invoke(prompt)
        first_spot = response.content.strip()
        print(f"🧠 Selected starting point: {first_spot}")
        
        # Return state updates: add spot to itinerary and set current location
        return {"itinerary": [first_spot], "current_location": first_spot}
    
    # If already traveling, pick the next spot based on the current location
    current = state["current_location"]
    prompt = f"The user is currently at '{current}' in {state['destination']}. Recommend ONE nearby scenic spot suitable for a relaxing walk. Output only the exact name of the spot, nothing else."
    response = llm.invoke(prompt)
    next_spot = response.content.strip()
    print(f"🧠 Selected next stop: {next_spot}")
    
    return {"itinerary": [next_spot]}

# 3. Node B: The Fatigue Calculator (Integrates the Amap Tool)
def calculate_fatigue(state: TravelState):
    print("\n--- [Node: Calculator] Checking physical exertion ---")
    
    # If there is only one spot, the user hasn't started walking yet
    if len(state["itinerary"]) < 2:
         return {"cumulative_distance": 0, "needs_rest": False}
         
    # The newly added spot is the destination, the previous location is the origin
    destination_spot = state["itinerary"][-1]
    origin_spot = state["current_location"] 
    
    try:
        # Invoke the Amap tool to get real-world distance
        distance = calculate_walking_distance.invoke({
            "origin_name": origin_spot,
            "destination_name": destination_spot
        })
    except Exception as e:
        print(f"⚠️ Tool execution failed, defaulting to 1000m. Error: {e}")
        distance = 1000

    # Fatigue logic: Check if the threshold (e.g., 3000 meters) is exceeded
    total_after_this_walk = state["cumulative_distance"] + distance
    needs_rest = total_after_this_walk > 3000
    
    if needs_rest:
        print(f"🔴 WARNING! Total distance reached {total_after_this_walk}m. Rest required!")
    else:
        print(f"🟢 Energy levels optimal. Total distance: {total_after_this_walk}m.")

    # Return state updates
    return {
        "cumulative_distance": distance, # LangGraph will add this to the total
        "current_location": destination_spot, # Update the current physical location
        "needs_rest": needs_rest
    }

# 4. Conditional Edge: The Router
def route_logic(state: TravelState):
    # If fatigue threshold is met, route to the rest stop node
    if state["needs_rest"]:
        return "rest_node" 
    # If we have successfully planned 3 spots without exhaustion, finish the trip
    elif len(state["itinerary"]) >= 3:
        return "end" 
    # Otherwise, loop back to the planner for the next spot
    else:
        return "continue" 

# 5. Node C: The Rest Stop Finder
def find_rest_stop(state: TravelState):
    print("\n--- [Node: Rest Stop] Finding a place to relax ---")
    current = state["current_location"]
    prompt = f"The user is exhausted near '{current}' in {state['destination']}. Recommend ONE highly-rated cafe or teahouse nearby for resting. Output only the exact name of the shop, nothing else."
    response = llm.invoke(prompt)
    cafe = response.content.strip()
    print(f"☕ Found a rest stop: {cafe}")
    
    return {"itinerary": [f"[Rest Stop] {cafe}"]}

# ==========================================
# Graph Compilation
# ==========================================
workflow = StateGraph(TravelState)

# Add all nodes
workflow.add_node("planner", select_next_spot)
workflow.add_node("calculator", calculate_fatigue)
workflow.add_node("rest_stop", find_rest_stop)

# Set the flow
workflow.set_entry_point("planner")
workflow.add_edge("planner", "calculator")

# Add the conditional branching logic
workflow.add_conditional_edges(
    "calculator",
    route_logic,
    {
        "rest_node": "rest_stop", # Trigger intervention
        "end": END,               # Finish normally
        "continue": "planner"     # Loop back
    }
)

# End the workflow after resting (for this MVP)
workflow.add_edge("rest_stop", END)

# Compile into a runnable application
app = workflow.compile()

# ==========================================
# Test Execution Area
# ==========================================
if __name__ == "__main__":
    # Initialize the starting state
    initial_state = {
        "destination": "杭州", 
        "itinerary": [],
        "cumulative_distance": 0,
        "current_location": "",
        "needs_rest": False
    }
    
    print("🚀 Starting the Fatigue-Aware Slow Travel Agent...")
    final_state = app.invoke(initial_state)
    
    print("\n===============================")
    print("🏁 Final Itinerary:")
    for i, spot in enumerate(final_state["itinerary"]):
        print(f"{i+1}. {spot}")
    print(f"🚶 Total Walking Distance: {final_state['cumulative_distance']} meters")