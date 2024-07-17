import time
import pandas as pd
import io
from contextlib import redirect_stdout
import streamlit as st
from agent import generate_response, agent_executor

# Function to parse captured logs
def parse_logs(log_output):
    tool_used = None
    cypher_query_generated = []
    final_answer = []
    
    lines = log_output.strip().split('\n')
    capture_cypher = False
    capture_final_answer = False
    
    for line in lines:
        if "Action:" in line and "Cypher QA" in line:
            tool_used = "Cypher"
        elif "Action:" in line and "Vector Search Index" in line:
            tool_used = "Vector"
        
        if "Generated Cypher:" in line:
            capture_cypher = True
            continue
        if capture_cypher:
            if "Full Context:" in line:
                capture_cypher = False
            else:
                cypher_query_generated.append(line.strip())
        
        if "Final Answer:" in line:
            capture_final_answer = True
            final_answer.append(line.split("Final Answer:")[1].strip())
            continue
        if capture_final_answer:
            if line.strip() == "" or line.startswith("> Finished chain"):
                capture_final_answer = False
            else:
                final_answer.append(line.strip())
                
    # Clean up formatting characters
    clean_cypher_query = " ".join(cypher_query_generated).replace("[32;1m[1;3m", "").replace("[0m", "").strip()
    clean_final_answer = " ".join(final_answer).replace("[0m", "").strip()
    
    return tool_used, clean_cypher_query, clean_final_answer

# Function to handle submit and reset session state
def handle_submit(message, iteration):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """
    data = []

    # Handle the response
    with st.spinner('Thinking...'):
        # Start timing the query
        start_time = time.time()
        
        # Capture console output
        log_output = io.StringIO()
        with redirect_stdout(log_output):
            response = generate_response(message)
        
        # Parse the captured logs
        log_contents = log_output.getvalue()
        tool_used, cypher_query_generated, final_answer = parse_logs(log_contents)
        
        # End timing the query
        end_time = time.time()
        
        # Save the data
        data.append({
            "ID": iteration,
            "Tool Used": tool_used,
            "Cypher Query Generated": cypher_query_generated,
            "Final Answer": final_answer,
            "Start-Time": start_time,
            "End-Time": end_time
        })

    return data

# Function to reset the agent and session state
def reset_agent():
    global agent_executor
    # Reinitialize the agent executor
    from agent import agent_executor

# Main function to run multiple iterations
def run_experiment(prompt, iterations):
    all_data = []

    for i in range(iterations):
        # Reset the agent and session state
        reset_agent()
        agent_executor.memory.clear()
        
        if "messages" in st.session_state:
            del st.session_state["messages"]

        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm the Legal Retrieval Bot!  How can I help you?"},
        ]

        # Generate a response and collect data
        result = handle_submit(prompt, i+1)
        all_data.append(result)
        
        # Save results to CSV
        df = pd.DataFrame(all_data)
        df.to_csv("cypher_search_results.csv", index=False)
        
        print(result)
        print(f"Iteration {i+1} completed.")

        # Pause between requests to avoid overloading the system
        time.sleep(5)

# Example prompt and number of iterations
prompt = "Generate an action plan for the regulations with IDs 3.2.8, 4.2.1, and 5.1.1."
iterations = 50

run_experiment(prompt, iterations)