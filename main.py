import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from agent.Agentic_workspace import GraphBuilder 

app = FastAPI()

# Input Request Schema
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query_travel_agent(request: QueryRequest):
    try:
        user_query = request.query
        print(f"User Query Received: {user_query}")
        
        # Initialize Graph and ReAct Agent
        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        # Save Graph Visualization safely
        try:
            png_graph = react_app.get_graph().draw_mermaid_png()
            with open("my_graph.png", "wb") as f:
                f.write(png_graph)
            print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")
        except Exception as img_err:
            print(f"Could not save graph image: {img_err}")

        # Format input messages for LangGraph
        messages = {"messages": [("user", user_query)]}

        # Invoke agent
        output = react_app.invoke(messages)

        # Parse output response
        if isinstance(output, dict) and "messages" in output:
            last_message = output["messages"][-1]
            final_output = last_message.content if hasattr(last_message, "content") else str(last_message)
        else:
            final_output = str(output)

        return {"answer": final_output}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
