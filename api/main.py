# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import sys
# import os

# # Add the parent directory to the path so we can import our core modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from core.routing_engine import find_optimal_parking, load_stateful_graph

# # Initialize the API
# app = FastAPI(title="UrbanLink ML Engine", description="Multi-Objective Routing API", version="1.0")

# # Global variable to keep the graph in RAM (so it doesn't load on every single request)
# CITY_GRAPH = None

# @app.on_event("startup")
# def load_graph_on_startup():
#     """Loads the city grid into server memory when the API starts."""
#     global CITY_GRAPH
#     print("Initializing server: Loading UrbanLink environment...")
    
#     # Check both potential locations for the data folder
#     path_inside_project = "data/city_graph_with_parking.graphml"
#     path_outside_project = "../data/city_graph_with_parking.graphml"
    
#     if os.path.exists(path_inside_project):
#         target_path = path_inside_project
#     elif os.path.exists(path_outside_project):
#         target_path = path_outside_project
#     else:
#         raise FileNotFoundError("Could not find the graphml file! Please check where the data folder was created.")
        
#     CITY_GRAPH = load_stateful_graph(filepath=target_path)
#     print("Environment loaded. Server is ready!")

# # Define the expected JSON payload from the frontend
# class RouteRequest(BaseModel):
#     start_node: int
#     end_node: int
#     w_dist: float = 1.0   # Weight for travel time
#     w_price: float = 10.0 # Weight for parking cost
#     w_eco: float = 0.5    # Weight for gamification points

# @app.post("/api/v1/get-point-c")
# async def get_optimal_point_c(request: RouteRequest):
#     """
#     Endpoint for Node.js backend to request the optimal parking hub.
#     """
#     if CITY_GRAPH is None:
#         raise HTTPException(status_code=503, detail="Server environment is still loading.")

#     # Pass the data to our Multi-Objective DP engine
#     optimal_hub, metrics = find_optimal_parking(
#         CITY_GRAPH, 
#         request.start_node, 
#         request.end_node,
#         w_dist=request.w_dist,
#         w_price=request.w_price,
#         w_eco=request.w_eco
#     )

#     if not optimal_hub:
#         raise HTTPException(status_code=404, detail="Could not find a valid parking route.")

#     # Return the clean JSON response to the website
#     return {
#         "status": "success",
#         "message": "Optimal Point C located.",
#         "data": {
#             "parking_node_id": optimal_hub,
#             "trip_metrics": metrics
#         }
#     }


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.routing_engine import find_optimal_parking, load_stateful_graph, load_ml_model

app = FastAPI(title="UrbanLink ML Engine", description="AI-Augmented Multi-Objective API")

CITY_GRAPH = None
ML_MODEL = None

@app.on_event("startup")
def load_environment():
    global CITY_GRAPH, ML_MODEL
    print("Loading UrbanLink Graph and AI Models...")
    
    graph_path = "data/city_graph_with_parking.graphml" if os.path.exists("data/city_graph_with_parking.graphml") else "../data/city_graph_with_parking.graphml"
    model_path = "data/parking_model.json" if os.path.exists("data/parking_model.json") else "../data/parking_model.json"
    
    CITY_GRAPH = load_stateful_graph(filepath=graph_path)
    ML_MODEL = load_ml_model(filepath=model_path)
    print("🚀 System Online. AI Ready.")

# Updated Request Model to include real-world context
class RouteRequest(BaseModel):
    start_node: int
    end_node: int
    hour_of_day: int     # 0-23
    day_of_week: int     # 0 (Mon) to 6 (Sun)
    is_raining: int      # 0 or 1
    w_dist: float = 1.0
    w_price: float = 10.0
    w_eco: float = 0.5

@app.post("/api/v1/get-point-c")
async def get_optimal_point_c(request: RouteRequest):
    if CITY_GRAPH is None or ML_MODEL is None:
        raise HTTPException(status_code=503, detail="Server or AI model is still loading.")

    # Package the context for the AI
    context = {
        "hour_of_day": request.hour_of_day,
        "day_of_week": request.day_of_week,
        "is_raining": request.is_raining
    }

    optimal_hub, metrics = find_optimal_parking(
        CITY_GRAPH, 
        ML_MODEL,
        request.start_node, 
        request.end_node,
        context=context,
        w_dist=request.w_dist,
        w_price=request.w_price,
        w_eco=request.w_eco
    )

    if not optimal_hub:
        raise HTTPException(status_code=404, detail="Could not find a valid route.")

    return {
        "status": "success",
        "data": {
            "parking_node_id": optimal_hub,
            "trip_metrics": metrics
        }
    }