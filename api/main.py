from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import torch
import networkx as nx
import random
import osmnx as ox

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.routing_engine import orchestrate_smart_match, load_infrastructure, load_ai_models

app = FastAPI(title="UrbanLink Tri-Model ML Engine", description="Industry-grade Orchestration API")

# Global State
CITY_GRAPH = None
MODELS = None
EDGE_INDEX = None
X_CURRENT = None

@app.on_event("startup")
def boot_sequence():
    global CITY_GRAPH, MODELS, EDGE_INDEX, X_CURRENT
    print("🚀 Booting UrbanLink Tri-Model Orchestration Engine...")
    
    # Load Infrastructure
    CITY_GRAPH = load_infrastructure()
    
    # Load all 5 Machine Learning Models
    MODELS = load_ai_models()
    
    # Build Spatial Tensors
    if CITY_GRAPH and MODELS:
        print("Mapping Spatial Index...")
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(CITY_GRAPH.nodes())}
        edges = list(nx.relabel_nodes(CITY_GRAPH, node_mapping).edges())
        EDGE_INDEX = torch.tensor([[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long)
        
        # Simulated live 23-hour traffic heartbeat
        num_nodes = len(CITY_GRAPH.nodes())
        X_CURRENT = torch.zeros((num_nodes, 23, 2)) 
        
    print("✅ System Online. Awaiting Tripartite routing requests.")

# --- 1. NEW GPS-BASED PAYLOADS ---
class RouteRequest(BaseModel):
    driver_lat: float
    driver_lng: float
    rider_lat: float
    rider_lng: float
    user_type: int = 0  # 0 for Student/Eco-conscious, 1 for Executive/Impatient
    is_raining: int = 0 # 0 for Clear, 1 for Rain

class PolylineRequest(BaseModel):
    driver_lat: float
    driver_lng: float
    rider_lat: float
    rider_lng: float
    hub_node_id: int # We keep this as an ID because the backend generated it

# --- 2. THE ROUTING ENDPOINT ---
@app.post("/api/v1/route")
async def execute_unified_route(req: RouteRequest):
    if CITY_GRAPH is None or MODELS is None:
        raise HTTPException(status_code=503, detail="System booting. Models not loaded yet.")

    try:
        # THE MAGIC: Snap raw GPS coordinates to the nearest physical street nodes
        # Note: osmnx expects X = Longitude, Y = Latitude
        driver_node = ox.distance.nearest_nodes(CITY_GRAPH, X=req.driver_lng, Y=req.driver_lat)
        rider_node = ox.distance.nearest_nodes(CITY_GRAPH, X=req.rider_lng, Y=req.rider_lat)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not map GPS coordinates to city streets. {e}")

    # Pass the snapped nodes to our Orchestration Engine
    hub, metrics = orchestrate_smart_match(
        G=CITY_GRAPH, 
        models=MODELS,
        driver_node=driver_node, 
        rider_node=rider_node,
        edge_index=EDGE_INDEX,
        x_current=X_CURRENT,
        user_type=req.user_type,
        is_raining=req.is_raining
    )

    if not hub:
        raise HTTPException(status_code=404, detail="No viable path found considering current AI constraints.")

    # Return the hub's actual GPS coordinates alongside its ID for the frontend
    hub_lat = CITY_GRAPH.nodes[hub]['y']
    hub_lng = CITY_GRAPH.nodes[hub]['x']

    return {
        "status": "success",
        "optimal_hub": {
            "node_id": hub,
            "lat": hub_lat,
            "lng": hub_lng
        },
        "intelligence": metrics
    }

# --- 3. THE POLYLINE ENDPOINT ---
@app.post("/api/v1/get-polylines")
async def get_map_polylines(req: PolylineRequest):
    if CITY_GRAPH is None:
        raise HTTPException(status_code=503, detail="Map infrastructure not loaded.")
        
    try:
        # Snap the GPS coordinates again for the polyline trace
        driver_node = ox.distance.nearest_nodes(CITY_GRAPH, X=req.driver_lng, Y=req.driver_lat)
        rider_node = ox.distance.nearest_nodes(CITY_GRAPH, X=req.rider_lng, Y=req.rider_lat)

        # Calculate the shortest physical paths through the street network
        driver_path = nx.shortest_path(CITY_GRAPH, driver_node, req.hub_node_id, weight='length')
        rider_path = nx.shortest_path(CITY_GRAPH, rider_node, req.hub_node_id, weight='length')
        
        # Extract the Latitude (y) and Longitude (x) for every turn in the path
        driver_coords = [[CITY_GRAPH.nodes[n]['y'], CITY_GRAPH.nodes[n]['x']] for n in driver_path]
        rider_coords = [[CITY_GRAPH.nodes[n]['y'], CITY_GRAPH.nodes[n]['x']] for n in rider_path]
        hub_coords = [CITY_GRAPH.nodes[req.hub_node_id]['y'], CITY_GRAPH.nodes[req.hub_node_id]['x']]
        
        return {
            "status": "success",
            "geometry": {
                "driver_route": driver_coords,
                "rider_route": rider_coords,
                "hub_location": hub_coords
            }
        }
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=404, detail="Could not calculate physical street route.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))