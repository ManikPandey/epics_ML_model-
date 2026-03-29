# import networkx as nx
# import osmnx as ox
# import os
# import random

# def load_stateful_graph(filepath="../data/city_graph_with_parking.graphml"):
#     """Loads our city grid that contains the injected parking hubs."""
#     if not os.path.exists(filepath):
#         filepath = "data/city_graph_with_parking.graphml"
#     print(f"Loading graph from {filepath}...")
#     return ox.load_graphml(filepath)

# def find_optimal_parking(G, start_node, end_node, w_dist=1.0, w_price=10.0, w_eco=0.5):
#     """
#     The Multi-Objective Optimizer.
#     w_dist: Weight for distance (how much the user hates driving/walking)
#     w_price: Weight for price (how much the user wants to save money)
#     w_eco: Weight for Eco-Points (how much the user cares about gamification)
#     """
#     best_hub = None
#     lowest_cost_score = float('inf')
#     best_metrics = {}

#     # 1. Extract all nodes that are designated as parking hubs
#     hubs = [n for n, data in G.nodes(data=True) if data.get('is_parking_hub') in [True, 'True']]
#     print(f"Evaluating {len(hubs)} Smart Parking Hubs for the optimal route...")

#     # 2. Iterate through every hub and calculate the cost function J(C)
#     for hub in hubs:
#         try:
#             # Calculate shortest path distance in meters using OSMnx 'length' weight
#             dist_ac = nx.shortest_path_length(G, start_node, hub, weight='length')
#             dist_cb = nx.shortest_path_length(G, hub, end_node, weight='length')
#             total_distance = dist_ac + dist_cb
            
#             # Extract the business logic from the node
#             price = float(G.nodes[hub].get('base_price', 0.0))
#             eco_points = float(G.nodes[hub].get('eco_point_reward', 0.0))
            
#             # THE MATH: Calculate J(C)
#             # Distance + (Price * Price_Weight) - (EcoPoints * Eco_Weight)
#             cost = (w_dist * total_distance) + (w_price * price) - (w_eco * eco_points)
            
#             # If this hub has the lowest score so far, save it!
#             if cost < lowest_cost_score:
#                 lowest_cost_score = cost
#                 best_hub = hub
#                 best_metrics = {
#                     "total_distance_meters": round(total_distance, 2),
#                     "price_usd": price,
#                     "eco_points_earned": eco_points,
#                     "cost_score": round(cost, 2)
#                 }
                
#         except nx.NetworkXNoPath:
#             # In a real city grid, one-way streets might make a path impossible. Skip it.
#             continue
            
#     return best_hub, best_metrics

# if __name__ == "__main__":
#     G = load_stateful_graph()
    
#     # Let's pick two random nodes to represent where the User is (A) and where they want to go (B)
#     nodes = list(G.nodes())
#     start_node = random.choice(nodes)
#     end_node = random.choice(nodes)
    
#     print(f"\nTrip Request: Start Node [{start_node}] --> End Node [{end_node}]")
    
#     # Run the engine!
#     optimal_hub, metrics = find_optimal_parking(G, start_node, end_node)
    
#     if optimal_hub:
#         print("\n✅ --- OPTIMAL 'POINT C' FOUND --- ✅")
#         print(f"Parking Hub Node ID:   {optimal_hub}")
#         print(f"Total Travel Distance: {metrics['total_distance_meters']} meters")
#         print(f"Parking Price:         ${metrics['price_usd']}")
#         print(f"Eco-Points Rewarded:   {metrics['eco_points_earned']}")
#         print(f"Algorithm Cost Score:  {metrics['cost_score']} (Lowest is best)")
#     else:
#         print("Could not find a valid route to any parking hub.")


import networkx as nx
import osmnx as ox
import os
import random
import xgboost as xgb
import pandas as pd

def load_stateful_graph(filepath="../data/city_graph_with_parking.graphml"):
    if not os.path.exists(filepath):
        filepath = "data/city_graph_with_parking.graphml"
    return ox.load_graphml(filepath)

def load_ml_model(filepath="../data/parking_model.json"):
    """Loads the trained XGBoost model into memory."""
    if not os.path.exists(filepath):
        filepath = "data/parking_model.json"
    
    if not os.path.exists(filepath):
        print("Warning: ML Model not found. Did you run train_parking_model.py?")
        return None
        
    model = xgb.XGBClassifier()
    model.load_model(filepath)
    return model

def find_optimal_parking(G, ml_model, start_node, end_node, context, w_dist=1.0, w_price=10.0, w_eco=0.5):
    """
    AI-Augmented Multi-Objective Optimizer.
    'context' contains current real-world conditions (time, weather).
    """
    best_hub = None
    lowest_cost_score = float('inf')
    best_metrics = {}

    hubs = [n for n, data in G.nodes(data=True) if data.get('is_parking_hub') in [True, 'True']]

    for hub in hubs:
        try:
            dist_ac = nx.shortest_path_length(G, start_node, hub, weight='length')
            dist_cb = nx.shortest_path_length(G, hub, end_node, weight='length')
            total_distance = dist_ac + dist_cb
            
            price = float(G.nodes[hub].get('base_price', 0.0))
            eco_points = float(G.nodes[hub].get('eco_point_reward', 0.0))
            
            # 1. Ask the AI for the probability of availability
            p_available = 1.0 # Default if no model
            if ml_model:
                # Format the data exactly as the model expects it
                features = pd.DataFrame([{
                    'hour_of_day': context['hour_of_day'],
                    'day_of_week': context['day_of_week'],
                    'is_raining': context['is_raining'],
                    'base_price': price
                }])
                # predict_proba returns [[P(Class 0), P(Class 1)]]
                p_available = ml_model.predict_proba(features)[0][1]
                
                # Prevent division by zero for completely full spots
                p_available = max(p_available, 0.01)

            # 2. THE NEW MATH: Calculate AI-Augmented Expected Cost
            base_cost = (w_dist * total_distance) + (w_price * price) - (w_eco * eco_points)
            
            # If base_cost is negative (high eco points), shift it to positive for accurate math, 
            # or just apply probability penalty as an addition. Let's use a division modifier.
            expected_cost = base_cost / p_available
            
            if expected_cost < lowest_cost_score:
                lowest_cost_score = expected_cost
                best_hub = hub
                best_metrics = {
                    "total_distance_meters": round(total_distance, 2),
                    "price_usd": price,
                    "eco_points_earned": eco_points,
                    "ai_probability_available": round(p_available * 100, 2),
                    "base_cost": round(base_cost, 2),
                    "expected_cost": round(expected_cost, 2)
                }
                
        except nx.NetworkXNoPath:
            continue
            
    return best_hub, best_metrics