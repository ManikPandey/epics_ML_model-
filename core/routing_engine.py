import networkx as nx
import osmnx as ox
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. DEEP LEARNING ARCHITECTURES ---

# Model 1: ESTAM (Spatiotemporal Graph Neural Network)
class ESTAM(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32, out_channels=2):
        super(ESTAM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_channels, batch_first=True)
        self.gat = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        lstm_out, (h_n, c_n) = self.lstm(x)
        t_out = h_n.squeeze(0)
        s_out = F.relu(self.gat(t_out, edge_index))
        return torch.sigmoid(self.linear(s_out))

# Model 2: DQN Contextual Bandit (Dynamic Incentives)
class IncentiveBandit(nn.Module):
    def __init__(self):
        super(IncentiveBandit, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 16)
        self.out = nn.Linear(16, 4) # 4 Action Tiers: 50, 100, 250, 500 Eco-Points
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- 2. MODEL LOADERS ---

def load_infrastructure():
    path = "data/city_graph_with_parking.graphml" if os.path.exists("data") else "../data/city_graph_with_parking.graphml"
    return ox.load_graphml(path)

def load_ai_models():
    """Loads all 5 ML files generated from the Colab Research Phase."""
    base = "data/" if os.path.exists("data") else "../data/"
    
    # 1. Load ESTAM
    estam = ESTAM()
    estam.load_state_dict(torch.load(f"{base}estam_bhopal_elite_weights.pth", map_location='cpu'))
    estam.eval()
    
    # 2. Load DQN Bandit
    bandit = IncentiveBandit()
    bandit.load_state_dict(torch.load(f"{base}eco_bandit_weights.pth", map_location='cpu'))
    bandit.eval()
    
    # 3. Load Quantile GBMs (ETA Predictors)
    gbm_med = joblib.load(f"{base}gbm_eta_median.pkl")
    gbm_low = joblib.load(f"{base}gbm_eta_lower.pkl")
    gbm_high = joblib.load(f"{base}gbm_eta_upper.pkl")
    
    return estam, bandit, (gbm_med, gbm_low, gbm_high)

# --- 3. THE UNIFIED TRIPARTITE ENGINE ---

def orchestrate_smart_match(G, models, driver_node, rider_node, edge_index, x_current, user_type, is_raining):
    estam, bandit, gbms = models
    gbm_med, gbm_low, gbm_high = gbms
    
    best_hub = None
    lowest_cost = float('inf')
    best_metrics = {}
    current_hour = datetime.now().hour

    # THE FIX: Run Dijkstra ONCE outward from the users with a 10km cutoff (10000 meters)
    # This takes milliseconds instead of seconds.
    try:
        driver_distances = nx.single_source_dijkstra_path_length(G, driver_node, cutoff=10000, weight='length')
        rider_distances = nx.single_source_dijkstra_path_length(G, rider_node, cutoff=10000, weight='length')
    except Exception as e:
        print(f"Graph traversal error: {e}")
        return None, None

    with torch.no_grad():
        ai_spatial = estam(x_current, edge_index)
        
    hubs = [n for n, data in G.nodes(data=True) if data.get('is_parking_hub') in [True, 'True']]
    node_list = list(G.nodes())
    
    # THE FIX: Only evaluate hubs that are reachable by BOTH users within 10km
    reachable_hubs = [h for h in hubs if h in driver_distances and h in rider_distances]
    
    for hub in reachable_hubs:
        try:
            hub_idx = node_list.index(hub)
            traffic_risk = ai_spatial[hub_idx][0].item()
            p_available = ai_spatial[hub_idx][1].item()
            
            if p_available < 0.10: 
                continue
                
            # Grab the pre-calculated distances instantly! O(1) time complexity.
            dist_driver_m = driver_distances[hub]
            dist_rider_m = rider_distances[hub]
            
            eta_features = pd.DataFrame({
                'Distance_km': [dist_driver_m / 1000.0, dist_rider_m / 1000.0],
                'Traffic_Risk': [traffic_risk, traffic_risk],
                'Hour': [current_hour, current_hour],
                'Is_Raining': [is_raining, is_raining]
            })
            
            etas_med = gbm_med.predict(eta_features)
            etas_high = gbm_high.predict(eta_features)
            
            driver_eta = etas_med[0]
            rider_eta = etas_med[1]
            
            walking_dist_km = min(1.0, (dist_rider_m / 1000.0))
            state_tensor = torch.FloatTensor([[user_type, is_raining, walking_dist_km]])
            
            with torch.no_grad():
                best_action_idx = torch.argmax(bandit(state_tensor)).item()
                
            dynamic_eco_points = [50, 100, 250, 500][best_action_idx]
            
            sync_penalty = abs(driver_eta - rider_eta)
            price = float(G.nodes[hub].get('base_price', 0.0))
            
            total_time = driver_eta + rider_eta
            base_cost = (1.5 * total_time) + (5.0 * sync_penalty) + (2.0 * price) - (0.05 * dynamic_eco_points)
            expected_cost = base_cost / max(p_available, 0.05)
            
            if expected_cost < lowest_cost:
                lowest_cost = expected_cost
                best_hub = hub
                best_metrics = {
                    "distances": {"driver_km": round(dist_driver_m/1000, 2), "rider_km": round(dist_rider_m/1000, 2)},
                    "eta_prediction": {
                        "driver_arrival_mins": round(driver_eta, 1),
                        "rider_arrival_mins": round(rider_eta, 1),
                        "worst_case_delay_mins": round(etas_high[0] - driver_eta, 1)
                    },
                    "ai_parking_probability": round(p_available * 100, 1),
                    "ai_traffic_risk": round(traffic_risk * 100, 1),
                    "dynamic_incentive": {"eco_points_offered": dynamic_eco_points},
                    "hub_price_usd": price
                }
                
        except Exception:
            continue
            
    return best_hub, best_metrics