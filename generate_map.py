import osmnx as ox
import networkx as nx
import random
import os

print("🌍 Downloading the physical road network of Bhopal from OpenStreetMap...")
print("This may take 1-3 minutes depending on your internet connection.")

# 1. Download the drivable road network for Bhopal
G = ox.graph_from_place("Bhopal, Madhya Pradesh, India", network_type="drive")

print(f"✅ Downloaded {len(G.nodes())} intersections.")
print("Injecting Smart Parking Hub infrastructure data...")

# 2. Inject Parking Hub Data
# We turn roughly 5% of all intersections into valid "UrbanLink Smart Hubs"
for node, data in G.nodes(data=True):
    is_hub = random.random() < 0.05
    data['is_parking_hub'] = str(is_hub)
    
    if is_hub:
        data['base_price'] = round(random.uniform(2.0, 15.0), 2)
        data['eco_point_reward'] = random.choice([0, 50, 100, 200])

# 3. Save the graph to the data folder
os.makedirs("data", exist_ok=True)
ox.save_graphml(G, "data/city_graph_with_parking.graphml")

print("✅ Success! City graph securely saved to 'data/city_graph_with_parking.graphml'.")