import networkx as nx
import osmnx as ox
import random
import os

def inject_parking_hubs(input_filepath="data/city_graph.graphml", hub_percentage=0.05):
    """
    Loads the city graph, selects nodes to be parking hubs, 
    and injects business logic (capacity, price, eco-points).
    """
    # Fix the path depending on where the user runs the script from
    if not os.path.exists(input_filepath):
        input_filepath = "../data/city_graph.graphml"
        
    print(f"Loading graph from {input_filepath}...")
    G = ox.load_graphml(input_filepath)

    nodes = list(G.nodes())
    num_hubs = int(len(nodes) * hub_percentage)

    print(f"Total intersections: {len(nodes)}. Converting {num_hubs} nodes into Smart Parking Hubs...")

    # Randomly select a subset of nodes to serve as parking hubs
    hub_nodes = set(random.sample(nodes, num_hubs))

    # Inject data into every node in the graph
    for node in G.nodes():
        if node in hub_nodes:
            G.nodes[node]['is_parking_hub'] = True
            G.nodes[node]['capacity'] = random.randint(10, 50)          # Number of spots
            G.nodes[node]['base_price'] = round(random.uniform(5.0, 25.0), 2) # Cost to park ($)
            G.nodes[node]['eco_point_reward'] = random.randint(50, 200) # Gamification points
        else:
            G.nodes[node]['is_parking_hub'] = False
            G.nodes[node]['capacity'] = 0
            G.nodes[node]['base_price'] = 0.0
            G.nodes[node]['eco_point_reward'] = 0

    # Save the updated graph with our custom business logic
    output_filepath = input_filepath.replace(".graphml", "_with_parking.graphml")
    ox.save_graphml(G, output_filepath)
    
    print(f"Success! Enhanced graph saved to {output_filepath}")
    return G, hub_nodes

if __name__ == "__main__":
    G, hubs = inject_parking_hubs()
    print("Step 2 Complete! The environment is now ready for the Multi-Objective DP Routing Algorithm.")