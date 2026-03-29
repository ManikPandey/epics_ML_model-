import osmnx as ox
import networkx as nx
import os

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

def generate_city_graph(address="Washington Square Park, New York City, New York, USA", dist=1000, network_type="drive"):
    """
    Downloads the street network from OpenStreetMap around a specific address.
    dist = radius in meters (1000m is a good size for testing)
    """
    print(f"Downloading street network for {dist} meters around: {address}...")
    
    # FIX: Using graph_from_address which is much more reliable
    G = ox.graph_from_address(address, dist=dist, network_type=network_type)
    
    print(f"Graph downloaded! Nodes (Intersections): {len(G.nodes)}, Edges (Streets): {len(G.edges)}")
    
    # Save the graph locally so we don't have to download it every time we test
    filepath = "../data/city_graph.graphml"
    ox.save_graphml(G, filepath)
    print(f"Graph saved to {filepath}")
    
    return G

def plot_graph(G):
    """
    Plots the graph to visually verify our digital twin.
    """
    print("Plotting graph...")
    # This will open a window with the map
    fig, ax = ox.plot_graph(G, node_size=2, edge_color='#333333', bgcolor='white')

if __name__ == "__main__":
    # Test location and radius
    test_address = "Washington Square Park, New York City, New York, USA"
    
    city_graph = generate_city_graph(address=test_address, dist=1000)
    plot_graph(city_graph)