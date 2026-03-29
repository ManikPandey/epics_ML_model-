import osmnx as ox
import networkx as nx
import os

def plot_parking_hubs(filepath="../data/city_graph_with_parking.graphml"):
    """
    Loads the stateful graph and plots it, highlighting the parking hubs.
    """
    # Adjust path if running from the root folder
    if not os.path.exists(filepath):
        filepath = "data/city_graph_with_parking.graphml"
        
    print(f"Loading stateful graph from {filepath}...")
    G = ox.load_graphml(filepath)

    # Prepare lists to hold the colors and sizes for each node
    node_colors = []
    node_sizes = []
    
    # Count how many hubs we have for the console output
    hub_count = 0

    for node, data in G.nodes(data=True):
        # GraphML saves booleans as strings ('True' / 'False'), so we check for both
        is_hub = data.get('is_parking_hub', 'False')
        
        if is_hub == True or is_hub == 'True':
            node_colors.append('#ff0000')  # Red for Parking Hubs
            node_sizes.append(30)          # Make hubs larger
            hub_count += 1
        else:
            node_colors.append('#999999')  # Grey for normal intersections
            node_sizes.append(2)           # Make normal intersections tiny

    print(f"Found {hub_count} Smart Parking Hubs. Rendering map...")

    # Plot the graph with our custom colors (removed the 'title' argument)
    fig, ax = ox.plot_graph(
        G, 
        node_color=node_colors, 
        node_size=node_sizes, 
        edge_color='#cccccc', 
        edge_linewidth=0.5,
        bgcolor='white'
    )

if __name__ == "__main__":
    plot_parking_hubs()