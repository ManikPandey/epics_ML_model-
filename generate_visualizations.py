import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import random
import os
from datetime import datetime
import joblib
import pandas as pd
import numpy as np

# Assuming the architecture from Step 10 exists conceptually/locally for imports or definition
# For this script to run standalone for visualizations, we'll redefine simplified conceptual parts if needed
# or just stick to map-based visualizations as requested

# Load Infrastructure
def load_infrastructure():
    path = "data/city_graph_with_parking.graphml" if os.path.exists("data") else "../data/city_graph_with_parking.graphml"
    if not os.path.exists(path):
        print(f"Error: City graph not found at {path}. Please generate it first.")
        return None
    return ox.load_graphml(path)

# --- VISUALIZATION FUNCTIONS ---

def plot_bhopal_map(G):
    """Generates a plain visualization of the Bhopal road network."""
    print("Generating plain Bhopal map...")
    fig, ax = ox.plot_graph(G, node_size=1, edge_color='#333333', edge_linewidth=0.5, show=False, close=False)
    ax.set_title('Bhopal Road Network', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('bhopal_map.png', dpi=300)
    print("✅ Plain map saved as 'bhopal_map.png'.")
    plt.close()

def plot_parking_spots(G):
    """Highlights all identified Smart Parking Hubs on the Bhopal map."""
    print("Generating map with parking spots...")
    # Identify parking nodes
    parking_nodes = [n for n, data in G.nodes(data=True) if data.get('is_parking_hub') in [True, 'True']]
    
    # Create color list: red for parking, default color for others
    node_colors = ['red' if n in parking_nodes else '#1f77b4' for n in G.nodes()]
    node_sizes = [15 if n in parking_nodes else 1 for n in G.nodes()]
    
    fig, ax = ox.plot_graph(G, node_color=node_colors, node_size=node_sizes, edge_color='#cccccc', edge_linewidth=0.5, show=False, close=False)
    ax.set_title('Bhopal Road Network with Smart Parking Hubs (Red)', fontweight='bold', fontsize=14)
    
    # Add legend
    red_patch = mpatches.Patch(color='red', label='Smart Parking Hub')
    plt.legend(handles=[red_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('bhopal_parking.png', dpi=300)
    print("✅ Parking spots map saved as 'bhopal_parking.png'.")
    plt.close()

def plot_scenario(G):
    """Conceptual visualization of a specific tripartite scenario on the Bhopal map."""
    print("Generating scenario visualization map...")
    
    # Define representative sample coordinates (not using placeholders, using specific coordinates)
    # Pick nodes near coordinates for node plotting overlay
    rider_coord = (23.235, 77.400) # (Lat, Lon)
    driver_coord = (23.250, 77.420)
    hub_coord = (23.242, 77.408)
    
    rider_node = ox.distance.nearest_nodes(G, X=rider_coord[1], Y=rider_coord[0])
    driver_node = ox.distance.nearest_nodes(G, X=driver_coord[1], Y=driver_coord[0])
    hub_node = ox.distance.nearest_nodes(G, X=hub_coord[1], Y=hub_coord[0])
    
    fig, ax = ox.plot_graph(G, node_size=1, edge_color='#dddddd', edge_linewidth=0.5, show=False, close=False)
    ax.set_title('Unified Tripartite Match Scenario', fontweight='bold', fontsize=14)
    
    # Plot scenario points with different colors/markers/sizes
    # Rider: Blue square
    ax.scatter(G.nodes[rider_node]['x'], G.nodes[rider_node]['y'], c='blue', marker='s', s=100, label='Rider Point', edgecolor='black', zorder=5)
    # Driver: Red circle
    ax.scatter(G.nodes[driver_node]['x'], G.nodes[driver_node]['y'], c='red', marker='o', s=100, label='Driver Point', edgecolor='black', zorder=5)
    # Hub: Green diamond
    ax.scatter(G.nodes[hub_node]['x'], G.nodes[hub_node]['y'], c='green', marker='D', s=150, label='Optimal Hub', edgecolor='black', zorder=6)
    
    # Add conceptual paths (simplified, straight lines for conceptual visualization over graph, 
    # as path tracing code is complex overlay)
    ax.plot([G.nodes[rider_node]['x'], G.nodes[hub_node]['x']], [G.nodes[rider_node]['y'], G.nodes[hub_node]['y']], c='blue', linestyle='--', linewidth=2, zorder=4)
    ax.plot([G.nodes[driver_node]['x'], G.nodes[hub_node]['x']], [G.nodes[driver_node]['y'], G.nodes[hub_node]['y']], c='red', linestyle='--', linewidth=2, zorder=4)
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('bhopal_scenario.png', dpi=300)
    print("✅ Scenario map saved as 'bhopal_scenario.png'.")
    plt.close()


def main():
    print("🚀 Starting visualization generation...")
    # Ensure data folder exists
    if not os.path.exists("data"):
        os.makedirs("data")
        print("Created 'data' folder.")

    G = load_infrastructure()
    if G is None: return

    plot_bhopal_map(G)
    plot_parking_spots(G)
    plot_scenario(G)
    # Conceptionally plotting orchestration is less map-based and more conceptual diagram
    # Better to focus on the map visualisations requested which implicitly showcase orchestration result

    print("\n✅ All visualizations generated successfully!")
    print("You can find the following images in your project root folder:")
    print("- bhopal_map.png")
    print("- bhopal_parking.png")
    print("- bhopal_scenario.png")

if __name__ == "__main__":
    # conceptual check for dependencies before running
    try:
        import matplotlib
        import joblib
        import pandas
        import numpy
        import osmnx
        import networkx
    except ImportError as e:
        print(f"Error: Missing dependency. {e}. Please install requirements first.")
        # and conceptual exit
    else:
        main()