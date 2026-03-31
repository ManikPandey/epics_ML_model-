import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import numpy as np

def generate_zoomed_visualization():
    print("🌍 Loading Bhopal infrastructure graph...")
    path = "data/city_graph_with_parking.graphml"
    
    if not os.path.exists(path):
        print(f"Error: Could not find {path}. Make sure you are running this in the root folder.")
        return

    # Updated for OSMnx 2.x compatibility
    G = ox.load_graphml(path)

    # 1. DEFINE LOCATIONS (MP Nagar, Bhopal)
    print("📍 Setting coordinates for MP Nagar Commercial Zone...")
    # Using standard text instead of emojis to prevent Font Warnings
    driver_coord = (23.2350, 77.4260)  # Zone 1
    rider_coord = (23.2240, 77.4340)   # Near Rani Kamlapati Station
    hub_coord = (23.2300, 77.4300)     # Central MP Nagar Parking

    # Snap coordinates to street intersections
    driver_node = ox.distance.nearest_nodes(G, X=driver_coord[1], Y=driver_coord[0])
    rider_node = ox.distance.nearest_nodes(G, X=rider_coord[1], Y=rider_coord[0])
    hub_node = ox.distance.nearest_nodes(G, X=hub_coord[1], Y=hub_coord[0])

    print("🚗 Calculating street-level routes...")
    try:
        driver_route = nx.shortest_path(G, driver_node, hub_node, weight='length')
        rider_route = nx.shortest_path(G, rider_node, hub_node, weight='length')
    except nx.NetworkXNoPath:
        print("Error: Could not find a drivable path between these nodes.")
        return

    print("🔍 Calculating dynamic zoom bounding box...")
    lats = [G.nodes[n]['y'] for n in driver_route + rider_route]
    lons = [G.nodes[n]['x'] for n in driver_route + rider_route]

    margin_lat = (max(lats) - min(lats)) * 0.2
    margin_lon = (max(lons) - min(lons)) * 0.2

    # Bounding Box for the crop
    north, south = max(lats) + margin_lat, min(lats) - margin_lat
    east, west = max(lons) + margin_lon, min(lons) - margin_lon

    print("🎨 Rendering high-resolution map...")
    # Plotting routes with updated professional color palette
    fig, ax = ox.plot_graph_routes(
        G, routes=[driver_route, rider_route],
        route_colors=['#e74c3c', '#3498db'], 
        route_linewidths=5,
        node_size=0,
        edge_color='#d5dbdb', 
        edge_linewidth=1.2,
        show=False, close=False,
        figsize=(14, 12)
    )

    # Apply the zoom manually to the axis
    ax.set_ylim(south, north)
    ax.set_xlim(west, east)

    ax.set_title('UrbanLink: MP Nagar Street-Level Routing', fontweight='bold', fontsize=20, pad=20)

    # 2. OVERLAY THE MARKERS (Visual cues instead of emojis)
    ax.scatter(G.nodes[driver_node]['x'], G.nodes[driver_node]['y'], c='#e74c3c', s=350, zorder=5, edgecolors='black', label="Driver")
    ax.scatter(G.nodes[rider_node]['x'], G.nodes[rider_node]['y'], c='#3498db', marker='s', s=350, zorder=5, edgecolors='black', label="Rider")
    ax.scatter(G.nodes[hub_node]['x'], G.nodes[hub_node]['y'], c='#2ecc71', marker='*', s=900, zorder=6, edgecolors='black', label="Hub")

    # 3. ADD CONTEXT BOX (Text-only to avoid Glyph errors)
    props = dict(boxstyle='round,pad=0.8', facecolor='#ffffff', alpha=0.95, edgecolor='#bdc3c7', linewidth=1.5)
    context_text = "LOCATION CONTEXT:\nMP Nagar Commercial District\nBhopal, Madhya Pradesh"
    ax.text(0.03, 0.96, context_text, transform=ax.transAxes, fontsize=14, fontweight='bold', 
            color='#2c3e50', verticalalignment='top', bbox=props, zorder=10)

    # 4. BUILD THE LEGEND
    driver_line = mlines.Line2D([], [], color='#e74c3c', linewidth=5, label='Driver Path')
    rider_line = mlines.Line2D([], [], color='#3498db', linewidth=5, label='Rider Path')
    hub_marker = mlines.Line2D([], [], color='white', marker='*', markerfacecolor='#2ecc71', 
                               markersize=20, markeredgecolor='black', label='Smart Parking Hub')
    
    ax.legend(handles=[driver_line, rider_line, hub_marker], loc='lower right', 
              fontsize=12, framealpha=1.0, edgecolor='#bdc3c7', borderpad=1)

    output_filename = 'bhopal_zoomed_routes.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Map saved as '{output_filename}' without font errors.")
    plt.close()

if __name__ == "__main__":
    generate_zoomed_visualization()