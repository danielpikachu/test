import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. Basic Configuration --------------------------
plt.switch_backend('Agg')  # Solve Streamlit matplotlib rendering issues

# Define color constants
COLORS = {
    'building': {'A': 'lightblue', 'C': 'lightcoral'},
    'floor_z': {-6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},
    'corridor_line': {'A': 'cyan', 'C': 'salmon'},
    'corridor_node': 'navy',
    'corridor_label': 'darkblue',
    # Different colors for different stairs
    'stair': {
        'Stairs1': '#FF5733',   # Orange-red
        'Stairs2': '#33FF57',   # Green
        'Stairs3': '#3357FF',   # Blue
        'Stairs4': '#FF33F5',   # Pink-purple
        'Stairs5': '#F5FF33',   # Yellow
        'Stairs6': '#33FFF5',   # Cyan
        'Stairs7': '#FF9933',   # Orange
        'Stairs8': '#9933FF',   # Purple
        'Stairs9': '#F533FF',   # Magenta
        'Stairs10': '#33FF99'   # Teal-green
    },
    'stair_label': 'darkred',  # Stair label color
    'classroom_label': 'black',
    'path': 'darkred',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold'
}

# -------------------------- 2. Core Function Implementation --------------------------
# Read JSON data
def load_school_data_detailed(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data file: {str(e)}")
        return None

# Plot 3D map
def plot_3d_map(school_data):
    # Enlarge figure size
    fig = plt.figure(figsize=(35, 30))
    ax = fig.add_subplot(111, projection='3d')

    # Enlarge axis tick labels
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # Iterate through all buildings
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        building_data = school_data[building_id]
        
        # Process each floor in the building
        for level in building_data['levels']:
            z = level['z']
            level_name = level['name']
            floor_border_color = COLORS['floor_z'].get(z, 'gray')
            building_fill_color = COLORS['building'][building_name]

            # Draw floor plane (thick border)
            fp = level['floorPlane']
            plane_vertices = [
                [fp['minX'], fp['minY'], z],
                [fp['maxX'], fp['minY'], z],
                [fp['maxX'], fp['maxY'], z],
                [fp['minX'], fp['maxY'], z],
                [fp['minX'], fp['minY'], z]
            ]
            x_plane = [p[0] for p in plane_vertices]
            y_plane = [p[1] for p in plane_vertices]
            z_plane = [p[2] for p in plane_vertices]
            
            ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=4, 
                    label=f"Building {building_name}-{level_name}" if f"Building {building_name}-{level_name}" not in ax.get_legend_handles_labels()[1] else "")
            ax.plot_trisurf(x_plane[:-1], y_plane[:-1], z_plane[:-1], 
                            color=building_fill_color, alpha=0.3)

            # Draw corridors (thick lines)
            for corr_idx, corridor in enumerate(level['corridors']):
                points = corridor['points']
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                z_coords = [p[2] for p in points]
                
                if 'name' in corridor and ('connectToBuilding' in corridor['name']):
                    corr_line_color = COLORS['connect_corridor']
                    corr_line_width = 12  # Enlarge inter-building corridor line width
                else:
                    corr_line_color = COLORS['corridor_line'][building_name]
                    corr_line_width = 8  # Enlarge regular corridor line width
                    corr_label = None
                
                ax.plot(x, y, z_coords, color=corr_line_color, linewidth=corr_line_width, 
                        alpha=0.8, label=corr_label if (corr_label and corr_label not in ax.get_legend_handles_labels()[1]) else "")
                
                # Corridor nodes (no labels)
                for p_idx, (px, py, pz) in enumerate(points):
                    ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=40, marker='s', alpha=0.9)

            # Draw stairs (different colors for different stairs)
            for stair in level['stairs']:
                stair_name = stair['name']
                x, y, _ = stair['coordinates']
                stair_label = f"Building {building_name}-{stair_name}"
                
                # Get color based on stair name, default to red if no match
                stair_color = COLORS['stair'].get(stair_name, 'red')
                
                # Draw stairs
                if stair_label not in ax.get_legend_handles_labels()[1]:
                    ax.scatter(x, y, z, color=stair_color, s=600, marker='^', label=stair_label)
                else:
                    ax.scatter(x, y, z, color=stair_color, s=600, marker='^')
                
                # Draw stair labels
                ax.text(x, y, z, stair_name, color=COLORS['stair_label'], fontweight='bold', fontsize=14)

            # Draw classrooms (enlarged size)
            for classroom in level['classrooms']:
                x, y, _ = classroom['coordinates']
                width, depth = classroom['size']
                class_name = classroom['name']

                ax.text(x, y, z, class_name, color=COLORS['classroom_label'], fontweight='bold', fontsize=14)
                ax.scatter(x, y, z, color=building_fill_color, s=160, edgecolors=floor_border_color)
                ax.plot([x, x + width, x + width, x, x],
                        [y, y, y + depth, y + depth, y],
                        [z, z, z, z, z],
                        color=floor_border_color, linestyle='--', alpha=0.6, linewidth=2)

    # Enlarge axis labels and title
    ax.set_xlabel('X Coordinate', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=18, fontweight='bold')
    ax.set_zlabel('Floor Height (Z value)', fontsize=18, fontweight='bold')
    ax.set_title('Campus 3D Navigation Map (Supports Inter-building Navigation between A/C)', fontsize=24, fontweight='bold', pad=20)
    
    # Enlarge legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=2)

    return fig, ax

# Custom graph data structure
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}  # Multiple mapping relationships

    def add_node(self, building_id, node_type, name, level, coordinates):
        building_name = building_id.replace('building', '')
        
        # Concise node ID
        if node_type == 'corridor':
            node_id = f"{building_name}-corr-{name}@{level}"
        else:
            node_id = f"{building_name}-{node_type}-{name}@{level}"
        
        # Store node information
        self.nodes[node_id] = {
            'building': building_name,
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        
        # Establish multiple mappings
        map_key = (building_id, node_type, name, level)
        self.node_id_map[map_key] = node_id
        # Additional mapping for classrooms
        if node_type == 'classroom':
            class_key = (building_name, name, level)
            self.node_id_map[class_key] = node_id
            
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

# Calculate Euclidean distance
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b)** 2 for a, b in zip(coords1, coords2)))

# Build navigation graph (fixed inter-building and internal connections)
def build_navigation_graph(school_data):
    graph = Graph()

    # Step 1: Add all building nodes (classrooms, stairs, corridors)
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_data = school_data[building_id]
        building_name = building_id.replace('building', '')
        
        for level in building_data['levels']:
            level_name = level['name']

            # 1. Add classroom nodes
            for classroom in level['classrooms']:
                class_name = classroom['name']
                graph.add_node(
                    building_id=building_id,
                    node_type='classroom',
                    name=class_name,
                    level=level_name,
                    coordinates=classroom['coordinates']
                )

            # 2. Add stair nodes
            for stair in level['stairs']:
                graph.add_node(
                    building_id=building_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

            # 3. Add corridor nodes (including inter-building corridors)
            for corr_idx, corridor in enumerate(level['corridors']):
                corr_name = corridor.get('name', f'corr{corr_idx}')
                for p_idx, point in enumerate(corridor['points']):
                    corridor_point_name = f"{corr_name}-p{p_idx}"
                    graph.add_node(
                        building_id=building_id,
                        node_type='corridor',
                        name=corridor_point_name,
                        level=level_name,
                        coordinates=point
                    )

    # Step 2: Add all connection relationships
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_data = school_data[building_id]
        building_name = building_id.replace('building', '')

        for level in building_data['levels']:
            level_name = level['name']
            
            # Get all corridor nodes for current building and level
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]

            # 1. Connect nodes within the same corridor
            for corr_idx, corridor in enumerate(level['corridors']):
                corr_name = corridor.get('name', f'corr{corr_idx}')
                corr_points = corridor['points']
                for p_idx in range(len(corr_points) - 1):
                    current_point_name = f"{corr_name}-p{p_idx}"
                    next_point_name = f"{corr_name}-p{p_idx + 1}"
                    current_node_id = graph.node_id_map.get((building_id, 'corridor', current_point_name, level_name))
                    next_node_id = graph.node_id_map.get((building_id, 'corridor', next_point_name, level_name))
                    
                    if current_node_id and next_node_id:
                        coords1 = graph.nodes[current_node_id]['coordinates']
                        coords2 = graph.nodes[next_node_id]['coordinates']
                        distance = euclidean_distance(coords1, coords2)
                        graph.add_edge(current_node_id, next_node_id, distance)

            # 2. Connect nodes between different corridors
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2)
                    
                    if distance < 3.0:
                        graph.add_edge(node1_id, node2_id, distance)

            # 3. Connect classrooms to nearest corridor nodes (enhanced logic)
            class_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'classroom' 
                and node_info['level'] == level_name
            ]
            for class_node_id in class_nodes:
                class_coords = graph.nodes[class_node_id]['coordinates']
                min_dist = float('inf')
                nearest_corr_node_id = None
                
                for corr_node_id in corr_nodes:
                    corr_coords = graph.nodes[corr_node_id]['coordinates']
                    dist = euclidean_distance(class_coords, corr_coords)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(class_node_id, nearest_corr_node_id, min_dist)
                else:
                    # Add warning for isolated classrooms
                    st.warning(f"Warning: Classroom {graph.nodes[class_node_id]['name']} in Building {building_name}{level_name} has no corridor connection")

            # 4. Connect stairs to nearest corridor nodes
            stair_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'stair' 
                and node_info['level'] == level_name
            ]
            for stair_node_id in stair_nodes:
                stair_coords = graph.nodes[stair_node_id]['coordinates']
                min_dist = float('inf')
                nearest_corr_node_id = None
                
                for corr_node_id in corr_nodes:
                    corr_coords = graph.nodes[corr_node_id]['coordinates']
                    dist = euclidean_distance(stair_coords, corr_coords)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(stair_node_id, nearest_corr_node_id, min_dist)

        # 5. Connect nodes across floors in the same building
        for connection in building_data['connections']:
            from_obj_name, from_level = connection['from']
            to_obj_name, to_level = connection['to']
            
            from_obj_type = 'stair' if from_obj_name.startswith('Stairs') else 'corridor'
            to_obj_type = 'stair' if to_obj_name.startswith('Stairs') else 'corridor'
            
            if from_obj_type == 'corridor':
                from_obj_name = f"{from_obj_name}-p0"
            if to_obj_type == 'corridor':
                to_obj_name = f"{to_obj_name}-p0"
            
            from_node_id = graph.node_id_map.get((building_id, from_obj_type, from_obj_name, from_level))
            to_node_id = graph.node_id_map.get((building_id, to_obj_type, to_obj_name, to_level))
            
            if from_node_id and to_node_id:
                graph.add_edge(from_node_id, to_node_id, 5.0)

    # 6. Connect nodes across buildings (fixed level1 connection)
    a_building_id = 'buildingA'
    c_building_id = 'buildingC'
    
    # Fix level1 inter-building connection
    connect_level1 = 'level1'
    a_corr1_name = 'connectToBuildingC-p3'  # Correct node index
    a_connect1_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr1_name, connect_level1))
    c_corr1_name = 'connectToBuildingA-p0'
    c_connect1_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr1_name, connect_level1))
    
    if a_connect1_node_id and c_connect1_node_id:
        coords_a = graph.nodes[a_connect1_node_id]['coordinates']
        coords_c = graph.nodes[c_connect1_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_c)
        graph.add_edge(a_connect1_node_id, c_connect1_node_id, distance)
    else:
        st.warning("Level1 inter-building corridor connection nodes not found")
    
    # Keep level3 inter-building connection unchanged
    connect_level3 = 'level3'
    a_corr3_name = 'connectToBuildingC-p2'
    a_connect3_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr3_name, connect_level3))
    c_corr3_name = 'connectToBuildingA-p0'
    c_connect3_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr3_name, connect_level3))
    
    if a_connect3_node_id and c_connect3_node_id:
        coords_a = graph.nodes[a_connect3_node_id]['coordinates']
        coords_c = graph.nodes[c_connect3_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_c)
        graph.add_edge(a_connect3_node_id, c_connect3_node_id, distance)
    else:
        st.warning("Level3 inter-building corridor connection nodes not found")

    return graph

# Dijkstra's algorithm
def dijkstra(graph, start_node):
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    nodes = set(graph.nodes.keys())

    while nodes:
        min_node = min(nodes, key=lambda node: distances[node])
        nodes.remove(min_node)

        if distances[min_node] == float('inf'):
            break

        for neighbor, weight in graph.nodes[min_node]['neighbors'].items():
            alternative_route = distances[min_node] + weight
            if alternative_route < distances[neighbor]:
                distances[neighbor] = alternative_route
                previous_nodes[neighbor] = min_node

    return distances, previous_nodes

# Generate shortest path
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path if len(path) > 1 else None

# Navigation function
def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    try:
        # Find nodes using multiple mappings
        start_key = (start_building, start_classroom, start_level)
        end_key = (end_building, end_classroom, end_level)
        
        start_node = graph.node_id_map.get(start_key)
        end_node = graph.node_id_map.get(end_key)
        
        # Alternative: Construct node ID
        if not start_node:
            start_node = f"{start_building}-classroom-{start_classroom}@{start_level}"
        if not end_node:
            end_node = f"{end_building}-classroom-{end_classroom}@{end_level}"

        # Verify if nodes exist
        if start_node not in graph.nodes:
            return None, f"Starting classroom does not exist: {start_building}{start_classroom}@{start_level}", None
        if end_node not in graph.nodes:
            return None, f"Destination classroom does not exist: {end_building}{end_classroom}@{end_level}", None

        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)

        if path:
            total_distance = distances[end_node]
            # Combine path details into one line
            simplified_path = []
            for node_id in path:
                node_type = graph.nodes[node_id]['type']
                node_name = graph.nodes[node_id]['name']
                node_level = graph.nodes[node_id]['level']
                node_building = graph.nodes[node_id]['building']
                if node_type in ['classroom', 'stair']:
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
            
            # Connect all steps with arrows to form a single line display
            full_path_str = " → ".join(simplified_path)
            return path, f"Total distance: {total_distance:.2f} units", full_path_str
        else:
            return None, "No available path between the two classrooms", None
    except Exception as e:
        return None, f"Navigation error: {str(e)}", None

# Plot path on 3D map
def plot_path(ax, graph, path):
    try:
        x = []
        y = []
        z = []
        labels = []

        for node_id in path:
            coords = graph.nodes[node_id]['coordinates']
            x.append(coords[0])
            y.append(coords[1])
            z.append(coords[2])
            
            node_type = graph.nodes[node_id]['type']
            if node_type == 'classroom':
                labels.append(graph.nodes[node_id]['name'])
            elif node_type == 'stair':
                labels.append(graph.nodes[node_id]['name'])
            else:
                labels.append("")

        # Enlarge path line width
        ax.plot(x, y, z, color=COLORS['path'], linewidth=6, linestyle='-', marker='o', markersize=10)
        # Enlarge start and end markers
        ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=1000, marker='*', label='Start', edgecolors='black')
        ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=1000, marker='*', label='End', edgecolors='black')
        ax.text(x[0], y[0], z[0], f"Start\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=16)
        ax.text(x[-1], y[-1], z[-1], f"End\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=16)

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    except Exception as e:
        st.error(f"Failed to plot path: {str(e)}")

# Get all building, floor and classroom information
def get_classroom_info(school_data):
    try:
        buildings = [b for b in school_data.keys() if b.startswith('building')]
        building_names = [b.replace('building', '') for b in buildings]
        
        classrooms_by_building = {}
        levels_by_building = {}
        
        for building_id in buildings:
            building_name = building_id.replace('building', '')
            building_data = school_data[building_id]
            
            levels = []
            classrooms_by_level = {}
            
            for level in building_data['levels']:
                level_name = level['name']
                levels.append(level_name)
                classrooms = [classroom['name'] for classroom in level['classrooms']]
                classrooms_by_level[level_name] = classrooms
            
            levels_by_building[building_name] = levels
            classrooms_by_building[building_name] = classrooms_by_level
            
        return building_names, levels_by_building, classrooms_by_building
    except Exception as e:
        st.error(f"Failed to get classroom information: {str(e)}")
        return [], {}, {}

# -------------------------- 3. Streamlit Interface Logic --------------------------
def main():
    # Adjust margins
    st.markdown("""
        <style>
            .block-container {
                padding-left: 1rem;    /* Reduce left margin */
                padding-right: 1rem;   /* Reduce right margin */
                max-width: 100%;       /* Remove maximum width limit */
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("🏫 SCIS Campus Navigation System")
    st.markdown("3D Map & Inter-building Path Planning")

    # Load JSON data
    try:
        # Note: Ensure the JSON file is in the same directory as the script or provide the correct path
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        nav_graph = build_navigation_graph(school_data)
        building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
        st.success("✅ Campus data loaded successfully!")
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return

    # Layout adjustment: 1/3 for left interactive interface, 2/3 for right map
    col1, col2 = st.columns([1, 5])

    with col1:
        st.markdown("#### 📍 Select Locations")
        
        # Start point selection
        st.markdown("#### Start Point")
        start_building = st.selectbox("Building", building_names, key="start_building")
        start_levels = levels_by_building.get(start_building, [])
        start_level = st.selectbox("Floor", start_levels, key="start_level")
        start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
        start_classroom = st.selectbox("Classroom", start_classrooms, key="start_classroom")

        # End point selection
        st.markdown("#### Destination")
        end_building = st.selectbox("Building", building_names, key="end_building")
        end_levels = levels_by_building.get(end_building, [])
        end_level = st.selectbox("Floor", end_levels, key="end_level")
        end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
        end_classroom = st.selectbox("Classroom", end_classrooms, key="end_classroom")

        # Navigation button
        nav_button = st.button("🔍 Find Shortest Path", use_container_width=True)

    with col2:
        st.markdown("#### 🗺️ 3D Campus Map")
        
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        if nav_button:
            try:
                path, message, simplified_path = navigate(
                    nav_graph, 
                    start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                
                if path:
                    st.success(f"📊 Navigation result: {message}")
                    # Path details in single line display
                    st.markdown("##### 🛤️ Path Details")
                    st.info(simplified_path)  # Use info box to highlight single line path
                    
                    fig, ax = plot_3d_map(school_data)
                    plot_path(ax, nav_graph, path)
                    st.session_state['fig'] = fig
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"Navigation process error: {str(e)}")
        
        try:
            st.pyplot(st.session_state['fig'])
        except Exception as e:
            st.error(f"Failed to display map: {str(e)}")

if __name__ == "__main__":
    main()
    



