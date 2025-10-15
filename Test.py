import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. Basic Configuration --------------------------
plt.switch_backend('Agg')  # Resolve Streamlit matplotlib rendering issues

# Define color constants: including Buildings A/B/C
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral'},  # Building fill colors
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},  # Floor border colors
    'corridor_line': {'A': 'cyan', 'B': 'forestgreen', 'C': 'salmon'},  # Corridor line colors
    'corridor_node': 'navy',
    'corridor_label': 'darkblue',
    # Staircase color configuration (including Building B stairs)
    'stair': {
        'Stairs1': '#FF5733',   # Building A - Orange-red
        'Stairs2': '#33FF57',   # Building C - Green
        'Stairs3': '#3357FF',   # Building C - Blue
        'Stairs4': '#FF33F5',   # Building C - Pink-purple
        'Stairs5': '#F5FF33',   # Building C - Yellow
        'StairsB1': '#33FFF5',  # Building B - Cyan
        'StairsB2': '#FF9933',  # Building B - Orange
    },
    'stair_label': 'darkred',  # Staircase label color
    'classroom_label': 'black',
    'path': 'darkred',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold',
    'building_label': {'A': 'darkblue', 'B': 'darkgreen', 'C': 'darkred'}  # Building label colors
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

# Plot 3D map - All buildings shown during route planning
def plot_3d_map(school_data, display_options=None):
    # Enlarge figure size
    fig = plt.figure(figsize=(35, 30))
    ax = fig.add_subplot(111, projection='3d')

    # Enlarge axis tick labels
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # Show all content by default (including Building B)
    if display_options is None:
        display_options = {
            'start_level': None,
            'end_level': None,
            'path_stairs': set(),
            'show_all': True,
            'path': [],
            'start_building': None,  # Starting building information
            'end_building': None     # Destination building information
        }
    
    show_all = display_options['show_all']
    start_level = display_options['start_level']
    end_level = display_options['end_level']
    path_stairs = display_options['path_stairs']
    path = display_options.get('path', [])
    start_building = display_options.get('start_building')
    end_building = display_options.get('end_building')

    # Store building label position information
    building_label_positions = {}

    # Iterate through all buildings (A/B/C)
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        # Keep B building visible during route planning (remove hide logic)
        building_data = school_data[building_id]
        
        # Record highest displayed floor and maximum Y value for the building
        displayed_levels = []  # Stores displayed floors
        max_displayed_z = -float('inf')
        max_displayed_y = -float('inf')
        corresponding_x = 0  # X coordinate corresponding to maximum Y value
        level_count = 0
        
        # Process each floor of the building
        for level in building_data['levels']:
            level_name = level['name']
            z = level['z']
            
            # Determine if current floor should be displayed
            show_level = show_all
            if not show_all:
                # Show start/end floors + B building floors connected to path
                if building_name == 'B':
                    # Show B building floors that have path stairs or connect to A/C
                    show_level = any((building_name, s_name, level_name) in path_stairs for s_name in ['StairsB1', 'StairsB2'])
                    # Also show B's level1 (connecting floor) if path involves A/B or C/B
                    if (start_building == 'B' or end_building == 'B') or (start_building in ['A','C'] and end_building in ['A','C'] and 'B' in [start_building, end_building]):
                        show_level = show_level or (level_name == 'level1')
                else:
                    # Show start and end floors for A/C
                    show_level = (level_name == start_level) or (level_name == end_level)
            
            # If floor will be displayed, record relevant information
            if show_level:
                displayed_levels.append(level)
                if z > max_displayed_z:
                    max_displayed_z = z
                
                # Get floor plane information
                fp = level['floorPlane']
                current_max_y = fp['maxY']
                if current_max_y > max_displayed_y:
                    max_displayed_y = current_max_y
                    corresponding_x = (fp['minX'] + fp['maxX']) / 2
            
            level_count += 1
            
            # Adapt floor colors for each building
            floor_border_color = COLORS['floor_z'].get(z, 'gray')
            building_fill_color = COLORS['building'].get(building_name, 'lightgray')

            # Draw floor plane (only for floors that need to be displayed)
            if show_level:
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
                
                # Avoid duplicate legend entries
                legend_label = f"Building {building_name}-{level_name}"
                if legend_label not in ax.get_legend_handles_labels()[1]:
                    ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=4, label=legend_label)
                else:
                    ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=4)
                ax.plot_trisurf(x_plane[:-1], y_plane[:-1], z_plane[:-1], 
                                color=building_fill_color, alpha=0.3)

                # Draw corridors (added external corridor judgment, reading JSON styles)
                for corr_idx, corridor in enumerate(level['corridors']):
                    points = corridor['points']
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    z_coords = [p[2] for p in points]
                    
                    # 1. First check if it's an external corridor (read type and style from JSON)
                    is_external = corridor.get('type') == 'external'
                    if is_external:
                        # Read style from JSON, use default gray dashed line if not configured
                        ext_style = corridor.get('style', {})
                        corr_line_color = ext_style.get('color', 'gray')  # Prefer JSON color, default to gray
                        corr_line_style = ext_style.get('lineType', '--')  # Prefer JSON line type, default to dashed
                        corr_line_width = 10  # Enlarge external corridor line width for emphasis
                        corr_label = f"External Corridor ({building_name}-{corridor.get('name', f'corr{corr_idx}')})"
                    
                    # 2. Non-external corridor: check if it's a connecting corridor
                    elif 'name' in corridor and ('connectToBuilding' in corridor['name']):
                        corr_line_color = COLORS['connect_corridor']
                        corr_line_style = '-'  # Use solid line for connecting corridors
                        corr_line_width = 12  # Enlarge inter-building corridor line width
                        corr_label = f"Connecting Corridor ({building_name}-{level_name})"
                    
                    # 3. Regular internal corridor
                    else:
                        corr_line_color = COLORS['corridor_line'].get(building_name, 'gray')
                        corr_line_style = '-'  # Use solid line for internal corridors
                        corr_line_width = 8  # Enlarge regular corridor line width
                        corr_label = None
                    
                    # Draw corridor (add linestyle parameter for dashed/solid lines)
                    if corr_label and corr_label not in ax.get_legend_handles_labels()[1]:
                        ax.plot(x, y, z_coords, 
                                color=corr_line_color, 
                                linestyle=corr_line_style,  # Apply line style (dashed/solid)
                                linewidth=corr_line_width, 
                                alpha=0.8, 
                                label=corr_label)
                    else:
                        ax.plot(x, y, z_coords, 
                                color=corr_line_color, 
                                linestyle=corr_line_style,  # Apply line style (dashed/solid)
                                linewidth=corr_line_width, 
                                alpha=0.8)
                    
                    # Corridor nodes
                    for px, py, pz in points:
                        ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=40, marker='s', alpha=0.9)

                # Draw classrooms
                for classroom in level['classrooms']:
                    x, y, _ = classroom['coordinates']
                    width, depth = classroom['size']
                    class_name = classroom['name']

                    ax.text(x, y, z, class_name, color=COLORS['classroom_label'], fontweight='bold', fontsize=14)
                    ax.scatter(x, y, z, color=building_fill_color, s=160, edgecolors=floor_border_color)
                    # Classroom border
                    ax.plot([x, x + width, x + width, x, x],
                            [y, y, y + depth, y + depth, y],
                            [z, z, z, z, z],
                            color=floor_border_color, linestyle='--', alpha=0.6, linewidth=2)

            # Draw staircases (include B building stairs)
            for stair in level['stairs']:
                stair_name = stair['name']
                # Check if it's a staircase on the path
                is_path_stair = (building_name, stair_name, level_name) in path_stairs
                
                if show_all or show_level or is_path_stair:
                    x, y, _ = stair['coordinates']
                    stair_label = f"Building {building_name}-{stair_name}"
                    
                    # Adapt staircase colors for each building
                    stair_color = COLORS['stair'].get(stair_name, 'red')
                    
                    # Use more prominent style for staircases on the path
                    marker_size = 800 if is_path_stair else 600
                    marker_edge_width = 3 if is_path_stair else 1
                    
                    # Avoid duplicate legend entries
                    if stair_label not in ax.get_legend_handles_labels()[1]:
                        ax.scatter(x, y, z, color=stair_color, s=marker_size, marker='^', 
                                  label=stair_label, edgecolors='black', linewidths=marker_edge_width)
                    else:
                        ax.scatter(x, y, z, color=stair_color, s=marker_size, marker='^',
                                  edgecolors='black', linewidths=marker_edge_width)
                    
                    # Staircase label
                    ax.text(x, y, z, stair_name, color=COLORS['stair_label'], fontweight='bold', fontsize=14)
        
        # Store building label positions (include B building)
        if level_count > 0 and len(displayed_levels) > 0:
            # For all buildings, place label appropriately
            if building_name == 'B':
                label_y = max_displayed_y - 2.0  # B building label placed outside Y (more negative)
            else:
                label_y = max_displayed_y + 2.0  # A/C building labels placed outside Y
            label_z = max_displayed_z + 1.0
            center_x = corresponding_x
            
            building_label_positions[building_name] = (center_x, label_y, label_z)

    # Add building labels (include B building)
    for building_name, (x, y, z) in building_label_positions.items():
        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", 
                        facecolor=COLORS['building'].get(building_name, 'lightgray'), alpha=0.7)
        ax.text(
            x, y, z, 
            f"Building {building_name}", 
            color=COLORS['building_label'].get(building_name, 'black'), 
            fontweight='bold', 
            fontsize=30,
            ha='center', 
            va='center', 
            bbox=bbox_props
        )

    # Draw path (when there is a path and not showing all)
    if path and not show_all:
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

            # Path line
            ax.plot(x, y, z, color=COLORS['path'], linewidth=6, linestyle='-', marker='o', markersize=10, label='Navigation Path')
            # Start marker
            ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=1000, marker='*', label='Start', edgecolors='black')
            # End marker
            ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=1000, marker='*', label='End', edgecolors='black')
            # Start label
            ax.text(x[0], y[0], z[0], f"Start\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=16)
            # End label
            ax.text(x[-1], y[-1], z[-1], f"End\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=16)
        except Exception as e:
            st.warning(f"Path drawing warning: {str(e)}")

    # Axis labels and title
    ax.set_xlabel('X Coordinate', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=18, fontweight='bold')
    ax.set_zlabel('Floor Height (Z Value)', fontsize=18, fontweight='bold')
    ax.set_title('Campus 3D Navigation Map (A/B/C Building Navigation)', fontsize=24, fontweight='bold', pad=20)
    
    # Legend
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
        
        # Add B building nodes (remove B building skip logic)
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
        # Add additional mapping for classrooms
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
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# Build navigation graph (include B building)
def build_navigation_graph(school_data):
    graph = Graph()

    # Step 1: Add all building nodes (A/B/C - classrooms, stairs, corridors)
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        # Process all buildings (A/B/C, remove B skip logic)
        building_data = school_data[building_id]
        
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

            # 2. Add staircase nodes (include B building stairs)
            for stair in level['stairs']:
                graph.add_node(
                    building_id=building_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

            # 3. Add corridor nodes
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

    # Step 2: Add all connections (A/B/C)
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        # Process all buildings (remove B skip logic)
        building_data = school_data[building_id]

        for level in building_data['levels']:
            level_name = level['name']
            
            # Get all corridor nodes for current building and floor
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
                    
                    if distance < 3.0:  # Corridor nodes with close distance are considered connected
                        graph.add_edge(node1_id, node2_id, distance)

            # 3. Connect classrooms to nearest corridor node
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
                    st.warning(f"Warning: Classroom {graph.nodes[class_node_id]['name']} in Building {building_name}{level_name} has no corridor connection")

            # 4. Connect staircases to nearest corridor node
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

        # 5. Connect nodes across different floors in the same building (include B)
        for connection in building_data['connections']:
            from_obj_name, from_level = connection['from']
            to_obj_name, to_level = connection['to']
            
            # Process all connections (remove B filter)
            from_obj_type = 'stair' if from_obj_name.startswith('Stairs') else 'corridor'
            to_obj_type = 'stair' if to_obj_name.startswith('Stairs') else 'corridor'
            
            if from_obj_type == 'corridor':
                from_obj_name = f"{from_obj_name}-p0"
            if to_obj_type == 'corridor':
                to_obj_name = f"{to_obj_name}-p0"
            
            from_node_id = graph.node_id_map.get((building_id, from_obj_type, from_obj_name, from_level))
            to_node_id = graph.node_id_map.get((building_id, to_obj_type, to_obj_name, to_level))
            
            if from_node_id and to_node_id:
                graph.add_edge(from_node_id, to_node_id, 5.0)  # Staircase connection weight fixed at 5

    # 6. Connect nodes between buildings (A-B, B-C, A-C)
    # A-B connection (level1)
    a_building_id = 'buildingA'
    b_building_id = 'buildingB'
    c_building_id = 'buildingC'
    
    # A-B level1 connection
    ab_connect_level = 'level1'
    a_b_corr_name = 'connectToBuildingB-p1'  # A's external corridor to B
    a_b_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_b_corr_name, ab_connect_level))
    b_a_corr_name = 'connectToBuildingAAndC-p1'  # B's external corridor to A
    b_a_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_a_corr_name, ab_connect_level))
    
    if a_b_node_id and b_a_node_id:
        coords_a = graph.nodes[a_b_node_id]['coordinates']
        coords_b = graph.nodes[b_a_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_b)
        graph.add_edge(a_b_node_id, b_a_node_id, distance)
    else:
        st.warning("Could not find A-B level1 inter-building corridor connection nodes")
    
    # B-C level1 connection
    bc_connect_level = 'level1'
    b_c_corr_name = 'connectToBuildingAAndC-p0'  # B's external corridor to C
    b_c_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_c_corr_name, bc_connect_level))
    c_b_corr_name = 'connectToBuildingB-p1'  # C's external corridor to B
    c_b_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_b_corr_name, bc_connect_level))
    
    if b_c_node_id and c_b_node_id:
        coords_b = graph.nodes[b_c_node_id]['coordinates']
        coords_c = graph.nodes[c_b_node_id]['coordinates']
        distance = euclidean_distance(coords_b, coords_c)
        graph.add_edge(b_c_node_id, c_b_node_id, distance)
    else:
        st.warning("Could not find B-C level1 inter-building corridor connection nodes")
    
    # Original A-C connections (keep existing)
    # Level 1 inter-building connection
    connect_level1 = 'level1'
    a_corr1_name = 'connectToBuildingC-p3'
    a_connect1_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr1_name, connect_level1))
    c_corr1_name = 'connectToBuildingA-p0'
    c_connect1_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr1_name, connect_level1))
    
    if a_connect1_node_id and c_connect1_node_id:
        coords_a = graph.nodes[a_connect1_node_id]['coordinates']
        coords_c = graph.nodes[c_connect1_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_c)
        graph.add_edge(a_connect1_node_id, c_connect1_node_id, distance)
    else:
        st.warning("Could not find level 1 A-C inter-building corridor connection nodes")
    
    # Level 3 inter-building connection
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
        st.warning("Could not find level 3 A-C inter-building corridor connection nodes")

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

# Navigation function (support A/B/C buildings)
def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    # Validate buildings (A/B/C allowed)
    valid_buildings = ['A', 'B', 'C']
    if start_building not in valid_buildings or end_building not in valid_buildings:
        return None, "Invalid building selection, only Buildings A, B and C are supported", None, None
        
    try:
        # Use multiple mappings to find nodes
        start_key = (start_building, start_classroom, start_level)
        end_key = (end_building, end_classroom, end_level)
        
        start_node = graph.node_id_map.get(start_key)
        end_node = graph.node_id_map.get(end_key)
        
        # Alternative: construct node IDs
        if not start_node:
            start_node = f"{start_building}-classroom-{start_classroom}@{start_level}"
        if not end_node:
            end_node = f"{end_building}-classroom-{end_classroom}@{end_level}"

        # Verify nodes exist
        if start_node not in graph.nodes:
            return None, f"Starting classroom does not exist: {start_building}{start_classroom}@{start_level}", None, None
        if end_node not in graph.nodes:
            return None, f"Destination classroom does not exist: {end_building}{end_classroom}@{end_level}", None, None

        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)

        if path:
            total_distance = distances[end_node]
            simplified_path = []
            # Collect staircases on the path
            path_stairs = set()
            # Track previous node's building
            prev_building = None
            
            for node_id in path:
                node_type = graph.nodes[node_id]['type']
                node_name = graph.nodes[node_id]['name']
                node_level = graph.nodes[node_id]['level']
                node_building = graph.nodes[node_id]['building']
                
                # Record staircases on the path
                if node_type == 'stair':
                    path_stairs.add((node_building, node_name, node_level))
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
                
                # Process classroom nodes
                elif node_type == 'classroom':
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
                
                # Process corridor nodes, detect if it's a connecting corridor
                elif node_type == 'corridor':
                    # Check if it's a corridor connecting two buildings
                    if 'connectToBuilding' in node_name:
                        # Determine which buildings the corridor connects
                        if 'connectToBuildingA' in node_name:
                            connected_building = 'A'
                        elif 'connectToBuildingB' in node_name:
                            connected_building = 'B'
                        elif 'connectToBuildingC' in node_name:
                            connected_building = 'C'
                        else:
                            connected_building = 'Other'
                            
                        # Only add corridor information when building changes
                        if prev_building and prev_building != node_building:
                            simplified_path.append(f"Cross corridor from Building {prev_building} to Building {node_building}({node_level})")
                
                # Update previous node's building
                if node_type in ['classroom', 'stair', 'corridor']:
                    prev_building = node_building
            
            full_path_str = " → ".join(simplified_path)
            # Return display options (keep B building visible)
            display_options = {
                'start_level': start_level,
                'end_level': end_level,
                'path_stairs': path_stairs,
                'show_all': False,
                'path': path,
                'start_building': start_building,
                'end_building': end_building
            }
            return path, f"Total distance: {total_distance:.2f} units", full_path_str, display_options
        else:
            return None, "No available path between the two classrooms", None, None
    except Exception as e:
        return None, f"Navigation error: {str(e)}", None, None

# Draw path on 3D map
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

        # Path line
        ax.plot(x, y, z, color=COLORS['path'], linewidth=6, linestyle='-', marker='o', markersize=10)
       

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    except Exception as e:
        st.error(f"Failed to draw path: {str(e)}")

# Get all building, floor and classroom information (include B building)
def get_classroom_info(school_data):
    try:
        # Show all buildings (A/B/C)
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
        st.error(f"Failed to retrieve classroom information: {str(e)}")
        return [], {}, {}

# Reset app state to initial state (show all buildings including B)
def reset_app_state():
    st.session_state['display_options'] = {
        'start_level': None,
        'end_level': None,
        'path_stairs': set(),
        'show_all': True,  # Show all buildings including B after reset
        'path': [],
        'start_building': None,
        'end_building': None
    }
    st.session_state['current_path'] = None
    # Clear path result display
    if 'path_result' in st.session_state:
        del st.session_state['path_result']

# -------------------------- 3. Streamlit Interface Logic --------------------------
def main():
    # Adjust margins
    st.markdown("""
        <style>
        .stApp {
                padding-top: 0.0rem !important; 
            }
            body {
                position: relative;
                min-height: 100vh;
                margin: 0;
                padding: 0;
            }
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 100%;
                padding-bottom: 80px; 
            }
         
            .author-tag {
                position: fixed; 
                bottom: 50px;  
                right: 60px;    
                font-size: 16px;
                font-weight: bold;
                color: #666;    
                background: transparent;;
                padding: 6px 12px;
                border: none;
                border-radius: 0;
                z-index: 9999;  
                
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="author-tag">Created By DANIEL </div>', unsafe_allow_html=True)
    st.subheader("🏫SCIS Campus Navigation System")
    st.markdown("3D Map & Inter-building Path Planning (A/B/C Building Navigation)")

    # Initialize session state variables
    if 'display_options' not in st.session_state:
        st.session_state['display_options'] = {
            'start_level': None,
            'end_level': None,
            'path_stairs': set(),
            'show_all': True,  # Show all buildings including B in initial state
            'path': [],
            'start_building': None,
            'end_building': None
        }
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = None

    # Load JSON data
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        # Global variable graph for plot_3d_map use
        global graph
        graph = build_navigation_graph(school_data)
        building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
        st.success("✅ Campus data loaded successfully! Initial state shows A/B/C buildings")
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return

    # Layout adjustment: left 1/3 for interface, right 2/3 for map
    col1, col2 = st.columns([1, 5])

    with col1:
        st.markdown("#### 📍 Select Locations")
        
        # Start point selection (include B building)
        st.markdown("#### Start Point")
        start_building = st.selectbox("Building", building_names, key="start_building")
        start_levels = levels_by_building.get(start_building, [])
        start_level = st.selectbox("Floor", start_levels, key="start_level")
        start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
        start_classroom = st.selectbox("Classroom", start_classrooms, key="start_classroom")

        # End point selection (include B building)
        st.markdown("#### End Point")
        end_building = st.selectbox("Building", building_names, key="end_building")
        end_levels = levels_by_building.get(end_building, [])
        end_level = st.selectbox("Floor", end_levels, key="end_level")
        end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
        end_classroom = st.selectbox("Classroom", end_classrooms, key="end_classroom")

        # Navigation button and reset button
        nav_button = st.button("🔍 Find Shortest Path", use_container_width=True)
        
        # Add reset view button (shows all buildings including B after reset)
        reset_button = st.button(
            "🔄 Reset View", 
            use_container_width=True,
            help="Click to return to initial state, showing all floors (including Building B) and clearing path"
        )
        
        # Handle reset button click
        if reset_button:
            reset_app_state()
            st.rerun()  # Rerun app to refresh interface

    with col2:
        st.markdown("#### 🗺️ 3D Campus Map")
        
        # Handle navigation button click (keep B building visible)
        if nav_button:
            try:
                path, message, simplified_path, display_options = navigate(
                    graph, 
                    start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                
                if path and display_options:
                    st.success(f"📊 Navigation result: {message}")
                    st.markdown("##### 🛤️ Path Details")
                    st.info(simplified_path)
                    
                    # Save path and display options to session state (keep B visible)
                    st.session_state['current_path'] = path
                    st.session_state['display_options'] = display_options
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"Navigation process error: {str(e)}")
        
        # Draw map (include B building)
        try:
            # If there's a route planning result, use saved display options (keep B visible)
            if st.session_state['current_path'] is not None:
                fig, ax = plot_3d_map(school_data, st.session_state['display_options'])
                # Draw path
                plot_path(ax, graph, st.session_state['current_path'])
            else:
                # Show all floors in initial state and after reset (including B)
                fig, ax = plot_3d_map(school_data)
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to display map: {str(e)}")

if __name__ == "__main__":
    main()
















