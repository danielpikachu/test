import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

st.set_page_config(
    page_title="SCIS Navigation System",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* 核心：让整个页面内容上移 */
.main .block-container {
    padding-top: 25px !important;
    padding-left: 15px !important;
}
</style>
""", unsafe_allow_html=True)

plt.switch_backend('Agg')

# --------------------------
# Google Sheets Configuration
# --------------------------
SHEET_NAME = 'Navigation visitors'
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

def get_credentials():
    try:
        service_account_info = st.secrets["google_service_account"]
        return Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPE
        )
    except KeyError:
        st.error("google_service_account not found in Streamlit Secrets, please check TOML format")
        return None
    except Exception as e:
        st.error(f"Failed to load credentials: {str(e)}")
        return None

def init_google_sheet():
    try:
        creds = get_credentials()
        if not creds:
            return None
        client = gspread.authorize(creds)
        
        try:
            sheet = client.open(SHEET_NAME)
        except gspread.exceptions.SpreadsheetNotFound:
            sheet = client.create(SHEET_NAME)
        
        try:
            stats_worksheet = sheet.worksheet("Access_Stats")
        except gspread.exceptions.WorksheetNotFound:
            stats_worksheet = sheet.add_worksheet(title="Access_Stats", rows="1000", cols=3)
            stats_worksheet.append_row(["Timestamp", "Access_Count", "Total_Accesses"])
            stats_worksheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, 1])
        
        return stats_worksheet
    except Exception as e:
        return None

def update_access_count(worksheet):
    if not worksheet:
        return 0
        
    try:
        records = worksheet.get_all_values()
        if len(records) < 2:
            return 0
            
        last_row = records[-1]
        total = int(last_row[2]) if last_row[2].isdigit() else 0
        new_total = total + 1
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([current_time, 1, new_total])
        
        return new_total
    except Exception as e:
        return 0

def get_total_accesses(worksheet):
    if not worksheet:
        return 0
        
    try:
        records = worksheet.get_all_values()
        if len(records) < 2:
            return 0
            
        last_row = records[-1]
        return int(last_row[2]) if last_row[2].isdigit() else 0
    except Exception as e:
        return 0

# --------------------------
# Color Scheme
# --------------------------
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral', 'Gate': 'gold'},
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 7: 'orange', 12: 'purple', 17: 'teal'},
    'corridor_line': {'A': 'cyan', 'B': 'forestgreen', 'C': 'salmon', 'Gate': 'darkgoldenrod'},
    'corridor_node': 'navy',
    'corridor_label': 'darkblue',
    'stair': {
        'Stairs1': '#FF5733',
        'Stairs2': '#33FF57',
        'Stairs3': '#3357FF',
        'Stairs4': '#FF33F5',
        'Stairs5': '#F5FF33',
        'StairsB1': '#33FFF5',
        'StairsB2': '#FF9933',
        'GateStairs': '#FFD700'
    },
    'stair_label': 'darkred',
    'classroom_label': 'black',
    'path': 'red',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold',
    'building_label': {'A': 'darkblue', 'B': 'darkgreen', 'C': 'darkred', 'Gate': 'darkgoldenrod'}
}

def load_school_data_detailed(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data file: {str(e)}")
        return None

# ====================== 3D Plot Function ======================
def plot_3d_map_plotly(school_data, graph=None, display_options=None):
    fig = go.Figure()

    if display_options is None:
        display_options = {
            'start_level': None,
            'end_level': None,
            'path_stairs': set(),
            'show_all': True,
            'path': [],
            'start_building': None,
            'end_building': None
        }
    
    show_all = display_options['show_all']
    start_level = display_options['start_level']
    end_level = display_options['end_level']
    path_stairs = display_options['path_stairs']
    path = display_options.get('path', [])
    start_building = display_options.get('start_building')
    end_building = display_options.get('end_building')

    building_label_positions = {}
    shown_stairs_legends = set()

    for building_id in school_data.keys():
        if building_id == 'gate':
            building_name = 'Gate'
        elif building_id.startswith('building'):
            building_name = building_id.replace('building', '')
        else:
            continue
            
        building_data = school_data[building_id]
        
        displayed_levels = []
        max_displayed_z = -float('inf')
        max_displayed_y = -float('inf')
        corresponding_x = 0
        level_count = 0
        
        for level in building_data['levels']:
            level_name = level['name']
            raw_z = level['z']
            z = raw_z + 10
            
            show_level = show_all
            if not show_all:
                if building_name == 'B':
                    show_level = (level_name == 'level1') or any((building_name, s, level_name) in path_stairs for s in ['StairsB1','StairsB2'])
                elif building_name == 'Gate':
                    show_level = True
                else:
                    show_level = (level_name == start_level) or (level_name == end_level)
            
            if show_level:
                displayed_levels.append(level)
                if z > max_displayed_z:
                    max_displayed_z = z
                
                fp = level['floorPlane']
                current_max_y = fp['maxY']
                if current_max_y > max_displayed_y:
                    max_displayed_y = current_max_y
                    corresponding_x = (fp['minX'] + fp['maxX']) / 2
            
            level_count += 1
            floor_border_color = COLORS['floor_z'].get(raw_z, 'gray')
            building_fill_color = COLORS['building'].get(building_name, 'lightgray')

            if show_level:
                fp = level['floorPlane']
                x_vals = [fp['minX'], fp['maxX'], fp['maxX'], fp['minX'], fp['minX']]
                y_vals = [fp['minY'], fp['minY'], fp['maxY'], fp['maxY'], fp['minY']]
                z_vals = [z] * 5

                fig.add_trace(go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals,
                    mode='lines',
                    line=dict(color=floor_border_color, width=4),
                    name=f"Building {building_name}-{level_name}",
                    legendgroup=f"Building {building_name}",
                    showlegend=True
                ))

                fig.add_trace(go.Mesh3d(
                    x=x_vals[:4], y=y_vals[:4], z=z_vals[:4],
                    color=building_fill_color, opacity=0.3, showlegend=False
                ))

                for corridor in level['corridors']:
                    points = corridor['points']
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    z_coords = [p[2]+10 for p in points]
                    
                    is_external = corridor.get('type') == 'external'
                    is_connect = 'connectToBuilding' in corridor.get('name','') or 'gateTo' in corridor.get('name','')
                    
                    if is_external:
                        corr_line_color = 'gray'
                        corr_line_width = 5
                        dash = 'dash'
                    elif is_connect:
                        corr_line_color = COLORS['connect_corridor']
                        corr_line_width = 7
                        dash = 'solid'
                    else:
                        corr_line_color = COLORS['corridor_line'].get(building_name, 'gray')
                        corr_line_width = 5
                        dash = 'solid'

                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z_coords,
                        mode='lines',
                        line=dict(color=corr_line_color, width=corr_line_width, dash=dash),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z_coords,
                        mode='markers',
                        marker=dict(color=COLORS['corridor_node'], size=3, symbol='square'),
                        showlegend=False
                    ))

                for classroom in level['classrooms']:
                    x, y, _ = classroom['coordinates']
                    w, d = classroom['size']
                    name = classroom['name']

                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z],
                        mode='markers+text',
                        marker=dict(color=building_fill_color, size=7, line=dict(color=floor_border_color, width=1)),
                        text=name, textposition="top center", textfont=dict(size=9, color='black'),
                        showlegend=False
                    ))

                    cx = [x, x+w, x+w, x, x]
                    cy = [y, y, y+d, y+d, y]
                    cz = [z]*5
                    fig.add_trace(go.Scatter3d(
                        x=cx, y=cy, z=cz,
                        mode='lines', line=dict(color=floor_border_color, width=1, dash='dash'),
                        opacity=0.6, showlegend=False
                    ))

            for stair in level['stairs']:
                s_name = stair['name']
                is_path = (building_name, s_name, level_name) in path_stairs
                
                if show_all or show_level or is_path:
                    x, y, _ = stair['coordinates']
                    color = COLORS['stair'].get(s_name, 'red')
                    size = 12 if is_path else 9
                    
                    legend_name = f"{building_name}-{s_name}"
                    show_legend = legend_name not in shown_stairs_legends
                    if show_legend:
                        shown_stairs_legends.add(legend_name)
                    
                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z],
                        mode='markers+text',
                        marker=dict(color=color, size=size, symbol='diamond', line=dict(color='black', width=2)),
                        text=s_name, textposition="top center", textfont=dict(size=9, color='darkred'),
                        name=legend_name,
                        legendgroup="Stairs",
                        showlegend=show_legend
                    ))
        
        if displayed_levels:
            label_z = max_displayed_z + 1.5
            label_y = max_displayed_y + (3 if building_name != 'B' else -2)
            building_label_positions[building_name] = (corresponding_x, label_y, label_z)

    for bld, (x, y, z) in building_label_positions.items():
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='text',
            text=f"Building {bld}",
            textfont=dict(size=14, color=COLORS['building_label'][bld], family='Arial bold'),
            showlegend=False
        ))

    if path and graph and not show_all:
        try:
            xs, ys, zs = [], [], []
            labels = []
            for nid in path:
                c = graph.nodes[nid]['coordinates']
                xs.append(c[0])
                ys.append(c[1])
                zs.append(c[2]+10)
                labels.append(graph.nodes[nid]['name'])

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines+markers',
                line=dict(color=COLORS['path'], width=5),
                marker=dict(color=COLORS['path'], size=4),
                name="Path"
            ))
            fig.add_trace(go.Scatter3d(
                x=[xs[0]], y=[ys[0]], z=[zs[0]],
                mode='markers+text', marker=dict(color=COLORS['start_marker'], size=14, symbol='square', line=dict(width=2)),
                text=f"Start\n{labels[0]}", textposition="top center", textfont=dict(size=11, color='green'),
                name="Start"
            ))
            fig.add_trace(go.Scatter3d(
                x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
                mode='markers+text', marker=dict(color=COLORS['end_marker'], size=14, symbol='square', line=dict(width=2)),
                text=f"End\n{labels[-1]}", textposition="top center", textfont=dict(size=11, color='purple'),
                name="End"
            ))
        except Exception:
            pass

    fig.update_layout(
        title=dict(text="Campus 3D Navigation Map", font=dict(size=22,color="gray"), x=0.5, xanchor='center'),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Floor (Z+10)",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.8)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=880
    )

    return fig

def plot_3d_map(school_data, graph=None, display_options=None):
    fig = plot_3d_map_plotly(school_data, graph, display_options)
    return fig, None

# --------------------------
# Graph & Pathfinding
# --------------------------
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}

    def add_node(self, building_id, node_type, name, level, coordinates):
        if building_id == 'gate':
            building_name = 'Gate'
        else:
            building_name = building_id.replace('building', '')
        
        if node_type == 'corridor':
            node_id = f"{building_name}-corr-{name}@{level}"
        else:
            node_id = f"{building_name}-{node_type}-{name}@{level}"
        
        self.nodes[node_id] = {
            'building': building_name,
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        
        map_key = (building_id, node_type, name, level)
        self.node_id_map[map_key] = node_id
        if node_type == 'classroom':
            class_key = (building_name, name, level)
            self.node_id_map[class_key] = node_id
            
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

def euclidean_distance(coords1, coords2, floor_penalty=15.0):
    base_dist = np.sqrt(sum((a - b)**2 for a, b in zip(coords1, coords2)))
    z1, z2 = coords1[2], coords2[2]
    floor_diff = abs(z1 - z2)
    penalty = floor_diff * floor_penalty
    total_dist = base_dist + penalty
    return total_dist

# ====================== 方向函数：全部改为深黄色高亮 ======================
def get_direction_between_nodes(graph, current_node_id, next_node_id):
    current_node = graph.nodes[current_node_id]
    next_node = graph.nodes[next_node_id]
    
    curr_x, curr_y, curr_z = current_node['coordinates']
    next_x, next_y, next_z = next_node['coordinates']
    
    curr_is_stair = current_node['type'] == 'stair'
    next_is_stair = next_node['type'] == 'stair'
    
    if curr_is_stair and next_is_stair:
        if next_z > curr_z:
            return "<span style='color:DarkGoldenRod; font-weight:bold;'>up</span>"
        elif next_z < curr_z:
            return "<span style='color:DarkGoldenRod; font-weight:bold;'>down</span>"
        else:
            return ""
    
    x_diff = next_x - curr_x
    y_diff = next_y - curr_y
    threshold = 0.1
    
    if abs(x_diff) > threshold or abs(y_diff) > threshold:
        if y_diff > threshold:
            return "<span style='color:DarkGoldenRod; font-weight:bold;'>forward</span>"
        elif y_diff < -threshold:
            return "<span style='color:DarkGoldenRod; font-weight:bold;'>backward</span>"
        elif x_diff > threshold:
            return "<span style='color:DarkGoldenRod; font-weight:bold;'>right</span>"
        elif x_diff < -threshold:
            return "<span style='color:DarkGoldenRod; font-weight:bold;'>left</span>"
    
    return ""

def build_navigation_graph(school_data):
    graph = Graph()

    for building_id in school_data.keys():
        if not (building_id.startswith('building') or building_id == 'gate'):
            continue
            
        building_data = school_data[building_id]
        
        for level in building_data['levels']:
            level_name = level['name']

            for classroom in level['classrooms']:
                class_name = classroom['name']
                graph.add_node(
                    building_id=building_id,
                    node_type='classroom',
                    name=class_name,
                    level=level_name,
                    coordinates=classroom['coordinates']
                )

            for stair in level['stairs']:
                graph.add_node(
                    building_id=building_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

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

    for building_id in school_data.keys():
        if not (building_id.startswith('building') or building_id == 'gate'):
            continue
            
        if building_id == 'gate':
            building_name = 'Gate'
        else:
            building_name = building_id.replace('building', '')
        
        building_data = school_data[building_id]

        for level in building_data['levels']:
            level_name = level['name']
            
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]

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
                        distance = euclidean_distance(coords1, coords2, floor_penalty=0)
                        graph.add_edge(current_node_id, next_node_id, distance)

            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2, floor_penalty=0)
                    
                    if distance < 3.0:
                        graph.add_edge(node1_id, node2_id, distance)

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
                    dist = euclidean_distance(class_coords, corr_coords, floor_penalty=0)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(class_node_id, nearest_corr_node_id, min_dist)

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
                    dist = euclidean_distance(stair_coords, corr_coords, floor_penalty=0)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(stair_node_id, nearest_corr_node_id, min_dist)

        stair_names = set()
        for node_id, node_info in graph.nodes.items():
            if node_info['type'] == 'stair':
                stair_names.add((node_info['building'], node_info['name']))

        for (building, stair_name) in stair_names:
            stair_level_nodes = []
            for node_id, node_info in graph.nodes.items():
                if (node_info['building'] == building and 
                    node_info['type'] == 'stair' and 
                    node_info['name'] == stair_name):
                    stair_level_nodes.append((node_id, node_info['coordinates'], node_info['level']))
            
            stair_level_nodes.sort(key=lambda x: x[1][2])
            for i in range(len(stair_level_nodes)-1):
                node1_id, coords1, _ = stair_level_nodes[i]
                node2_id, coords2, _ = stair_level_nodes[i+1]
                
                dist = euclidean_distance(coords1, coords2, floor_penalty=15.0)
                graph.add_edge(node1_id, node2_id, dist)

        for connection in building_data['connections']:
            from_obj_name, from_level = connection['from']
            to_obj_name, to_level = connection['to']
            
            if from_obj_name.startswith(('Stairs', 'GateStairs')):
                from_obj_type = 'stair'
            elif any(
                from_obj_name == cls['name'] 
                for level in building_data['levels'] 
                for cls in level.get('classrooms', [])
            ):
                from_obj_type = 'classroom'
            else:
                from_obj_type = 'corridor'
            
            from_node_name = f"{from_obj_name}-p0" if from_obj_type == 'corridor' else from_obj_name
            from_node_id = graph.node_id_map.get((building_id, from_obj_type, from_node_name, from_level))

            to_building_id = building_id
            target_building_map = {
                'ENTRANCE': 'buildingA',
                'connectToBuildingAAndC': 'buildingB',
                'SCHOOL CLINIC': 'buildingC',
                'connectToBuildingB': 'buildingB',
                'connectToBuildingC': 'buildingC'
            }
            for keyword, target_building in target_building_map.items():
                if keyword in to_obj_name:
                    to_building_id = target_building
                    break
            
            if to_obj_name.startswith(('Stairs', 'GateStairs')):
                to_obj_type = 'stair'
            elif to_building_id in school_data:
                to_obj_type = 'classroom' if any(
                    to_obj_name == cls['name']
                    for level in school_data[to_building_id]['levels']
                    for cls in level.get('classrooms', [])
                ) else 'corridor'
            else:
                to_obj_type = 'corridor'
            
            to_node_name = f"{to_obj_name}-p0" if to_obj_type == 'corridor' else to_obj_name
            to_node_id = graph.node_id_map.get((to_building_id, to_obj_type, to_node_name, to_level))

            if from_node_id and to_node_id:
                from_coords = graph.nodes[from_node_id]['coordinates']
                to_coords = graph.nodes[to_node_id]['coordinates']
                if building_id != to_building_id:
                    distance = euclidean_distance(from_coords, to_coords, floor_penalty=0)
                else:
                    distance = euclidean_distance(from_coords, to_coords, floor_penalty=15.0)
                graph.add_edge(from_node_id, to_node_id, distance)

        a_building_id = 'buildingA'
        b_building_id = 'buildingB'
        c_building_id = 'buildingC'
        
        ab_connect_level = 'level1'
        a_b_corr_name = 'connectToBuildingB-p1'
        a_b_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_b_corr_name, ab_connect_level))
        b_a_corr_name = 'connectToBuildingAAndC-p1'
        b_a_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_a_corr_name, ab_connect_level))
        
        if a_b_node_id and b_a_node_id:
            coords_a = graph.nodes[a_b_node_id]['coordinates']
            coords_b = graph.nodes[b_a_node_id]['coordinates']
            distance = euclidean_distance(coords_a, coords_b, floor_penalty=0)
            graph.add_edge(a_b_node_id, b_a_node_id, distance)
        
        bc_connect_level = 'level1'
        b_c_corr_name = 'connectToBuildingAAndC-p0'
        b_c_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_c_corr_name, bc_connect_level))
        c_b_corr_name = 'connectToBuildingB-p1'
        c_b_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_b_corr_name, bc_connect_level))
        
        if b_c_node_id and c_b_node_id:
            coords_b = graph.nodes[b_c_node_id]['coordinates']
            coords_c = graph.nodes[c_b_node_id]['coordinates']
            distance = euclidean_distance(coords_b, coords_c, floor_penalty=0)
            graph.add_edge(b_c_node_id, c_b_node_id, distance)
        
        connect_level1 = 'level1'
        a_corr1_name = 'connectToBuildingC-p3'
        a_connect1_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr1_name, connect_level1))
        c_corr1_name = 'connectToBuildingA-p0'
        c_connect1_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr1_name, connect_level1))
        
        if a_connect1_node_id and c_connect1_node_id:
            coords_a = graph.nodes[a_connect1_node_id]['coordinates']
            coords_c = graph.nodes[c_connect1_node_id]['coordinates']
            distance = euclidean_distance(coords_a, coords_c, floor_penalty=0)
            graph.add_edge(a_connect1_node_id, c_connect1_node_id, distance)
        
        connect_level3 = 'level3'
        a_corr3_name = 'connectToBuildingC-p2'
        a_connect3_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr3_name, connect_level3))
        c_corr3_name = 'connectToBuildingA-p0'
        c_connect3_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr3_name, connect_level3))
        
        if a_connect3_node_id and c_connect3_node_id:
            coords_a = graph.nodes[a_connect3_node_id]['coordinates']
            coords_c = graph.nodes[c_connect3_node_id]['coordinates']
            distance = euclidean_distance(coords_a, coords_c, floor_penalty=0)
            graph.add_edge(a_connect3_node_id, c_connect3_node_id, distance)

    return graph

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

def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path if len(path) > 1 else None

# ====================== 导航函数：自动补全深黄色 forward ======================
def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    valid_buildings = ['A', 'B', 'C', 'Gate']
    if start_building not in valid_buildings or end_building not in valid_buildings:
        return None, "Invalid building selection, only Buildings A, B, C and Gate are supported", None, None
        
    try:
        start_key = (start_building, start_classroom, start_level)
        end_key = (end_building, end_classroom, end_level)
        
        start_node = graph.node_id_map.get(start_key)
        end_node = graph.node_id_map.get(end_key)
        
        if not start_node:
            start_node = f"{start_building}-classroom-{start_classroom}@{start_level}"
        if not end_node:
            end_node = f"{end_building}-classroom-{end_classroom}@{end_level}"

        if start_node not in graph.nodes:
            return None, f"Starting classroom does not exist: {start_building}{start_classroom}@{start_level}", None, None
        if end_node not in graph.nodes:
            return None, f"Destination classroom does not exist: {end_building}{end_classroom}@{end_level}", None, None

        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)

        if path:
            total_distance = distances[end_node]
            simplified_path = []
            path_stairs = set()
            prev_building = None
            
            for i in range(len(path)):
                node_id = path[i]
                node_info = graph.nodes[node_id]
                node_type = node_info['type']
                node_name = node_info['name']
                node_level = node_info['level']
                node_building = node_info['building']
                
                node_desc = ""
                if node_type == 'stair':
                    path_stairs.add((node_building, node_name, node_level))
                    node_desc = f"Building {node_building} {node_name} ({node_level})"
                elif node_type == 'classroom':
                    node_desc = f"Building {node_building} {node_name} ({node_level})"
                elif node_type == 'corridor':
                    if 'connectToBuilding' in node_name or 'gateTo' in node_name:
                        if 'connectToBuildingA' in node_name or 'gateToA' in node_name:
                            connected_building = 'A'
                        elif 'connectToBuildingB' in node_name or 'gateToB' in node_name:
                            connected_building = 'B'
                        elif 'connectToBuildingC' in node_name or 'gateToC' in node_name:
                            connected_building = 'C'
                        elif 'gateTo' in node_name:
                            connected_building = 'Gate'
                        else:
                            connected_building = 'Other'
                            
                        if prev_building and prev_building != node_building:
                            node_desc = f"Cross corridor from Building {prev_building} to Building {node_building} ({node_level})"
                
                if node_desc:
                    if i < len(path) - 1:
                        next_node_id = path[i+1]
                        direction = get_direction_between_nodes(graph, node_id, next_node_id)
                        
                        # 自动补全 深黄色 forward
                        if not direction:
                            direction = "<span style='color:DarkGoldenRod; font-weight:bold;'>forward</span>"
                        
                        if direction:
                            node_desc += f" {direction}"
                    
                    simplified_path.append(node_desc)
                
                if node_type in ['classroom', 'stair', 'corridor']:
                    prev_building = node_building
            
            full_path_str = " → ".join(simplified_path)
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

def get_classroom_info(school_data):
    try:
        buildings = [b for b in school_data.keys() if b.startswith('building') or b == 'gate']
        building_names = []
        for b in buildings:
            if b == 'gate':
                building_names.append('Gate')
            else:
                building_names.append(b.replace('building', ''))
        
        classrooms_by_building = {}
        levels_by_building = {}
        
        for building_id in buildings:
            if building_id == 'gate':
                building_name = 'Gate'
            else:
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

def reset_app_state():
    st.session_state['display_options'] = {
        'start_level': None,
        'end_level': None,
        'path_stairs': set(),
        'show_all': True,
        'path': [],
        'start_building': None,
        'end_building': None
    }
    st.session_state['current_path'] = None
    if 'path_result' in st.session_state:
        del st.session_state['path_result']

# --------------------------
# Page Logic
# --------------------------
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'
    if 'display_options' not in st.session_state:
        st.session_state['display_options'] = {
            'start_level': None,
            'end_level': None,
            'path_stairs': set(),
            'show_all': True,
            'path': [],
            'start_building': None,
            'end_building': None
        }
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = None

    # --------------------------
    # Welcome Page with Background
    # --------------------------
    if st.session_state['page'] == 'welcome':
        def add_bg_from_local(image_file):
            with open(image_file, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            css = f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: url("data:image/jpeg;base64,{encoded}");
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
                background-attachment: fixed !important;
            }}

            h1 {{
                color: white !important;
                text-align: center !important;
                margin-top: 25vh !important;
                font-size: 48px !important;
                font-weight: 900 !important;
            }}

            div.stButton > button:first-child {{
                background-color: #4682B4 !important;
                color: white !important;
                font-size: 20px !important;
                height: 60px !important;
                width: 280px !important;
                border-radius: 12px !important;
                border: none !important;
                font-weight: bold !important;
                display: block !important;
                margin: 30px 650px auto !important;
            }}

            div.stButton > button:first-child:hover {{
                background-color: #45a049 !important;
            }}

            </style>
            """
            st.markdown(css, unsafe_allow_html=True)

        add_bg_from_local("background.jpg")

    # --------------------------
    # Welcome Page
    # --------------------------
    if st.session_state['page'] == 'welcome':
        if 'worksheet' not in st.session_state:
            st.session_state['worksheet'] = init_google_sheet()
        
        total_accesses = get_total_accesses(st.session_state['worksheet'])
        
        st.markdown("<h1>NAVIGATE YOUR CAMPUS</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:white; font-size:20px; opacity:0.9;'>Find Classrooms, labs, resources in stunning 3D</p>", unsafe_allow_html=True)

        
        if st.button('EXPLORE 3D MAP'):
            update_access_count(st.session_state['worksheet'])
            st.session_state['page'] = 'main'
            st.rerun()

    # --------------------------
    # Main Interface
    # --------------------------
    else:
        with st.sidebar:
            st.header("📍 Select Locations")
            school_data = load_school_data_detailed('school_data_detailed.json')
            if school_data is None:
                st.error("Failed to load school data!")
                return
            building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
            
            st.subheader("Start Point")
            start_building = st.selectbox("Building", building_names, key="start_building")
            start_levels = levels_by_building.get(start_building, [])
            start_level = st.selectbox("Floor", start_levels, key="start_level")
            start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
            start_classroom = st.selectbox("Classroom", start_classrooms, key="start_classroom")

            st.subheader("End Point")
            end_building = st.selectbox("Building", building_names, key="end_building")
            end_levels = levels_by_building.get(end_building, [])
            end_level = st.selectbox("Floor", end_levels, key="end_level")
            end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
            end_classroom = st.selectbox("Classroom", end_classrooms, key="end_classroom")

            st.divider()
            nav_button = st.button("🔍 Find Shortest Path", use_container_width=True)
            reset_button = st.button("🔄 Reset View", use_container_width=True)
            exit_button = st.button("🚪 Back to Welcome", use_container_width=True)

            if reset_button:
                reset_app_state()
                st.rerun()
            if exit_button:
                reset_app_state()
                st.session_state['page'] = 'welcome'
                st.rerun()

        st.markdown("<h2 style='margin:0; padding:0; text-align:left; line-height:1.2;'>🏫 SCIS Campus Navigation System</h2>", unsafe_allow_html=True)
        
        
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            st.error("Failed to load data file!")
            return
        
        graph = build_navigation_graph(school_data)
        st.success("✅ Campus data loaded successfully!")

        if nav_button:
            try:
                path, message, simplified_path, display_options = navigate(
                    graph, start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                if path and display_options:
                    st.success(f"📊 Navigation Result: {message}")
                    st.markdown("#### 🛤️ Path Details")
                    # 启用 HTML 渲染，显示深黄色高亮
                    st.markdown(f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px;'>{simplified_path}</div>", unsafe_allow_html=True)
                    st.session_state['current_path'] = path
                    st.session_state['display_options'] = display_options
            except Exception as e:
                st.error(f"Error: {e}")

        if st.session_state['current_path'] is not None:
            fig = plot_3d_map(school_data, graph, st.session_state['display_options'])[0]
        else:
            fig = plot_3d_map(school_data, graph)[0]
        
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={
                'displayModeBar': True,
                'scrollZoom': True,
                'editable': False
            },
            theme="streamlit",
            kwargs={"layout": {"margin": {"t": 10}}}
        )

if __name__ == "__main__":
    main()
