import json
from config_reader import get_latest_people_flow
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
import copy

# 导入多语言支持
from lang_utils import init_language, get_lang, get_text, language_selector, get_direction_text, get_node_type_text
from languages import LANGUAGES

# ====================== 移动端适配核心：页面配置 ======================
st.set_page_config(
    page_title="SCIS Navigation System",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items=None
)

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
# Color Scheme（新增电梯配色）
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
    'building_label': {'A': 'darkblue', 'B': 'darkgreen', 'C': 'darkred', 'Gate': 'darkgoldenrod'},
    # 电梯配色新增
    'elevator': {
        'ElevatorB1': '#00BFFF'
    },
    'elevator_label': 'darkblue',
    'path_elevator': 'deepskyblue'
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
                    show_level = (level_name == 'level1') or any((building_name, s, level_name) in path_stairs for s in ['StairsB1','StairsB2','ElevatorB1'])
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

            # 绘制楼梯
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

            # ========== 绘制电梯 ==========
            if 'elevators' in level:
                for elevator in level['elevators']:
                    elev_name = elevator['name']
                    is_path = (building_name, elev_name, level_name) in path_stairs
                    x, y, _ = elevator['coordinates']
                    z = raw_z + 10
                    elev_color = COLORS['elevator'].get(elev_name, 'deepskyblue')
                    marker_size = 13 if is_path else 10

                    legend_key = f"{building_name}-{elev_name}"
                    show_leg = legend_key not in shown_stairs_legends
                    if show_leg:
                        shown_stairs_legends.add(legend_key)

                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z],
                        mode='markers+text',
                        marker=dict(color=elev_color, size=marker_size, symbol='square-open', line=dict(color='black', width=1.5)),
                        text=elev_name, textposition="top center", textfont=dict(size=9, color=COLORS['elevator_label']),
                        name=legend_key,
                        legendgroup="Elevators",
                        showlegend=show_leg
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

    # ====================== 手机端自动关闭图例，电脑端显示图例 ======================
    is_mobile = False
    try:
        ua = st.context.headers.get("User-Agent", "").lower()
        is_mobile = any(m in ua for m in ["mobile", "android", "iphone", "ipad", "phone"])
    except:
        pass

    fig.update_layout(
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Floor (Z+10)",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.8)
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        showlegend=False if is_mobile else True
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

# ====================== 方向函数：电梯上下识别 ======================
def get_direction_between_nodes(graph, current_node_id, next_node_id):
    current_node = graph.nodes[current_node_id]
    next_node = graph.nodes[next_node_id]
    
    curr_x, curr_y, curr_z = current_node['coordinates']
    next_x, next_y, next_z = next_node['coordinates']
    
    curr_is_stair = current_node['type'] == 'stair'
    next_is_stair = next_node['type'] == 'stair'
    curr_is_elev = current_node['type'] == 'elevator'
    next_is_elev = next_node['type'] == 'elevator'

    # 电梯上下楼层提示
    if (curr_is_elev and next_is_elev) or (curr_is_stair and next_is_stair):
        if next_z > curr_z:
            return get_direction_text('up')
        elif next_z < curr_z:
            return get_direction_text('down')
        else:
            return ""
    
    x_diff = next_x - curr_x
    y_diff = next_y - curr_y
    threshold = 0.1
    
    if abs(x_diff) > threshold or abs(y_diff) > threshold:
        if y_diff > threshold:
            return get_direction_text('forward')
        elif y_diff < -threshold:
            return get_direction_text('backward')
        elif x_diff > threshold:
            return get_direction_text('right')
        elif x_diff < -threshold:
            return get_direction_text('left')
    
    return ""

def build_navigation_graph(school_data):
    graph = Graph()

    for building_id in school_data.keys():
        if not (building_id.startswith('building') or building_id == 'gate'):
            continue
            
        building_data = school_data[building_id]
        
        for level in building_data['levels']:
            level_name = level['name']

            # 1. 添加教室节点
            for classroom in level['classrooms']:
                class_name = classroom['name']
                graph.add_node(
                    building_id=building_id,
                    node_type='classroom',
                    name=class_name,
                    level=level_name,
                    coordinates=classroom['coordinates']
                )

            # 2. 添加电梯节点
            if 'elevators' in level:
                for elevator in level['elevators']:
                    graph.add_node(
                        building_id=building_id,
                        node_type='elevator',
                        name=elevator['name'],
                        level=level_name,
                        coordinates=elevator['coordinates']
                    )

            # 3. 添加楼梯节点
            for stair in level['stairs']:
                graph.add_node(
                    building_id=building_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

            # 4. 添加走廊节点
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

    # ========== 连通所有节点 ==========
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
            
            # 获取当前楼层所有走廊节点
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]

            # 走廊点位前后相连
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

            # 近距离走廊节点互通
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2, floor_penalty=0)
                    
                    if distance < 3.0:
                        graph.add_edge(node1_id, node2_id, distance)

            # 教室绑定最近走廊
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

            # 楼梯绑定最近走廊
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

            # ========== 电梯绑定最近走廊 ==========
            elevator_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name
                and node_info['type'] == 'elevator'
                and node_info['level'] == level_name
            ]
            for elev_node_id in elevator_nodes:
                elev_coords = graph.nodes[elev_node_id]['coordinates']
                min_dist = float('inf')
                nearest_corr = None
                for corr_node_id in corr_nodes:
                    corr_coords = graph.nodes[corr_node_id]['coordinates']
                    d = euclidean_distance(elev_coords, corr_coords, floor_penalty=0)
                    if d < min_dist:
                        min_dist = d
                        nearest_corr = corr_node_id
                if nearest_corr:
                    graph.add_edge(elev_node_id, nearest_corr, min_dist)

        # ========== 楼梯跨楼层竖向连通 ==========
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

        # ========== 电梯跨楼层竖向连通 ==========
        elevator_names = set()
        for node_id, node_info in graph.nodes.items():
            if node_info['type'] == 'elevator':
                elevator_names.add((node_info['building'], node_info['name']))

        for (building, elev_name) in elevator_names:
            elev_level_nodes = []
            for node_id, node_info in graph.nodes.items():
                if (node_info['building'] == building and
                    node_info['type'] == 'elevator' and
                    node_info['name'] == elev_name):
                    elev_level_nodes.append((node_id, node_info['coordinates'], node_info['level']))
            
            elev_level_nodes.sort(key=lambda x: x[1][2])
            for i in range(len(elev_level_nodes)-1):
                n1, c1, _ = elev_level_nodes[i]
                n2, c2, _ = elev_level_nodes[i+1]
                dist = euclidean_distance(c1, c2, floor_penalty=15.0)
                graph.add_edge(n1, n2, dist)

        # ========== 楼宇之间跨楼走廊连通逻辑（优化版：自动连接走廊末端） ==========
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
            
            # ========== 修改点1：自动找走廊的末端节点 ==========
            if from_obj_type == 'corridor':
                # 获取该走廊在当前楼层的所有节点
                from_corr_nodes = []
                for nid, info in graph.nodes.items():
                    if (info['building'] == building_name and 
                        info['type'] == 'corridor' and 
                        info['level'] == from_level and 
                        from_obj_name in info['name']):
                        from_corr_nodes.append(nid)
                
                if from_corr_nodes:
                    # 按坐标排序，取最后一个（末端）
                    from_corr_nodes.sort(key=lambda nid: (graph.nodes[nid]['coordinates'][0], graph.nodes[nid]['coordinates'][1]))
                    from_node_name = from_corr_nodes[-1].split('-')[-1]
                    from_node_name = f"{from_obj_name}-{from_node_name}"
                else:
                    from_node_name = f"{from_obj_name}-p0"
            else:
                from_node_name = from_obj_name
            
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
            
            # ========== 修改点2：自动找目标走廊的起点 ==========
            if to_obj_type == 'corridor':
                target_building_name = to_building_id.replace('building', '') if to_building_id != 'gate' else 'Gate'
                to_corr_nodes = []
                for nid, info in graph.nodes.items():
                    if (info['building'] == target_building_name and
                        info['type'] == 'corridor' and
                        info['level'] == to_level and
                        to_obj_name in info['name']):
                        to_corr_nodes.append(nid)
                
                if to_corr_nodes:
                    # 按坐标排序，取第一个（起点）
                    to_corr_nodes.sort(key=lambda nid: (graph.nodes[nid]['coordinates'][0], graph.nodes[nid]['coordinates'][1]))
                    to_node_name = to_corr_nodes[0].split('-')[-1]
                    to_node_name = f"{to_obj_name}-{to_node_name}"
                else:
                    to_node_name = f"{to_obj_name}-p0"
            else:
                to_node_name = to_obj_name
            
            to_node_id = graph.node_id_map.get((to_building_id, to_obj_type, to_node_name, to_level))

            if from_node_id and to_node_id:
                from_coords = graph.nodes[from_node_id]['coordinates']
                to_coords = graph.nodes[to_node_id]['coordinates']
                if building_id != to_building_id:
                    distance = euclidean_distance(from_coords, to_coords, floor_penalty=0)
                else:
                    distance = euclidean_distance(from_coords, to_coords, floor_penalty=15.0)
                graph.add_edge(from_node_id, to_node_id, distance)

        # AB、BC、AC楼宇互通（原有代码原样保留）
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

        # ========== 新增：AC楼宇 level2 互通 ==========
        connect_level2 = 'level2'
        a_corr2_name = 'connectToBuildingC-p3'   # A楼 level2 走廊末端
        a_connect2_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr2_name, connect_level2))
        c_corr2_name = 'connectToBuildingA-p0'   # C楼 level2 走廊起点
        c_connect2_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr2_name, connect_level2))

        if a_connect2_node_id and c_connect2_node_id:
            coords_a = graph.nodes[a_connect2_node_id]['coordinates']
            coords_c = graph.nodes[c_connect2_node_id]['coordinates']
            distance = euclidean_distance(coords_a, coords_c, floor_penalty=0)
            graph.add_edge(a_connect2_node_id, c_connect2_node_id, distance)

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

# ====================== 导航核心函数：模式切换 ======================
def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    valid_buildings = ['A', 'B', 'C', 'Gate']
    if start_building not in valid_buildings or end_building not in valid_buildings:
        return None, get_text('building_not_found'), None, None
        
    try:
        corridor_blocked = False
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

        # ===== 根据用户选择切换模式 =====
        temp_graph = copy.deepcopy(graph)
        
        if st.session_state.get("is_disabled", False):
            # YES模式：只禁用B楼的楼梯，保留电梯
            for nid, node_data in temp_graph.nodes.items():
                if node_data['type'] == 'stair' and node_data['building'] == 'B':
                    node_data['neighbors'] = {}
        else:
            # NO模式：禁用所有电梯，只使用楼梯
            for nid, node_data in temp_graph.nodes.items():
                if node_data['type'] == 'elevator':
                    node_data['neighbors'] = {}

        # ===== 新增逻辑：人流量检测（完全独立，不影响无障碍） =====             
        
        
        # 从 Supabase 读取最新人流量状态
        latest_flow = get_latest_people_flow()
       
        
        if latest_flow == 1:
            # 封锁 A-C 二楼连廊节点
            corridor_nodes_to_block = []
            for nid, node_data in temp_graph.nodes.items():
                if node_data['type'] == 'corridor' and node_data['level'] == 'level2':
                    # 判断是否是 A-C 连廊节点（根据JSON中的名称）
                    if 'connectToBuildingC' in node_data['name'] or 'connectToBuildingA' in node_data['name']:
                        corridor_nodes_to_block.append(nid)
            
            # 切断这些节点的所有连接（封锁路径）
            for nid in corridor_nodes_to_block:
                temp_graph.nodes[nid]['neighbors'] = {}
            
            if corridor_nodes_to_block:
                corridor_blocked = True

        distances, previous_nodes = dijkstra(temp_graph, start_node)
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
                # 楼梯、电梯都存入高亮集合
                if node_type == 'stair':
                    path_stairs.add((node_building, node_name, node_level))
                    node_desc = f"Building {node_building} {node_name} ({node_level})"
                elif node_type == 'elevator':
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
                            node_desc = get_text('cross_corridor').format(prev_building, node_building, node_level)
                
                if node_desc:
                    if i < len(path) - 1:
                        next_node_id = path[i+1]
                        direction = get_direction_between_nodes(graph, node_id, next_node_id)
                        
                        if not direction:
                            direction = get_direction_text('forward')
                        
                        if direction:
                            node_desc += f" {direction}"
                    
                    simplified_path.append(node_desc)
                
                if node_type in ['classroom', 'stair', 'corridor', 'elevator']:
                    prev_building = node_building
            
            full_path_str = " → ".join(simplified_path)
            if corridor_blocked:
                full_path_str = "🚧 [A-C二楼连廊已封锁，已自动绕行] " + full_path_str
            display_options = {
                'start_level': start_level,
                'end_level': end_level,
                'path_stairs': path_stairs,
                'show_all': False,
                'path': path,
                'start_building': start_building,
                'end_building': end_building
            }
            return path, get_text('total_distance').format(total_distance), full_path_str, display_options
        else:
            return None, get_text('no_path'), None, None
    except Exception as e:
        return None, get_text('navigation_error').format(str(e)), None, None

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
    # 重置无障碍选项
    st.session_state['is_disabled'] = False

# --------------------------
# Page Logic
# --------------------------
def main():
    # 初始化语言
    init_language()
    
    # 会话状态初始化
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
    if 'is_disabled' not in st.session_state:
        st.session_state['is_disabled'] = False

    # 欢迎页面
    if st.session_state['page'] == 'welcome':
        def add_bg_from_local(image_file):
            try:
                with open(image_file, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                css = """
                <style>
                .stApp {
                    background-image: url("data:image/jpeg;base64,%s");
                    background-size: cover !important;
                    background-position: center !important;
                    background-repeat: no-repeat !important;
                    background-attachment: fixed !important;
                }
                .welcome-container {
                    display: flex !important;
                    flex-direction: column !important;
                    align-items: center !important;
                    text-align: center !important;
                    width: 100%% !important;
                    margin-top: 35vh !important;
                }
                .welcome-title {
                    color: white !important;
                    font-size: clamp(28px, 8vw, 48px) !important;
                    font-weight: 900 !important;
                    white-space: nowrap !important;
                    margin: 0 !important;
                }
                .welcome-subtitle {
                    color: white !important;
                    font-size: clamp(14px, 3vw, 20px) !important;
                    opacity: 0.9 !important;
                    margin: 5px 0 25px 0 !important;
                }
                /* 访问次数样式 - 白色字体 */
                .visit-counter {
                    color: white !important;
                    font-size: clamp(14px, 2vw, 18px) !important;
                    opacity: 0.85 !important;
                    margin-top: 10px !important;
                }
                div.stButton > button:first-child {
                    background-color: #4682B4 !important;
                    color: white !important;
                    font-size: clamp(16px, 4vw, 20px) !important;
                    padding: 16px 24px !important;
                    border-radius: 12px !important;
                    font-weight: bold !important;
                    border: none !important;
                    width: 100%% !important;
                }
                </style>
                """ % encoded
                st.markdown(css, unsafe_allow_html=True)
            except:
                pass

        add_bg_from_local("background.jpg")

        if 'worksheet' not in st.session_state:
            st.session_state['worksheet'] = init_google_sheet()

        # ===== 获取访问次数 =====
        total_visits = get_total_accesses(st.session_state['worksheet'])

        st.markdown(f"""
        <div class="welcome-container">
            <h1 class="welcome-title">{get_text('welcome_title')}</h1>
            <div class="welcome-subtitle">{get_text('welcome_subtitle')}</div>
        </div>
        """, unsafe_allow_html=True)

        col_empty1, col_center, col_empty2 = st.columns([1, 0.5, 1])
        with col_center:
            if st.button(get_text('explore_3d')):
                update_access_count(st.session_state['worksheet'])
                st.session_state['page'] = 'main'
                st.rerun()
            
            # ===== 在按钮正下方显示访问次数 =====
            st.markdown(f"""
            <div class="visit-counter">
                👀 Total Visits: {total_visits}
            </div>
            """, unsafe_allow_html=True)

    # 主导航界面
    else:
        with st.sidebar:
            # 语言选择器
            language_selector()
            st.divider()
            
            # 无障碍设置区域
            st.subheader(get_text('accessibility_setting'))
            access_choice = st.radio(
                get_text('barrier_free_access'),
                options=["No", "Yes"],
                index=0,
                help=get_text('select_no')
            )
            # 更新session state
            st.session_state['is_disabled'] = (access_choice == "Yes")
            st.divider()

            st.header(get_text('select_locations'))
            school_data = load_school_data_detailed('school_data_detailed.json')
            if school_data is None:
                st.error(get_text('loading_error'))
                return
            building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
            
            st.subheader(get_text('start_point'))
            start_building = st.selectbox(get_text('building'), building_names, key="start_building")
            start_levels = levels_by_building.get(start_building, [])
            start_level = st.selectbox(get_text('floor'), start_levels, key="start_level")
            start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
            start_classroom = st.selectbox(get_text('classroom'), start_classrooms, key="start_classroom")

            st.subheader(get_text('end_point'))
            end_building = st.selectbox(get_text('building'), building_names, key="end_building")
            end_levels = levels_by_building.get(end_building, [])
            end_level = st.selectbox(get_text('floor'), end_levels, key="end_level")
            end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
            end_classroom = st.selectbox(get_text('classroom'), end_classrooms, key="end_classroom")

            st.divider()
            nav_button = st.button(get_text('find_path'), use_container_width=True)
            reset_button = st.button(get_text('reset_view'), use_container_width=True)
            exit_button = st.button(get_text('back_to_welcome'), use_container_width=True)

            if reset_button:
                reset_app_state()
                st.rerun()
            if exit_button:
                reset_app_state()
                st.session_state['page'] = 'welcome'
                st.rerun()

        st.markdown(f"<h2 style='margin:0; padding:0; text-align:left; line-height:1.2; font-size:clamp(18px,5vw,26px);'>{get_text('campus_navigation')}</h2>", unsafe_allow_html=True)
        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)
        
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            st.error(get_text('loading_error'))
            return
        
        graph = build_navigation_graph(school_data)
        st.success(get_text('data_loaded'))

        display_options = st.session_state['display_options']

        if nav_button:
            try:
                path, message, simplified_path, new_display_options = navigate(
                    graph, start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                if path and new_display_options:
                    st.success(f"{get_text('navigation_result')} {message}")
                    st.markdown(f"#### {get_text('path_details')}")
                    st.markdown(f"<div style='background-color:#f0f2f6; padding:10px; border-radius:5px; word-wrap:break-word; white-space:normal;'>{simplified_path}</div>", unsafe_allow_html=True)
                    st.session_state['display_options'] = new_display_options
                    display_options = new_display_options
                elif path is None and message:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(get_text('navigation_error').format(str(e)))

        # 渲染3D地图
        fig, _ = plot_3d_map(school_data, graph, display_options)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
