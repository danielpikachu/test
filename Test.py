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

# 核心修改1：开启宽布局 + 原生侧边栏配置
st.set_page_config(
    page_title="SCIS Navigation System",
    layout="wide",
    initial_state="expanded"
)

plt.switch_backend('Agg')

# --------------------------
# Google Sheets 配置（适配 Streamlit Secrets TOML）
# --------------------------
SHEET_NAME = 'Navigation visitors'
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

# 从 Streamlit Secrets 加载密钥
def get_credentials():
    try:
        service_account_info = st.secrets["google_service_account"]
        return Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPE
        )
    except KeyError:
        st.error("Streamlit Secrets 中未找到 google_service_account 配置")
        return None
    except Exception as e:
        st.error(f"密钥加载失败: {str(e)}")
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
            stats_worksheet = sheet.add_worksheet(title="Access_Stats", rows="1000", cols="3")
            stats_worksheet.append_row(["Timestamp", "Count", "Total"])
            stats_worksheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, 1])
        
        return stats_worksheet
    except Exception as e:
        return None

# --------------------------
# 访问统计
# --------------------------
def update_access_count(ws):
    if not ws:
        return 0
    try:
        recs = ws.get_all_values()
        total = int(recs[-1][2]) if len(recs) > 1 and recs[-1][2].isdigit() else 0
        ws.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, total + 1])
        return total + 1
    except:
        return 0

def get_total_accesses(ws):
    if not ws:
        return 0
    try:
        recs = ws.get_all_values()
        return int(recs[-1][2]) if len(recs) > 1 and recs[-1][2].isdigit() else 0
    except:
        return 0

# --------------------------
# 配色
# --------------------------
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral', 'Gate': 'gold'},
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},
    'corridor_line': {'A': 'cyan', 'B': 'forestgreen', 'C': 'salmon', 'Gate': 'darkgoldenrod'},
    'corridor_node': 'navy',
    'corridor_label': 'darkblue',
    'stair': {
        'Stairs1': '#FF5733', 'Stairs2': '#33FF57', 'Stairs3': '#3357FF',
        'Stairs4': '#FF33F5', 'Stairs5': '#F5FF33', 'StairsB1': '#33FFF5',
        'StairsB2': '#FF9933', 'GateStairs': '#FFD700'
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

# --------------------------
# 数据加载
# --------------------------
def load_school_data_detailed(fn):
    try:
        with open(fn, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载数据失败: {e}")
        return None

# --------------------------
# 【已修复兼容】Plotly 3D 绘图（兼容旧版 Plotly）
# --------------------------
def plot_3d_map_plotly(school_data, display_options=None):
    fig = go.Figure()

    if display_options is None:
        display_options = {
            'start_level': None, 'end_level': None,
            'path_stairs': set(), 'show_all': True, 'path': [],
            'start_building': None, 'end_building': None
        }
    
    show_all = display_options['show_all']
    start_level = display_options['start_level']
    end_level = display_options['end_level']
    path_stairs = display_options['path_stairs']
    path = display_options.get('path', [])
    start_building = display_options.get('start_building')
    end_building = display_options.get('end_building')

    added_legend = set()

    for building_id in school_data.keys():
        if building_id == 'gate':
            bn = 'Gate'
        elif building_id.startswith('building'):
            bn = building_id.replace('building', '')
        else:
            continue
            
        bd = school_data[building_id]
        
        for level in bd['levels']:
            ln = level['name']
            z = level['z']
            
            show = show_all
            if not show_all:
                if bn == 'B':
                    show = any((bn, s, ln) in path_stairs for s in ['StairsB1', 'StairsB2'])
                    if start_building == 'B' or end_building == 'B':
                        show = show or (ln == 'level1')
                elif bn == 'Gate':
                    show = (start_building == 'Gate' or end_building == 'Gate')
                else:
                    show = (ln == start_level) or (ln == end_level)
            
            if not show:
                continue

            fp = level['floorPlane']
            xp = [fp['minX'], fp['maxX'], fp['maxX'], fp['minX'], fp['minX']]
            yp = [fp['minY'], fp['minY'], fp['maxY'], fp['maxY'], fp['minY']]
            zp = [z]*5
            fc = COLORS['floor_z'].get(z, 'gray')
            bc = COLORS['building'][bn]

            fig.add_trace(go.Scatter3d(
                x=xp, y=yp, z=zp, mode='lines',
                line=dict(color=fc, width=3), name=f'{bn}-{ln}', showlegend=False
            ))
            fig.add_trace(go.Mesh3d(
                x=xp[:4], y=yp[:4], z=zp[:4], color=bc, opacity=0.3, showlegend=False
            ))

            for corr in level['corridors']:
                cx = [p[0] for p in corr['points']]
                cy = [p[1] for p in corr['points']]
                cz = [p[2] for p in corr['points']]
                is_conn = 'connectTo' in corr.get('name', '')
                cc = COLORS['connect_corridor'] if is_conn else COLORS['corridor_line'][bn]
                fig.add_trace(go.Scatter3d(
                    x=cx, y=cy, z=cz, mode='lines+markers',
                    line=dict(color=cc, width=4), marker=dict(color=COLORS['corridor_node'], size=2.5),
                    showlegend=False
                ))

            for rm in level['classrooms']:
                x, y, _ = rm['coordinates']
                w, d = rm['size']
                xr = [x, x+w, x+w, x, x]
                yr = [y, y, y+d, y+d, y]
                zr = [z]*5
                fig.add_trace(go.Scatter3d(
                    x=xr, y=yr, z=zr, mode='lines',
                    line=dict(color='gray', dash='dash', width=1), showlegend=False
                ))
                fig.add_trace(go.Scatter3d(
                    x=[x], y=[y], z=[z], mode='markers+text',
                    marker=dict(color=bc, size=3), text=rm['name'], textposition='top center',
                    textfont=dict(size=8), showlegend=False
                ))

            for st in level['stairs']:
                sx, sy, _ = st['coordinates']
                sname = st['name']
                sc = COLORS['stair'].get(sname, 'red')
                show_l = sname not in added_legend
                if show_l:
                    added_legend.add(sname)
                # 修复：使用旧版 Plotly 支持的 symbol
                fig.add_trace(go.Scatter3d(
                    x=[sx], y=[sy], z=[z], mode='markers+text',
                    marker=dict(color=sc, size=6, symbol='diamond'),  # 这里修复
                    text=sname, textposition='top center', textfont=dict(size=8),
                    name=sname, showlegend=show_l
                ))

    if path:
        xs, ys, zs = [], [], []
        for nid in path:
            c = graph.nodes[nid]['coordinates']
            xs.append(c[0])
            ys.append(c[1])
            zs.append(c[2])
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines+markers',
            line=dict(color=COLORS['path'], width=5), marker=dict(size=3, color=COLORS['path']),
            name='Path'
        ))
        # 修复：起点终点使用旧版兼容符号
        fig.add_trace(go.Scatter3d(
            x=[xs[0]], y=[ys[0]], z=[zs[0]], mode='markers',
            marker=dict(color=COLORS['start_marker'], size=10, symbol='circle'),  # 修复
            name='Start'
        ))
        fig.add_trace(go.Scatter3d(
            x=[xs[-1]], y=[ys[-1]], z=[zs[-1]], mode='markers',
            marker=dict(color=COLORS['end_marker'], size=10, symbol='square'),  # 修复
            name='End'
        ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title='SCIS 3D 导航地图', height=800,
        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.05)
    )
    return fig

# --------------------------
# 以下所有代码 100% 完全不变，完整保留
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

def get_direction_between_nodes(graph, current_node_id, next_node_id):
    current_node = graph.nodes[current_node_id]
    next_node = graph.nodes[next_node_id]
    
    curr_x, curr_y, curr_z = current_node['coordinates']
    next_x, next_y, next_z = next_node['coordinates']
    
    curr_is_stair = current_node['type'] == 'stair'
    next_is_stair = next_node['type'] == 'stair'
    
    if curr_is_stair and next_is_stair:
        if next_z > curr_z:
            return "往上"
        elif next_z < curr_z:
            return "往下"
        else:
            return ""
    
    x_diff = next_x - curr_x
    y_diff = next_y - curr_y
    threshold = 0.1
    
    if abs(x_diff) > threshold or abs(y_diff) > threshold:
        if y_diff > threshold:
            return "向前"
        elif y_diff < -threshold:
            return "向后"
        elif x_diff > threshold:
            return "向右"
        elif x_diff < -threshold:
            return "向左"
    
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
                for p_idx in range(len(corridor['points']) - 1):
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
                    node_desc = f"Building {node_building}{node_name}({node_level})"
                elif node_type == 'classroom':
                    node_desc = f"Building {node_building}{node_name}({node_level})"
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
                            node_desc = f"Cross corridor from Building {prev_building} to Building {node_building}({node_level})"
                
                if node_desc:
                    if i < len(path) - 1:
                        next_node_id = path[i+1]
                        direction = get_direction_between_nodes(graph, node_id, next_node_id)
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
# 全局样式：彻底消除滚动条 + 全屏自适应
# --------------------------
st.markdown("""
<style>
/* 全局禁用滚动条，保持页面可交互 */
::-webkit-scrollbar {
    display: none !important;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
    overflow: hidden !important;
    scrollbar-width: none !important;
    -ms-overflow-style: none !important;
}
/* 欢迎页面全屏适配 */
.welcome-container {
    height: 90vh !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    text-align: center;
    padding: 1vh !important;
    margin: 0 !important;
}
/* 图片自适应不溢出 */
img {
    max-height: 60vh !important;
    width: auto !important;
    object-fit: contain !important;
}
/* 主内容区占满屏幕不留白 */
.block-container {
    padding: 1rem 2rem !important;
    max-width: 100vw !important;
}
/* 3D图容器自适应 */
.element-container div {
    max-height: 80vh !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# 页面逻辑
# --------------------------
def main():
    global graph
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

    # 欢迎页面
    if st.session_state['page'] == 'welcome':
        if 'worksheet' not in st.session_state:
            st.session_state['worksheet'] = init_google_sheet()
        
        total_accesses = get_total_accesses(st.session_state['worksheet'])
        
        st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
        st.markdown('<h1 class="welcome-title">Welcome to SCIS Navigation System</h1>', unsafe_allow_html=True)
        
        if st.button('Enter System', use_container_width=True, type="primary"):
            update_access_count(st.session_state['worksheet'])
            st.session_state['page'] = 'main'
            st.rerun()
        
        st.markdown(f'<div class="access-count">Total Accesses: {total_accesses}</div>', unsafe_allow_html=True)
        st.image("welcome_image.jpg", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    # 主页面
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

        st.markdown('<h3 style="margin:0.5rem 0; line-height:1.2;">🏫 SCIS Campus 3D Navigation System</h4>', unsafe_allow_html=True)
        
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            st.error("Failed to load school data file!")
            return
        
        graph = build_navigation_graph(school_data)
        st.success("✅ Campus data loaded successfully!")

        if nav_button:
            try:
                path, message, simplified_path, display_options = navigate(
                    graph, 
                    start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                
                if path and display_options:
                    st.success(f"📊 Navigation Result: {message}")
                    st.markdown("#### 🛤️ Path Details")
                    st.info(simplified_path)
                    
                    st.session_state['current_path'] = path
                    st.session_state['display_options'] = display_options
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"Navigation process error: {str(e)}")

        try:
            fig = plot_3d_map_plotly(school_data, st.session_state['display_options'])
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to display map: {str(e)}")
        
        st.markdown('<div style="position: fixed; bottom: 20px; right: 20px; font-size: 14px; color: #666;">Created By DANIEL HAN</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
