import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os

plt.switch_backend('Agg')

st.set_page_config(page_title="SCIS Navigation System")

# --------------------------
# Google Sheets 配置（适配 Streamlit Secrets TOML）
# --------------------------
SHEET_NAME = 'Navigation visitors'  # 你的Google表格名称
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

# 从 Streamlit Secrets (TOML格式) 加载密钥
def get_credentials():
    try:
        # 从Streamlit Secrets读取TOML格式的密钥（部署时用）
        # 注意：Secrets中需以 [google_service_account] 为section名
        service_account_info = st.secrets["google_service_account"]
        return Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPE
        )
    except KeyError:
        st.error("Streamlit Secrets中未找到google_service_account配置，请检查TOML格式")
        return None
    except Exception as e:
        st.error(f"密钥加载失败: {str(e)}")
        return None

def init_google_sheet():
    """初始化Google Sheets连接并确保表格结构正确"""
    try:
        # 加载凭据（使用Streamlit Secrets）
        creds = get_credentials()
        if not creds:
            return None
        client = gspread.authorize(creds)
        
        # 尝试打开表格，如果不存在则创建
        try:
            sheet = client.open(SHEET_NAME)
        except gspread.exceptions.SpreadsheetNotFound:
            sheet = client.create(SHEET_NAME)
            # 可选：共享表格给你的邮箱（方便查看）
            # sheet.share('your-email@gmail.com', perm_type='user', role='writer')
        
        # 尝试获取统计工作表，如果不存在则创建
        try:
            stats_worksheet = sheet.worksheet("Access_Stats")
        except gspread.exceptions.WorksheetNotFound:
            stats_worksheet = sheet.add_worksheet(title="Access_Stats", rows="1000", cols="3")
            # 设置表头
            stats_worksheet.append_row(["Timestamp", "Access_Count", "Total_Accesses"])
            # 初始化第一行数据
            stats_worksheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, 1])
        
        return stats_worksheet
    except Exception as e:
        st.warning(f"Google Sheets初始化失败: {str(e)}. 访问次数统计功能暂时不可用。")
        return None

# --------------------------
# 访问次数统计逻辑
# --------------------------
def update_access_count(worksheet):
    """更新访问次数统计"""
    if not worksheet:
        return 0
        
    try:
        # 获取所有记录
        records = worksheet.get_all_values()
        if len(records) < 2:  # 至少需要表头和一行数据
            return 0
            
        # 获取最后一行数据
        last_row = records[-1]
        total = int(last_row[2]) if last_row[2].isdigit() else 0
        new_total = total + 1
        
        # 添加新记录
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([current_time, 1, new_total])
        
        return new_total
    except Exception as e:
        st.warning(f"更新访问次数失败: {str(e)}")
        return 0

def get_total_accesses(worksheet):
    """获取总访问次数"""
    if not worksheet:
        return 0
        
    try:
        records = worksheet.get_all_values()
        if len(records) < 2:
            return 0
            
        last_row = records[-1]
        return int(last_row[2]) if last_row[2].isdigit() else 0
    except Exception as e:
        st.warning(f"获取总访问次数失败: {str(e)}")
        return 0

# --------------------------
# 地图与导航核心逻辑（更新支持Gate）
# --------------------------
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral', 'Gate': 'gold'},
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},
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
        'StairsGate1': '#8B4513',
        'StairsGate2': '#DAA520',
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
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data file: {str(e)}")
        return None

def plot_3d_map(school_data, display_options=None):
    fig = plt.figure(figsize=(35, 30))
    ax = fig.add_subplot(111, projection='3d')

    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

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

    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        building_data = school_data[building_id]
        
        displayed_levels = []
        max_displayed_z = -float('inf')
        max_displayed_y = -float('inf')
        corresponding_x = 0
        level_count = 0
        
        for level in building_data['levels']:
            level_name = level['name']
            z = level['z']
            
            show_level = show_all
            if not show_all:
                if building_name == 'B':
                    show_level = any((building_name, s_name, level_name) in path_stairs for s_name in ['StairsB1', 'StairsB2'])
                    if (start_building == 'B' or end_building == 'B') or (start_building in ['A','C','Gate'] and end_building in ['A','C','Gate'] and 'B' in [start_building, end_building]):
                        show_level = show_level or (level_name == 'level1')
                elif building_name == 'Gate':
                    show_level = any((building_name, s_name, level_name) in path_stairs for s_name in ['StairsGate1', 'StairsGate2'])
                    if (start_building == 'Gate' or end_building == 'Gate'):
                        show_level = show_level or (level_name == 'level1')
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
            
            floor_border_color = COLORS['floor_z'].get(z, 'gray')
            building_fill_color = COLORS['building'].get(building_name, 'lightgray')

            if show_level:
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
                
                legend_label = f"Building {building_name}-{level_name}"
                if legend_label not in ax.get_legend_handles_labels()[1]:
                    ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=4, label=legend_label)
                else:
                    ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=4)
                ax.plot_trisurf(x_plane[:-1], y_plane[:-1], z_plane[:-1], 
                                color=building_fill_color, alpha=0.3)

                for corr_idx, corridor in enumerate(level['corridors']):
                    points = corridor['points']
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    z_coords = [p[2] for p in points]
                    
                    is_external = corridor.get('type') == 'external'
                    if is_external:
                        ext_style = corridor.get('style', {})
                        corr_line_color = ext_style.get('color', 'gray')
                        corr_line_style = ext_style.get('lineType', '--')
                        corr_line_width = 10
                        corr_label = f"External Corridor ({building_name}-{corridor.get('name', f'corr{corr_idx}')})"
                    
                    elif 'name' in corridor and ('connectToBuilding' in corridor['name']):
                        corr_line_color = COLORS['connect_corridor']
                        corr_line_style = '-'
                        corr_line_width = 12
                        corr_label = f"Connecting Corridor ({building_name}-{level_name})"
                    
                    else:
                        corr_line_color = COLORS['corridor_line'].get(building_name, 'gray')
                        corr_line_style = '-'
                        corr_line_width = 8
                        corr_label = None
                    
                    if corr_label and corr_label not in ax.get_legend_handles_labels()[1]:
                        ax.plot(x, y, z_coords, 
                                color=corr_line_color, 
                                linestyle=corr_line_style,
                                linewidth=corr_line_width, 
                                alpha=0.8, 
                                label=corr_label)
                    else:
                        ax.plot(x, y, z_coords, 
                                color=corr_line_color, 
                                linestyle=corr_line_style,
                                linewidth=corr_line_width, 
                                alpha=0.8)
                    
                    for px, py, pz in points:
                        ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=40, marker='s', alpha=0.9)

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

            for stair in level['stairs']:
                stair_name = stair['name']
                is_path_stair = (building_name, stair_name, level_name) in path_stairs
                
                if show_all or show_level or is_path_stair:
                    x, y, _ = stair['coordinates']
                    stair_label = f"Building {building_name}-{stair_name}"
                    
                    stair_color = COLORS['stair'].get(stair_name, 'red')
                    
                    marker_size = 800 if is_path_stair else 600
                    marker_edge_width = 3 if is_path_stair else 1
                    
                    if stair_label not in ax.get_legend_handles_labels()[1]:
                        ax.scatter(x, y, z, color=stair_color, s=marker_size, marker='^', 
                                  label=stair_label, edgecolors='black', linewidths=marker_edge_width)
                    else:
                        ax.scatter(x, y, z, color=stair_color, s=marker_size, marker='^',
                                  edgecolors='black', linewidths=marker_edge_width)
                    
                    ax.text(x, y, z, stair_name, color=COLORS['stair_label'], fontweight='bold', fontsize=14)
        
        if level_count > 0 and len(displayed_levels) > 0:
            if building_name == 'B':
                label_y = max_displayed_y - 2.0
            elif building_name == 'Gate':
                label_y = max_displayed_y + 3.0
            else:
                label_y = max_displayed_y + 2.0
            label_z = max_displayed_z + 1.0
            center_x = corresponding_x
            
            building_label_positions[building_name] = (center_x, label_y, label_z)

    # 绘制Gate建筑（如果JSON中没有，添加默认的Gate显示）
    if 'buildingGate' not in school_data:
        # 默认Gate位置
        gate_x, gate_y, gate_z = 0, 0, 0
        building_label_positions['Gate'] = (gate_x, gate_y + 3.0, gate_z + 1.0)
        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", 
                        facecolor=COLORS['building']['Gate'], alpha=0.7)
        ax.text(
            gate_x, gate_y + 3.0, gate_z + 1.0, 
            f"Building Gate", 
            color=COLORS['building_label']['Gate'], 
            fontweight='bold', 
            fontsize=30,
            ha='center', 
            va='center', 
            bbox=bbox_props
        )

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

            ax.plot(x, y, z, color=COLORS['path'], linewidth=6, linestyle='-', marker='o', markersize=10, label='Navigation Path')
            ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=1000, marker='*', label='Start', edgecolors='black')
            ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=1000, marker='*', label='End', edgecolors='black')
            ax.text(x[0], y[0], z[0], f"Start\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=16)
            ax.text(x[-1], y[-1], z[-1], f"End\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=16)
        except Exception as e:
            st.warning(f"Path drawing warning: {str(e)}")

    ax.set_xlabel('X Coordinate', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=18, fontweight='bold')
    ax.set_zlabel('Floor Height (Z Value)', fontsize=18, fontweight='bold')
    ax.set_title('Campus 3D Navigation Map (A/B/C/Gate Building Navigation)', fontsize=24, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=2)

    return fig, ax

class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}

    def add_node(self, building_id, node_type, name, level, coordinates):
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

def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b)**2 for a, b in zip(coords1, coords2)))

def build_navigation_graph(school_data):
    graph = Graph()

    # 第一步：添加所有节点（包括默认的Gate节点）
    # 先添加JSON中的节点
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        building_data = school_data[building_id]
        
        for level in building_data['levels']:
            level_name = level['name']

            # 添加教室节点
            for classroom in level['classrooms']:
                class_name = classroom['name']
                graph.add_node(
                    building_id=building_id,
                    node_type='classroom',
                    name=class_name,
                    level=level_name,
                    coordinates=classroom['coordinates']
                )

            # 添加楼梯节点
            for stair in level['stairs']:
                stair_name = stair['name']
                graph.add_node(
                    building_id=building_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

            # 添加走廊节点
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

    # 添加默认的Gate节点（如果JSON中没有）
    if 'buildingGate' not in school_data:
        # 默认Gate位置（可以根据实际情况调整坐标）
        gate_coords = [0.0, 0.0, 0.0]
        # 添加Gate的Entrance教室节点
        graph.add_node(
            building_id='buildingGate',
            node_type='classroom',
            name='Entrance',
            level='level1',
            coordinates=gate_coords
        )
        # 添加Gate的默认走廊节点
        graph.add_node(
            building_id='buildingGate',
            node_type='corridor',
            name='main-p0',
            level='level1',
            coordinates=gate_coords
        )
        # 连接Gate的Entrance到走廊
        gate_entrance_id = graph.node_id_map.get(('buildingGate', 'classroom', 'Entrance', 'level1'))
        gate_corridor_id = graph.node_id_map.get(('buildingGate', 'corridor', 'main-p0', 'level1'))
        if gate_entrance_id and gate_corridor_id:
            graph.add_edge(gate_entrance_id, gate_corridor_id, 1.0)

    # 第二步：添加建筑内部的边
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        building_data = school_data[building_id]

        for level in building_data['levels']:
            level_name = level['name']
            
            # 获取当前楼层的所有走廊节点
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]

            # 连接同一条走廊的点
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

            # 连接相邻的走廊节点（距离<3.0）
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2)
                    
                    if distance < 3.0:
                        graph.add_edge(node1_id, node2_id, distance)

            # 连接教室到最近的走廊
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

            # 连接楼梯到最近的走廊
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

        # 添加建筑内部的连接（楼梯-走廊等）
        for connection in building_data.get('connections', []):
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

    # 第三步：添加建筑之间的连接
    # A-B-C 原有连接
    a_building_id = 'buildingA'
    b_building_id = 'buildingB'
    c_building_id = 'buildingC'
    gate_building_id = 'buildingGate'
    
    # A-B 连接
    ab_connect_level = 'level1'
    a_b_corr_name = 'connectToBuildingB-p1'
    a_b_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_b_corr_name, ab_connect_level))
    b_a_corr_name = 'connectToBuildingAAndC-p1'
    b_a_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_a_corr_name, ab_connect_level))
    
    if a_b_node_id and b_a_node_id:
        coords_a = graph.nodes[a_b_node_id]['coordinates']
        coords_b = graph.nodes[b_a_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_b)
        graph.add_edge(a_b_node_id, b_a_node_id, distance)
    else:
        st.warning("Could not find A-B level1 inter-building corridor connection nodes")
    
    # B-C 连接
    bc_connect_level = 'level1'
    b_c_corr_name = 'connectToBuildingAAndC-p0'
    b_c_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_c_corr_name, bc_connect_level))
    c_b_corr_name = 'connectToBuildingB-p1'
    c_b_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_b_corr_name, bc_connect_level))
    
    if b_c_node_id and c_b_node_id:
        coords_b = graph.nodes[b_c_node_id]['coordinates']
        coords_c = graph.nodes[c_b_node_id]['coordinates']
        distance = euclidean_distance(coords_b, coords_c)
        graph.add_edge(b_c_node_id, c_b_node_id, distance)
    else:
        st.warning("Could not find B-C level1 inter-building corridor connection nodes")
    
    # A-C 直接连接
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
    
    # --------------------------
    # Gate建筑连接逻辑 - 确保Gate能和其他建筑连通
    # --------------------------
    # 获取所有建筑level1的走廊节点
    level1_corridor_nodes = {}
    for building_id in [a_building_id, b_building_id, c_building_id, gate_building_id]:
        building_name = building_id.replace('building', '')
        corridor_nodes = [
            (node_id, graph.nodes[node_id]['coordinates']) 
            for node_id, node_info in graph.nodes.items()
            if node_info['building'] == building_name 
            and node_info['type'] == 'corridor' 
            and node_info['level'] == 'level1'
        ]
        level1_corridor_nodes[building_name] = corridor_nodes
    
    # 如果Gate有节点，自动连接到最近的建筑
    if 'Gate' in level1_corridor_nodes and level1_corridor_nodes['Gate']:
        gate_nodes = level1_corridor_nodes['Gate']
        
        # 连接到A楼（如果有节点）
        if 'A' in level1_corridor_nodes and level1_corridor_nodes['A']:
            a_nodes = level1_corridor_nodes['A']
            min_dist = float('inf')
            best_gate_node = None
            best_a_node = None
            
            for gate_node_id, gate_coords in gate_nodes:
                for a_node_id, a_coords in a_nodes:
                    dist = euclidean_distance(gate_coords, a_coords)
                    if dist < min_dist:
                        min_dist = dist
                        best_gate_node = gate_node_id
                        best_a_node = a_node_id
            
            if best_gate_node and best_a_node:
                graph.add_edge(best_gate_node, best_a_node, min_dist)
        
        # 连接到B楼（如果有节点）
        if 'B' in level1_corridor_nodes and level1_corridor_nodes['B']:
            b_nodes = level1_corridor_nodes['B']
            min_dist = float('inf')
            best_gate_node = None
            best_b_node = None
            
            for gate_node_id, gate_coords in gate_nodes:
                for b_node_id, b_coords in b_nodes:
                    dist = euclidean_distance(gate_coords, b_coords)
                    if dist < min_dist:
                        min_dist = dist
                        best_gate_node = gate_node_id
                        best_b_node = b_node_id
            
            if best_gate_node and best_b_node:
                graph.add_edge(best_gate_node, best_b_node, min_dist)
        
        # 连接到C楼（如果有节点）
        if 'C' in level1_corridor_nodes and level1_corridor_nodes['C']:
            c_nodes = level1_corridor_nodes['C']
            min_dist = float('inf')
            best_gate_node = None
            best_c_node = None
            
            for gate_node_id, gate_coords in gate_nodes:
                for c_node_id, c_coords in c_nodes:
                    dist = euclidean_distance(gate_coords, c_coords)
                    if dist < min_dist:
                        min_dist = dist
                        best_gate_node = gate_node_id
                        best_c_node = c_node_id
            
            if best_gate_node and best_c_node:
                graph.add_edge(best_gate_node, best_c_node, min_dist)

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
        return None, f"Invalid building selection, only Buildings A, B, C and Gate are supported", None, None
        
    try:
        # 获取起始节点
        start_key = (start_building, start_classroom, start_level)
        start_node = graph.node_id_map.get(start_key)
        
        # 如果没找到，尝试直接构造节点ID
        if not start_node:
            start_node = f"{start_building}-classroom-{start_classroom}@{start_level}"
        
        # 检查起始节点是否存在
        if start_node not in graph.nodes:
            return None, f"Starting classroom does not exist: {start_building}{start_classroom}@{start_level}", None, None
        
        # 获取目标节点
        end_key = (end_building, end_classroom, end_level)
        end_node = graph.node_id_map.get(end_key)
        
        # 如果没找到，尝试直接构造节点ID
        if not end_node:
            end_node = f"{end_building}-classroom-{end_classroom}@{end_level}"
        
        # 检查目标节点是否存在
        if end_node not in graph.nodes:
            return None, f"Destination classroom does not exist: {end_building}{end_classroom}@{end_level}", None, None

        # 执行最短路径计算
        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)

        if path:
            total_distance = distances[end_node]
            simplified_path = []
            path_stairs = set()
            prev_building = None
            
            for node_id in path:
                node_info = graph.nodes.get(node_id, {})
                node_type = node_info.get('type', '')
                node_name = node_info.get('name', '')
                node_level = node_info.get('level', '')
                node_building = node_info.get('building', '')
                
                if node_type == 'stair':
                    path_stairs.add((node_building, node_name, node_level))
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
                
                elif node_type == 'classroom':
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
                
                elif node_type == 'corridor':
                    if 'connectToBuilding' in node_name:
                        connected_building = 'Other'
                        if 'connectToBuildingA' in node_name:
                            connected_building = 'A'
                        elif 'connectToBuildingB' in node_name:
                            connected_building = 'B'
                        elif 'connectToBuildingC' in node_name:
                            connected_building = 'C'
                        elif 'connectToBuildingGate' in node_name:
                            connected_building = 'Gate'
                            
                        if prev_building and prev_building != node_building:
                            simplified_path.append(f"Cross corridor from Building {prev_building} to Building {node_building}({node_level})")
                
                if node_type in ['classroom', 'stair', 'corridor']:
                    prev_building = node_building
            
            full_path_str = " → ".join(simplified_path) if simplified_path else "Direct route"
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
        # 详细的错误信息，方便调试
        st.error(f"Navigation error: {str(e)}")
        return None, f"Navigation error: {str(e)}", None, None

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

        ax.plot(x, y, z, color=COLORS['path'], linewidth=10, linestyle='-', marker='o', markersize=10)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    except Exception as e:
        st.error(f"Failed to draw path: {str(e)}")

def get_classroom_info(school_data):
    """获取所有建筑的教室信息，确保Gate建筑被包含"""
    try:
        # 强制确保Gate出现在建筑列表中
        all_buildings = ['A', 'B', 'C', 'Gate']
        classrooms_by_building = {}
        levels_by_building = {}
        
        # 先从JSON加载数据
        for building_id in school_data.keys():
            if not building_id.startswith('building'):
                continue
            building_name = building_id.replace('building', '')
            if building_name not in all_buildings:
                continue
                
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
        
        # 确保所有建筑（包括Gate）都有默认值
        for building in all_buildings:
            if building not in levels_by_building:
                levels_by_building[building] = ['level1']
                classrooms_by_building[building] = {'level1': ['Entrance' if building == 'Gate' else 'Classroom1']}
        
        return all_buildings, levels_by_building, classrooms_by_building
    except Exception as e:
        st.error(f"Failed to retrieve classroom information: {str(e)}")
        # 返回默认值
        return ['A', 'B', 'C', 'Gate'], \
        {
            'A': ['level1'], 
            'B': ['level1'], 
            'C': ['level1'], 
            'Gate': ['level1']
        }, \
        {
            'A': {'level1': ['Classroom1']}, 
            'B': {'level1': ['Classroom1']}, 
            'C': {'level1': ['Classroom1']}, 
            'Gate': {'level1': ['Entrance']}
        }

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

def welcome_page():
    # 初始化Google Sheets连接
    if 'worksheet' not in st.session_state:
        st.session_state['worksheet'] = init_google_sheet()
    
    # 获取当前总访问次数
    total_accesses = get_total_accesses(st.session_state['worksheet'])
    
    # 设置欢迎页面样式
    st.markdown("""
        <style>
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding-top: 20px; 
            margin-top: -500px;
            padding: 0;
            text-align: center;
        }
        .welcome-title {
            font-size: 3rem;
            margin-bottom: 2rem;
            margin-top: 0;
            color: #2c3e50;
        }
        .enter-button {
            font-size: 2rem;
            padding: 0.8rem 2rem;
            width: 200px;
            font-weight: bold;
        }
        .access-count {
            margin-top: 1rem;
            font-size: 0.8rem;
            color: #666;
            text-align: center;
            width: 100%;
        }
        .main > div:first-child {
            height: 100vh;
            overflow: hidden;
            padding-top: 0 !important; 
        }
        body {
            margin: 0 !important;
            padding: 0 !important;
            height: 100% !important;
            overflow: hidden !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 欢迎页面内容
    st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="welcome-title">Welcome to SCIS Navigation System</h1>', unsafe_allow_html=True)
    
    # 创建包含按钮和访问次数的列
    col = st.columns(1)
    with col[0]:
        if st.button('Enter Navigation System', key='enter_btn', use_container_width=True, 
                    help='Click to enter the 3D navigation system'):
            # 更新访问次数
            update_access_count(st.session_state['worksheet'])
            st.session_state['page'] = 'navigation'
            st.rerun()
        
        st.markdown(f'<p class="access-count">Total visits: {total_accesses}</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def navigation_page():
    # 初始化状态
    if 'display_options' not in st.session_state:
        reset_app_state()
    
    # 加载学校数据
    school_data = load_school_data_detailed('school_data_detailed.json')
    if not school_data:
        st.error("Failed to load school data file!")
        return
    
    # 获取教室信息（强制包含Gate）
    building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
    
    # 构建导航图
    global graph
    graph = build_navigation_graph(school_data)
    
    # 页面布局
    st.title("SCIS Campus 3D Navigation System")
    
    # 重置按钮
    if st.button("Reset View", key='reset_btn'):
        reset_app_state()
        st.rerun()
    
    # 导航控制面板
    st.sidebar.header("Navigation Controls")
    
    # 起点选择
    st.sidebar.subheader("Start Location")
    start_building = st.sidebar.selectbox("Building", building_names, key='start_building')
    start_level = ""
    start_classroom = ""
    if start_building in levels_by_building:
        start_level = st.sidebar.selectbox("Level", levels_by_building[start_building], key='start_level')
        if start_level in classrooms_by_building[start_building]:
            start_classroom = st.sidebar.selectbox("Classroom", classrooms_by_building[start_building][start_level], key='start_classroom')
    
    # 终点选择
    st.sidebar.subheader("End Location")
    end_building = st.sidebar.selectbox("Building", building_names, key='end_building')
    end_level = ""
    end_classroom = ""
    if end_building in levels_by_building:
        end_level = st.sidebar.selectbox("Level", levels_by_building[end_building], key='end_level')
        if end_level in classrooms_by_building[end_building]:
            end_classroom = st.sidebar.selectbox("Classroom", classrooms_by_building[end_building][end_level], key='end_classroom')
    
    # 导航按钮
    if st.sidebar.button("Navigate", key='navigate_btn'):
        # 检查是否选择了完整的起点和终点
        if not (start_building and start_level and start_classroom and end_building and end_level and end_classroom):
            st.warning("Please select complete start and end locations!")
        else:
            # 执行导航
            path, message, full_path_str, display_options = navigate(
                graph, start_building, start_classroom, start_level,
                end_building, end_classroom, end_level
            )
            
            # 处理返回结果，避免空值错误
            if path is not None and message is not None and full_path_str is not None and display_options is not None:
                st.session_state['current_path'] = path
                st.session_state['display_options'] = display_options
                st.session_state['path_result'] = (message, full_path_str)
            else:
                st.error(message if message else "Navigation failed!")
    
    # 显示导航结果
    if 'path_result' in st.session_state:
        message, full_path_str = st.session_state['path_result']
        st.subheader("Navigation Result")
        st.success(message)
        # 确保full_path_str不是None
        if full_path_str:
            st.write("Route: " + full_path_str)
        else:
            st.write("Route: No route information available")
    
    # 绘制3D地图
    fig, ax = plot_3d_map(school_data, st.session_state['display_options'])
    
    # 显示地图
    st.pyplot(fig)
    
    # 返回欢迎页按钮
    if st.button("Back to Welcome Page", key='back_btn'):
        st.session_state['page'] = 'welcome'
        st.rerun()

# 主程序
def main():
    # 初始化会话状态
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'
        reset_app_state()
    
    # 根据当前页面显示不同内容
    if st.session_state['page'] == 'welcome':
        welcome_page()
    elif st.session_state['page'] == 'navigation':
        navigation_page()

if __name__ == "__main__":
    main()
