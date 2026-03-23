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
# 地图与导航核心逻辑（保持不变，新增gate相关配色）
# --------------------------
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral', 'Gate': 'gold'},  # 新增Gate配色
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},
    'corridor_line': {'A': 'cyan', 'B': 'forestgreen', 'C': 'salmon', 'Gate': 'darkgoldenrod'},  # 新增Gate走廊配色
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
        'GateStairs': '#FFD700'  # 新增大门楼梯配色
    },
    'stair_label': 'darkred',
    'classroom_label': 'black',
    'path': 'red',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold',
    'building_label': {'A': 'darkblue', 'B': 'darkgreen', 'C': 'darkred', 'Gate': 'darkgoldenrod'}  # 新增Gate标签配色
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

    # 遍历所有建筑（包括gate）
    for building_id in school_data.keys():
        # 处理gate建筑（单独命名为Gate）
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
            z = level['z']
            
            # 调整显示逻辑，支持Gate建筑
            show_level = show_all
            if not show_all:
                if building_name == 'B':
                    show_level = any((building_name, s_name, level_name) in path_stairs for s_name in ['StairsB1', 'StairsB2'])
                    if (start_building == 'B' or end_building == 'B') or (start_building in ['A','C'] and end_building in ['A','C'] and 'B' in [start_building, end_building]):
                        show_level = show_level or (level_name == 'level1')
                elif building_name == 'Gate':
                    # Gate只有level1，只要起点/终点是Gate就显示
                    show_level = (start_building == 'Gate' or end_building == 'Gate') or any(('Gate', s_name, level_name) in path_stairs for s_name in ['GateStairs'])
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
                label_y = max_displayed_y + 5.0  # Gate标签位置调整
            else:
                label_y = max_displayed_y + 2.0
            label_z = max_displayed_z + 1.0
            center_x = corresponding_x
            
            building_label_positions[building_name] = (center_x, label_y, label_z)

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
    ax.set_title('Campus 3D Navigation Map (A/B/C Building + Gate Navigation)', fontsize=24, fontweight='bold', pad=20)
    
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=2)

    return fig, ax

class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}

    def add_node(self, building_id, node_type, name, level, coordinates):
        # 处理gate建筑的命名
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
            # Gate的classroom特殊处理
            class_key = (building_name, name, level)
            self.node_id_map[class_key] = node_id
            
        return node_id

    # 核心方法：添加节点间的边（双向）
    def add_edge(self, node1_id, node2_id, weight):
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

# 核心修改1：新增楼层差惩罚的距离计算函数
def euclidean_distance(coords1, coords2, floor_penalty=15.0):
    """
    计算欧式距离，新增楼层差惩罚（鼓励少上下楼）
    floor_penalty: 每跨一层楼的惩罚值（越大越不想爬楼梯）
    """
    # 基础欧式距离（x,y,z）
    base_dist = np.sqrt(sum((a - b)**2 for a, b in zip(coords1, coords2)))
    
    # 提取z坐标（楼层高度），计算楼层差
    z1, z2 = coords1[2], coords2[2]
    floor_diff = abs(z1 - z2)
    
    # 楼层差惩罚：每跨1层楼，额外加 floor_penalty 的成本
    penalty = floor_diff * floor_penalty
    
    # 总距离 = 基础距离 + 楼层惩罚
    total_dist = base_dist + penalty
    return total_dist

# 核心新增：计算两个节点之间的方向词
def get_direction_between_nodes(graph, current_node_id, next_node_id):
    """
    计算当前节点到下一个节点的方向词
    规则：
    1. 楼梯之间：仅用 往上/往下
    2. 其他节点：基于x/y坐标（x+→右，x-→左，y+→前，y-→后）
    """
    # 获取节点信息
    current_node = graph.nodes[current_node_id]
    next_node = graph.nodes[next_node_id]
    
    # 提取坐标
    curr_x, curr_y, curr_z = current_node['coordinates']
    next_x, next_y, next_z = next_node['coordinates']
    
    # 判断是否是楼梯节点
    curr_is_stair = current_node['type'] == 'stair'
    next_is_stair = next_node['type'] == 'stair'
    
    # 楼梯之间：只判断上下
    if curr_is_stair and next_is_stair:
        if next_z > curr_z:
            return "往上"
        elif next_z < curr_z:
            return "往下"
        else:
            return ""  # 同层楼梯，无方向
    
    # 非楼梯节点：判断平面方向
    x_diff = next_x - curr_x
    y_diff = next_y - curr_y
    
    # 阈值：避免微小坐标差导致方向错误
    threshold = 0.1
    
    if abs(x_diff) > threshold or abs(y_diff) > threshold:
        # 优先判断y轴（前后）
        if y_diff > threshold:
            return "向前"
        elif y_diff < -threshold:
            return "向后"
        # 再判断x轴（左右）
        elif x_diff > threshold:
            return "向右"
        elif x_diff < -threshold:
            return "向左"
    
    # 无明显位移
    return ""

def build_navigation_graph(school_data):
    graph = Graph()

    # 遍历所有建筑（包括gate）
    for building_id in school_data.keys():
        # 跳过非建筑节点（如果有）
        if not (building_id.startswith('building') or building_id == 'gate'):
            continue
            
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

    # 构建边（连接关系）
    for building_id in school_data.keys():
        if not (building_id.startswith('building') or building_id == 'gate'):
            continue
            
        # 处理建筑名称
        if building_id == 'gate':
            building_name = 'Gate'
        else:
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
                        # 同走廊无楼层惩罚
                        distance = euclidean_distance(coords1, coords2, floor_penalty=0)
                        graph.add_edge(current_node_id, next_node_id, distance)

            # 连接近距离的走廊节点
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    # 同层走廊无惩罚
                    distance = euclidean_distance(coords1, coords2, floor_penalty=0)
                    
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
                    # 同层教室到走廊无惩罚
                    dist = euclidean_distance(class_coords, corr_coords, floor_penalty=0)
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
                    # 修改1：楼梯到走廊用基础距离（同层无惩罚）
                    dist = euclidean_distance(stair_coords, corr_coords, floor_penalty=0)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(stair_node_id, nearest_corr_node_id, min_dist)

        # 修改2：同一楼梯不同楼层连接时，增加楼层惩罚
        # 新增：连接同一楼梯的不同楼层（核心！少上下楼的关键）
        stair_names = set()
        # 先收集所有楼梯名称
        for node_id, node_info in graph.nodes.items():
            if node_info['type'] == 'stair':
                stair_names.add((node_info['building'], node_info['name']))

        # 对每个楼梯，连接不同楼层的节点（加惩罚）
        for (building, stair_name) in stair_names:
            # 找到该楼梯所有楼层的节点
            stair_level_nodes = []
            for node_id, node_info in graph.nodes.items():
                if (node_info['building'] == building and 
                    node_info['type'] == 'stair' and 
                    node_info['name'] == stair_name):
                    stair_level_nodes.append((node_id, node_info['coordinates'], node_info['level']))
            
            # 按楼层排序，依次连接（加惩罚）
            stair_level_nodes.sort(key=lambda x: x[1][2])  # 按z坐标排序
            for i in range(len(stair_level_nodes)-1):
                node1_id, coords1, _ = stair_level_nodes[i]
                node2_id, coords2, _ = stair_level_nodes[i+1]
                
                # 楼梯跨层连接：加楼层惩罚（每跨1层加15）
                dist = euclidean_distance(coords1, coords2, floor_penalty=15.0)
                graph.add_edge(node1_id, node2_id, dist)

        # 处理建筑内部/跨建筑连接（支持 Gate ↔ A/B/C 自动连接）
        for connection in building_data['connections']:
            from_obj_name, from_level = connection['from']
            to_obj_name, to_level = connection['to']
            
            # ------------------------------
            # 第一步：确定「来源节点」的信息（当前建筑）
            # ------------------------------
            # 判断来源节点类型（走廊/楼梯/教室）
            if from_obj_name.startswith(('Stairs', 'GateStairs')):
                from_obj_type = 'stair'
            # 检查是否是当前建筑的教室（比如 Gate 的 Main Gate）
            elif any(
                from_obj_name == cls['name'] 
                for level in building_data['levels'] 
                for cls in level.get('classrooms', [])
            ):
                from_obj_type = 'classroom'
            else:
                from_obj_type = 'corridor'  # 默认是走廊
            
            # 走廊节点要加 -p0 后缀（和建节点时一致）
            from_node_name = f"{from_obj_name}-p0" if from_obj_type == 'corridor' else from_obj_name
            from_node_id = graph.node_id_map.get((building_id, from_obj_type, from_node_name, from_level))

            # ------------------------------
            # 第二步：确定「目标节点」的信息（可能是跨建筑）
            # ------------------------------
            # 先尝试：目标节点是否在当前建筑？（内部连接）
            to_building_id = building_id
            # 检查目标节点是否属于其他建筑（核心：自动识别跨建筑）
            target_building_map = {
                # 关键词 → 目标建筑ID
                'ENTRANCE': 'buildingA',          # 指向A楼
                'connectToBuildingAAndC': 'buildingB',  # 指向B楼
                'SCHOOL CLINIC': 'buildingC',     # 指向C楼
                'connectToBuildingB': 'buildingB',
                'connectToBuildingC': 'buildingC'
            }
            # 匹配目标节点对应的建筑
            for keyword, target_building in target_building_map.items():
                if keyword in to_obj_name:
                    to_building_id = target_building
                    break
            
            # 判断目标节点类型
            if to_obj_name.startswith(('Stairs', 'GateStairs')):
                to_obj_type = 'stair'
            # 检查是否是目标建筑的教室（比如 C 楼的 SCHOOL CLINIC）
            elif to_building_id in school_data:  # 目标建筑存在
                to_obj_type = 'classroom' if any(
                    to_obj_name == cls['name']
                    for level in school_data[to_building_id]['levels']
                    for cls in level.get('classrooms', [])
                ) else 'corridor'
            else:
                to_obj_type = 'corridor'
            
            # 走廊节点加 -p0 后缀
            to_node_name = f"{to_obj_name}-p0" if to_obj_type == 'corridor' else to_obj_name
            to_node_id = graph.node_id_map.get((to_building_id, to_obj_type, to_node_name, to_level))

            # ------------------------------
            # 第三步：建立连接（自动计算距离）
            # ------------------------------
            if from_node_id and to_node_id:
                # 跨楼连接：无楼层惩罚（鼓励同层跨楼）
                from_coords = graph.nodes[from_node_id]['coordinates']
                to_coords = graph.nodes[to_node_id]['coordinates']
                # 判断是否跨楼：跨楼则无惩罚，内部则有惩罚
                if building_id != to_building_id:
                    distance = euclidean_distance(from_coords, to_coords, floor_penalty=0)
                else:
                    distance = euclidean_distance(from_coords, to_coords, floor_penalty=15.0)
                graph.add_edge(from_node_id, to_node_id, distance)
            else:
                # 调试日志（方便排查）
                st.warning(
                    f"连接失败：{building_id} → {to_building_id}\n"
                    f"来源节点：({building_id}, {from_obj_type}, {from_node_name}, {from_level}) → {from_node_id}\n"
                    f"目标节点：({to_building_id}, {to_obj_type}, {to_node_name}, {to_level}) → {to_node_id}"
                )

    # A/B/C楼之间的原有连接（保持不变，跨楼连廊无楼层惩罚）
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
        # 跨楼连廊：无楼层惩罚（鼓励同层跨楼）
        distance = euclidean_distance(coords_a, coords_b, floor_penalty=0)
        graph.add_edge(a_b_node_id, b_a_node_id, distance)
    else:
        st.warning("Could not find A-B level1 inter-building corridor connection nodes")
    
    bc_connect_level = 'level1'
    b_c_corr_name = 'connectToBuildingAAndC-p0'
    b_c_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_c_corr_name, bc_connect_level))
    c_b_corr_name = 'connectToBuildingB-p1'
    c_b_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_b_corr_name, bc_connect_level))
    
    if b_c_node_id and c_b_node_id:
        coords_b = graph.nodes[b_c_node_id]['coordinates']
        coords_c = graph.nodes[c_b_node_id]['coordinates']
        # 跨楼连廊：无楼层惩罚
        distance = euclidean_distance(coords_b, coords_c, floor_penalty=0)
        graph.add_edge(b_c_node_id, c_b_node_id, distance)
    else:
        st.warning("Could not find B-C level1 inter-building corridor connection nodes")
    
    connect_level1 = 'level1'
    a_corr1_name = 'connectToBuildingC-p3'
    a_connect1_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr1_name, connect_level1))
    c_corr1_name = 'connectToBuildingA-p0'
    c_connect1_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr1_name, connect_level1))
    
    if a_connect1_node_id and c_connect1_node_id:
        coords_a = graph.nodes[a_connect1_node_id]['coordinates']
        coords_c = graph.nodes[c_connect1_node_id]['coordinates']
        # 跨楼连廊：无楼层惩罚
        distance = euclidean_distance(coords_a, coords_c, floor_penalty=0)
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
        # 跨楼连廊：无楼层惩罚（优先3楼跨楼）
        distance = euclidean_distance(coords_a, coords_c, floor_penalty=0)
        graph.add_edge(a_connect3_node_id, c_connect3_node_id, distance)
    else:
        st.warning("Could not find level 3 A-C inter-building corridor connection nodes")

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
    # 新增Gate到有效建筑列表
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
            
            # 遍历路径，添加方向词
            for i in range(len(path)):
                node_id = path[i]
                node_info = graph.nodes[node_id]
                node_type = node_info['type']
                node_name = node_info['name']
                node_level = node_info['level']
                node_building = node_info['building']
                
                # 基础节点描述
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
                    # 计算当前节点到下一个节点的方向
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
    try:
        # 包含gate建筑
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
        if st.button('Enter System', key='enter_btn', use_container_width=True):
            # 更新访问次数
            update_access_count(st.session_state['worksheet'])
            st.session_state['page'] = 'main'
            st.rerun()
    
        st.markdown(f'<div class="access-count">Total Accesses: {total_accesses}</div>', unsafe_allow_html=True)
    st.image("welcome_image.jpg", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def main_interface():
    st.markdown("""
        <style>
        .stApp {
                padding-top: 0.0rem !important; 
            }
             header {
                display: none !important;
                }
            body {
                position: relative;
                min-height: 100vh;
                margin: 0;
                padding: 0;
            }
            .block-container {
                padding-top: 0.2rem !important;
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
        /* 新增：侧边栏展开/收起按钮样式 */
        .sidebar-toggle-btn {
            position: absolute;
            left: 10px;
            top: 100px;
            z-index: 999;
            width: 40px;
            height: 40px;
            border-radius: 50%;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="author-tag">Created By DANIEL HAN</div>', unsafe_allow_html=True)
    st.subheader("🏫SCIS Campus Navigation System")
    st.markdown("3D Map & Inter-building Path Planning (A/B/C Building + Gate Navigation)")

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
    # 新增：初始化侧边栏展开状态
    if 'sidebar_expanded' not in st.session_state:
        st.session_state['sidebar_expanded'] = True

    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        global graph
        graph = build_navigation_graph(school_data)
        building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
        st.success("✅ Campus data loaded successfully! Initial state shows A/B/C buildings and Gate")
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return

    # 新增：侧边栏展开/收起按钮
    toggle_col, _ = st.columns([0.05, 0.95])
    with toggle_col:
        if st.button(
            "◀️" if st.session_state['sidebar_expanded'] else "▶️",
            key="sidebar_toggle",
            help="Hide/Show location selector",
            use_container_width=True
        ):
            st.session_state['sidebar_expanded'] = not st.session_state['sidebar_expanded']
            st.rerun()

    # 修改：根据展开状态设置列宽度
    sidebar_width = 1 if st.session_state['sidebar_expanded'] else 0.01
    map_width = 6 if st.session_state['sidebar_expanded'] else 6.99
    col1, col2 = st.columns([sidebar_width, map_width])

    with col1:
        # 修改：仅在展开状态显示选择器
        if st.session_state['sidebar_expanded']:
            with st.expander("📍 Select Locations", expanded=True):
                st.markdown("#### Start Point")
                start_building = st.selectbox("Building", building_names, key="start_building")
                start_levels = levels_by_building.get(start_building, [])
                start_level = st.selectbox("Floor", start_levels, key="start_level")
                start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
                start_classroom = st.selectbox("Classroom", start_classrooms, key="start_classroom")
        
                st.markdown("#### End Point")
                end_building = st.selectbox("Building", building_names, key="end_building")
                end_levels = levels_by_building.get(end_building, [])
                end_level = st.selectbox("Floor", end_levels, key="end_level")
                end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
                end_classroom = st.selectbox("Classroom", end_classrooms, key="end_classroom")
        
                nav_button = st.button("🔍 Find Shortest Path", use_container_width=True)
                
                reset_button = st.button(
                    "🔄 Reset View", 
                    use_container_width=True,
                    help="Click to return to initial state, showing all floors (including Building B and Gate) and clearing path"
                )
                
                exit_button = st.button(
                    "🚪 Exit to Welcome Page", 
                    use_container_width=True,
                    help="Click to return to the welcome page",
                    type="secondary"
                )
                
                if reset_button:
                    reset_app_state()
                    st.rerun()
                
                if exit_button:
                    reset_app_state()
                    st.session_state['page'] = 'welcome'
                    st.rerun()

    with col2:
        st.markdown("#### 🗺️ 3D Campus Map")
        
        # 修改：仅在展开状态且点击导航按钮时执行导航逻辑
        if st.session_state['sidebar_expanded'] and 'nav_button' in locals() and nav_button:
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
                    
                    st.session_state['current_path'] = path
                    st.session_state['display_options'] = display_options
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"Navigation process error: {str(e)}")
        
        try:
            if st.session_state['current_path'] is not None:
                fig, ax = plot_3d_map(school_data, st.session_state['display_options'])
                plot_path(ax, graph, st.session_state['current_path'])
            else:
                fig, ax = plot_3d_map(school_data)
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Failed to display map: {str(e)}")

def main():
    # 初始化会话状态，控制显示哪个页面
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'
    
    # 根据当前页面状态显示不同内容
    if st.session_state['page'] == 'welcome':
        welcome_page()
    else:
        main_interface()

if __name__ == "__main__":
    main()
