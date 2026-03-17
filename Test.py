import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
import heapq

# 解决matplotlib后端问题
plt.switch_backend('Agg')

# --------------------------
# 全局配置
# --------------------------
st.set_page_config(page_title="SCIS Navigation System", layout="wide")

# 颜色配置
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral', 'Gate': 'gold'},
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple', 17: 'pink'},
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
        'GateStairs': '#8B4513'
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

# 全局图实例
graph = None
school_data = None

# --------------------------
# Google Sheets 降级配置（避免启动失败）
# --------------------------
SHEET_NAME = 'Navigation visitors'
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

def get_credentials():
    """获取凭据（添加降级逻辑）"""
    try:
        # 优先从Streamlit Secrets加载
        service_account_info = st.secrets.get("google_service_account", {})
        if service_account_info:
            return Credentials.from_service_account_info(service_account_info, scopes=SCOPE)
        # 本地运行时加载本地密钥文件（可选）
        elif os.path.exists("service_account.json"):
            return Credentials.from_service_account_file("service_account.json", scopes=SCOPE)
        else:
            return None
    except Exception as e:
        st.warning(f"Google Sheets 凭据加载失败: {str(e)}")
        return None

def init_google_sheet():
    """初始化表格（完全降级）"""
    try:
        creds = get_credentials()
        if not creds:
            return None
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME)
        # 检查工作表
        try:
            stats_worksheet = sheet.worksheet("Access_Stats")
        except:
            stats_worksheet = sheet.add_worksheet(title="Access_Stats", rows="1000", cols="3")
            stats_worksheet.append_row(["Timestamp", "Access_Count", "Total_Accesses"])
            stats_worksheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, 1])
        return stats_worksheet
    except Exception as e:
        st.info(f"Google Sheets 不可用（非核心功能）: {str(e)}")
        return None

def update_access_count(worksheet):
    """更新访问次数（降级处理）"""
    if not worksheet:
        return 0
    try:
        records = worksheet.get_all_values()
        total = int(records[-1][2]) if len(records)>=2 and records[-1][2].isdigit() else 0
        new_total = total + 1
        worksheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, new_total])
        return new_total
    except:
        return 0

# --------------------------
# 核心图结构与导航算法
# --------------------------
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}

    def add_node(self, building_id, node_type, name, level, coordinates):
        """添加节点（兼容Gate建筑）"""
        building_name = 'Gate' if building_id == 'gate' else building_id.replace('building', '')
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
        
        # 多键映射
        self.node_id_map[(building_id, node_type, name, level)] = node_id
        if node_type == 'classroom':
            self.node_id_map[(building_name, name, level)] = node_id
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        """添加边（双向）"""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

def euclidean_distance(coords1, coords2):
    """欧式距离计算"""
    return np.sqrt(sum((a - b)**2 for a, b in zip(coords1, coords2)))

def dijkstra(graph, start_node_id, end_node_id):
    """Dijkstra最短路径算法（核心导航）"""
    # 初始化距离和前驱
    distances = {node: float('inf') for node in graph.nodes}
    predecessors = {node: None for node in graph.nodes}
    distances[start_node_id] = 0
    priority_queue = [(0, start_node_id)]
    
    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)
        # 找到终点提前退出
        if current_node == end_node_id:
            break
        # 跳过已处理的节点
        if current_dist > distances[current_node]:
            continue
        # 遍历邻居
        for neighbor, weight in graph.nodes[current_node]['neighbors'].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # 重构路径
    path = []
    current = end_node_id
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    # 路径有效性检查
    if len(path) == 1 and path[0] != start_node_id:
        return []
    return path

def get_real_world_direction(graph, current_node_id, next_node_id, prev_node_id=None):
    """补全方向计算逻辑（修复截断）"""
    current_node = graph.nodes[current_node_id]
    next_node = graph.nodes[next_node_id]
    
    # 提取平面坐标
    curr_x, curr_y, _ = current_node['coordinates']
    next_x, next_y, _ = next_node['coordinates']
    
    # 默认方向
    direction = ""
    # 简化方向判断（核心功能优先）
    if next_x > curr_x + 1:
        direction = "向右"
    elif next_x < curr_x - 1:
        direction = "向左"
    elif next_y > curr_y + 1:
        direction = "向前"
    elif next_y < curr_y - 1:
        direction = "向后"
    return direction

# --------------------------
# 3D绘图（修复graph未定义问题）
# --------------------------
def plot_3d_map(school_data, graph, display_options=None):
    """绘制3D地图（传入graph参数）"""
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

    # 遍历所有建筑（包括Gate）
    for building_id in school_data.keys():
        if building_id == 'gate' or building_id.startswith('building'):
            building_name = 'Gate' if building_id == 'gate' else building_id.replace('building', '')
            building_data = school_data[building_id]
            
            displayed_levels = []
            max_displayed_z = -float('inf')
            max_displayed_y = -float('inf')
            corresponding_x = 0
            level_count = 0
            
            for level in building_data['levels']:
                level_name = level['name']
                z = level['z']
                
                # 控制显示层级
                show_level = show_all
                if not show_all:
                    if building_name == 'Gate':
                        show_level = (level_name == start_level) or (level_name == end_level)
                    elif building_name == 'B':
                        show_level = any((building_name, s_name, level_name) in path_stairs for s_name in ['StairsB1', 'StairsB2'])
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
                
                # 绘制楼层平面
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

                    # 绘制走廊
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
                        elif 'name' in corridor and ('connectToBuilding' in corridor['name']):
                            corr_line_color = COLORS['connect_corridor']
                            corr_line_style = '-'
                            corr_line_width = 12
                        else:
                            corr_line_color = COLORS['corridor_line'].get(building_name, 'gray')
                            corr_line_style = '-'
                            corr_line_width = 8
                        
                        ax.plot(x, y, z_coords, 
                                color=corr_line_color, 
                                linestyle=corr_line_style,
                                linewidth=corr_line_width, 
                                alpha=0.8)
                        
                        # 走廊节点
                        for px, py, pz in points:
                            ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=40, marker='s', alpha=0.9)

                    # 绘制教室
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

                # 绘制楼梯
                for stair in level['stairs']:
                    stair_name = stair['name']
                    is_path_stair = (building_name, stair_name, level_name) in path_stairs
                    if show_all or show_level or is_path_stair:
                        x, y, _ = stair['coordinates']
                        stair_color = COLORS['stair'].get(stair_name, 'red')
                        marker_size = 800 if is_path_stair else 600
                        ax.scatter(x, y, z, color=stair_color, s=marker_size, marker='^',
                                  edgecolors='black', linewidths=3 if is_path_stair else 1)
                        ax.text(x, y, z, stair_name, color=COLORS['stair_label'], fontweight='bold', fontsize=14)
        
        # 建筑标签
        if level_count > 0 and len(displayed_levels) > 0:
            label_y = max_displayed_y + (3.0 if building_name == 'Gate' else 2.0)
            label_z = max_displayed_z + 1.0
            center_x = corresponding_x
            building_label_positions[building_name] = (center_x, label_y, label_z)

    # 添加建筑名称标签
    for building_name, (x, y, z) in building_label_positions.items():
        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", 
                        facecolor=COLORS['building'].get(building_name, 'lightgray'), alpha=0.7)
        ax.text(
            x, y, z, 
            f"Building {building_name}", 
            color=COLORS['building_label'].get(building_name, 'black'), 
            fontweight='bold', 
            fontsize=30,
            ha='center', va='center', bbox=bbox_props
        )

    # 绘制导航路径
    if path and not show_all and graph:
        try:
            x = [graph.nodes[node]['coordinates'][0] for node in path]
            y = [graph.nodes[node]['coordinates'][1] for node in path]
            z = [graph.nodes[node]['coordinates'][2] for node in path]
            
            ax.plot(x, y, z, color=COLORS['path'], linewidth=6, linestyle='-', marker='o', markersize=10, label='Navigation Path')
            ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=1000, marker='*', label='Start', edgecolors='black')
            ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=1000, marker='*', label='End', edgecolors='black')
            ax.text(x[0], y[0], z[0], "Start", color=COLORS['start_label'], fontweight='bold', fontsize=16)
            ax.text(x[-1], y[-1], z[-1], "End", color=COLORS['end_label'], fontweight='bold', fontsize=16)
        except Exception as e:
            st.warning(f"路径绘制失败: {str(e)}")

    # 图表配置
    ax.set_xlabel('X Coordinate', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=18, fontweight='bold')
    ax.set_zlabel('Floor Height (Z Value)', fontsize=18, fontweight='bold')
    ax.set_title('Campus 3D Navigation Map (A/B/C/Gate Building)', fontsize=24, fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=2)

    return fig, ax

# --------------------------
# 数据加载与图构建（核心修复：Gate-C直连）
# --------------------------
def load_school_data_detailed(filename):
    """加载JSON数据"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载JSON失败: {str(e)}")
        return None

def build_graph(school_data):
    """构建导航图（修复Gate-C直连）"""
    g = Graph()
    
    # 1. 添加所有节点（教室、走廊、楼梯）
    for building_id in school_data.keys():
        building_data = school_data[building_id]
        for level in building_data['levels']:
            level_name = level['name']
            # 添加教室节点
            for classroom in level['classrooms']:
                g.add_node(
                    building_id,
                    'classroom',
                    classroom['name'],
                    level_name,
                    classroom['coordinates']
                )
            # 添加走廊节点
            for corridor in level['corridors']:
                corr_name = corridor.get('name', f"corr_{len(level['corridors'])}")
                # 走廊每个点作为节点
                for idx, point in enumerate(corridor['points']):
                    g.add_node(
                        building_id,
                        'corridor',
                        f"{corr_name}_p{idx}",
                        level_name,
                        point
                    )
            # 添加楼梯节点
            for stair in level['stairs']:
                g.add_node(
                    building_id,
                    'stair',
                    stair['name'],
                    level_name,
                    stair['coordinates']
                )
    
    # 2. 添加边（内部连接）
    for building_id in school_data.keys():
        building_data = school_data[building_id]
        for level in building_data['levels']:
            level_name = level['name']
            # 走廊内部点连接
            for corridor in level['corridors']:
                corr_name = corridor.get('name', f"corr_{len(level['corridors'])}")
                points = corridor['points']
                for i in range(len(points)-1):
                    node1_id = g.node_id_map[(building_id, 'corridor', f"{corr_name}_p{i}", level_name)]
                    node2_id = g.node_id_map[(building_id, 'corridor', f"{corr_name}_p{i+1}", level_name)]
                    weight = euclidean_distance(points[i], points[i+1])
                    g.add_edge(node1_id, node2_id, weight)
            # 教室与最近走廊连接
            for classroom in level['classrooms']:
                class_node_id = g.node_id_map[(building_id.replace('building','') if building_id!='gate' else 'Gate', 
                                              'classroom', classroom['name'], level_name)]
                class_coords = classroom['coordinates']
                # 找最近的走廊节点
                min_dist = float('inf')
                nearest_corr_node = None
                for corridor in level['corridors']:
                    corr_name = corridor.get('name', f"corr_{len(level['corridors'])}")
                    for idx, point in enumerate(corridor['points']):
                        corr_node_id = g.node_id_map[(building_id, 'corridor', f"{corr_name}_p{idx}", level_name)]
                        dist = euclidean_distance(class_coords, point)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_corr_node = corr_node_id
                if nearest_corr_node:
                    g.add_edge(class_node_id, nearest_corr_node, min_dist)
            # 楼梯与最近走廊连接
            for stair in level['stairs']:
                stair_node_id = g.node_id_map[(building_id, 'stair', stair['name'], level_name)]
                stair_coords = stair['coordinates']
                min_dist = float('inf')
                nearest_corr_node = None
                for corridor in level['corridors']:
                    corr_name = corridor.get('name', f"corr_{len(level['corridors'])}")
                    for idx, point in enumerate(corridor['points']):
                        corr_node_id = g.node_id_map[(building_id, 'corridor', f"{corr_name}_p{idx}", level_name)]
                        dist = euclidean_distance(stair_coords, point)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_corr_node = corr_node_id
                if nearest_corr_node:
                    g.add_edge(stair_node_id, nearest_corr_node, min_dist)
    
    # 3. 跨楼层楼梯连接
    for building_id in school_data.keys():
        building_data = school_data[building_id]
        for connection in building_data.get('connections', []):
            from_stair, from_level = connection['from']
            to_stair, to_level = connection['to']
            # 找楼梯节点
            try:
                from_node_id = g.node_id_map[(building_id, 'stair', from_stair, from_level)]
                to_node_id = g.node_id_map[(building_id, 'stair', to_stair, to_level)]
                g.add_edge(from_node_id, to_node_id, 5.0)  # 楼梯权重固定
            except KeyError:
                continue
    
    # 4. 核心修复：Gate-C直连（强制最短路径）
    try:
        # Gate的gateToC-p0节点
        gate_c_node_id = g.node_id_map[('gate', 'corridor', 'gateToC_p0', 'level1')]
        # C楼的gateToC-p0节点
        c_gate_node_id = g.node_id_map[('buildingC', 'corridor', 'gateToC_p0', 'level1')]
        # 强制添加极短权重边
        g.add_edge(gate_c_node_id, c_gate_node_id, 0.1)
    except KeyError:
        # 兜底：直接连接Gate Main Gate和C楼SCHOOL CLINIC
        try:
            gate_main_gate_id = g.node_id_map[('Gate', 'classroom', 'Main Gate', 'level1')]
            c_clinic_id = g.node_id_map[('C', 'classroom', 'SCHOOL CLINIC', 'level1')]
            g.add_edge(gate_main_gate_id, c_clinic_id, 0.1)
        except:
            st.warning("Gate-C直连失败，但不影响整体运行")
    
    return g

# --------------------------
# Streamlit交互逻辑（完整实现）
# --------------------------
def main():
    global graph, school_data
    
    # 初始化
    st.title("🏫 SCIS Campus Navigation System")
    
    # 加载数据
    with st.spinner("Loading campus data..."):
        # 读取JSON文件（确保文件在同目录）
        school_data = load_school_data_detailed("school_data.json")  # 替换为你的JSON文件名
        if not school_data:
            st.error("无法加载校园数据，请检查JSON文件！")
            return
        # 构建图
        graph = build_graph(school_data)
        if not graph:
            st.error("图构建失败！")
            return
    
    # Google Sheets统计（非核心，失败不影响）
    worksheet = init_google_sheet()
    total_accesses = update_access_count(worksheet)
    if total_accesses > 0:
        st.sidebar.info(f"总访问次数: {total_accesses}")
    
    # 侧边栏选择起点/终点
    st.sidebar.header("导航设置")
    
    # 提取所有可选教室
    all_classrooms = []
    for building_id in school_data.keys():
        building_name = 'Gate' if building_id == 'gate' else building_id.replace('building', '')
        building_data = school_data[building_id]
        for level in building_data['levels']:
            for classroom in level['classrooms']:
                all_classrooms.append(f"{building_name}-{classroom['name']} ({level['name']})")
    
    # 选择起点和终点
    start_class = st.sidebar.selectbox("选择起点", all_classrooms)
    end_class = st.sidebar.selectbox("选择终点", all_classrooms)
    
    # 解析选择的教室
    def parse_class_selector(selector_str):
        # 格式：Building-Name (level)
        parts = selector_str.split('-')
        building = parts[0]
        rest = '-'.join(parts[1:])
        name_part, level_part = rest.split(' (')
        name = name_part.strip()
        level = level_part.replace(')', '').strip()
        return building, name, level
    
    start_building, start_name, start_level = parse_class_selector(start_class)
    end_building, end_name, end_level = parse_class_selector(end_class)
    
    # 导航按钮
    if st.sidebar.button("开始导航"):
        if start_class == end_class:
            st.warning("起点和终点不能相同！")
            return
        
        with st.spinner("计算最优路径..."):
            # 获取起点/终点节点ID
            try:
                start_node_id = graph.node_id_map[(start_building, 'classroom', start_name, start_level)]
                end_node_id = graph.node_id_map[(end_building, 'classroom', end_name, end_level)]
            except KeyError as e:
                st.error(f"节点未找到: {e}")
                return
            
            # 计算路径
            path = dijkstra(graph, start_node_id, end_node_id)
            if not path:
                st.error("未找到有效路径！")
                return
            
            # 提取路径相关楼梯
            path_stairs = set()
            for node_id in path:
                node = graph.nodes[node_id]
                if node['type'] == 'stair':
                    path_stairs.add((node['building'], node['name'], node['level']))
            
            # 绘制导航地图
            display_options = {
                'start_level': start_level,
                'end_level': end_level,
                'path_stairs': path_stairs,
                'show_all': False,
                'path': path,
                'start_building': start_building,
                'end_building': end_building
            }
            fig, ax = plot_3d_map(school_data, graph, display_options)
            st.pyplot(fig)
            
            # 显示导航步骤
            st.header("📝 导航指引")
            steps = []
            for i in range(len(path)-1):
                current_node = graph.nodes[path[i]]
                next_node = graph.nodes[path[i+1]]
                # 简化步骤描述
                if current_node['type'] == 'classroom':
                    step_desc = f"从 {current_node['building']}-{current_node['name']} 出发"
                elif current_node['type'] == 'stair':
                    step_desc = f"到达 {current_node['building']}-{current_node['name']}（{current_node['level']}）"
                    if next_node['level'] != current_node['level']:
                        step_desc += f"，前往 {next_node['level']} 楼层"
                else:
                    direction = get_real_world_direction(graph, path[i], path[i+1])
                    step_desc = f"{direction} 走向 {next_node['building']}-{next_node['name'] if next_node['type']!='corridor' else '走廊'}"
                steps.append(step_desc)
            
            # 显示步骤
            for idx, step in enumerate(steps, 1):
                st.write(f"{idx}. {step}")
            st.success(f"✅ 到达终点：{end_building}-{end_name}（{end_level}）")
    
    # 显示全图
    if st.sidebar.button("显示全校园地图"):
        with st.spinner("绘制全地图..."):
            fig, ax = plot_3d_map(school_data, graph, {'show_all': True})
            st.pyplot(fig)

if __name__ == "__main__":
    main()
