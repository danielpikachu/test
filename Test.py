import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os

# 配置后端和页面
plt.switch_backend('Agg')
st.set_page_config(page_title="SCIS Navigation System", layout="wide")

# --------------------------
# Google Sheets 配置（适配 Streamlit Secrets TOML）
# --------------------------
SHEET_NAME = 'Navigation visitors'  # Google表格名称
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
        st.error("Streamlit Secrets中未找到google_service_account配置，请检查TOML格式")
        return None
    except Exception as e:
        st.error(f"密钥加载失败: {str(e)}")
        return None

def init_google_sheet():
    """初始化Google Sheets连接并确保表格结构正确"""
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
            stats_worksheet.append_row(["Timestamp", "Access_Count", "Total_Accesses"])
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
# 颜色配置
# --------------------------
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
        'GateStairs': '#8B4513',
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
# 地图与导航核心逻辑
# --------------------------
def load_school_data_detailed(filename):
    """加载学校3D数据"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"数据文件 {filename} 未找到，请检查文件路径！")
        return None
    except json.JSONDecodeError:
        st.error("数据文件格式错误，不是有效的JSON文件！")
        return None
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        return None

def plot_3d_map(school_data, display_options=None):
    """绘制3D校园地图"""
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

    # 建筑映射
    building_mapping = {
        'buildingA': 'A',
        'buildingB': 'B',
        'buildingC': 'C',
        'gate': 'Gate'
    }
    
    # 遍历所有建筑
    for building_key in school_data.keys():
        if building_key not in building_mapping:
            continue
        building_name = building_mapping[building_key]
        building_data = school_data[building_key]
        
        displayed_levels = []
        max_displayed_z = -float('inf')
        max_displayed_y = -float('inf')
        corresponding_x = 0
        level_count = 0
        
        # 遍历建筑楼层
        for level in building_data['levels']:
            level_name = level['name']
            z = level['z']
            
            # 控制楼层显示逻辑
            show_level = show_all
            if not show_all:
                if building_name == 'B':
                    show_level = any((building_name, s_name, level_name) in path_stairs for s_name in ['StairsB1', 'StairsB2'])
                    if (start_building == 'B' or end_building == 'B'):
                        show_level = show_level or (level_name == 'level1')
                elif building_name == 'Gate':
                    # 只有起点/终点是Gate时才显示Gate
                    show_level = (start_building == 'Gate' or end_building == 'Gate') and any((building_name, s_name, level_name) in path_stairs for s_name in ['GateStairs'])
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
                # 绘制楼层平面
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
                ax.plot_trisurf(x_plane[:-1], y_plane[:-1], z_plane[:-1], color=building_fill_color, alpha=0.3)

                # 绘制走廊
                for corr_idx, corridor in enumerate(level.get('corridors', [])):
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
                    elif 'name' in corridor and ('connectToBuilding' in corridor['name'] or 'gateTo' in corridor['name']):
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
                        ax.plot(x, y, z_coords, color=corr_line_color, linestyle=corr_line_style,
                                linewidth=corr_line_width, alpha=0.8, label=corr_label)
                    else:
                        ax.plot(x, y, z_coords, color=corr_line_color, linestyle=corr_line_style,
                                linewidth=corr_line_width, alpha=0.8)
                    # 走廊节点
                    for px, py, pz in points:
                        ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=40, marker='s', alpha=0.9)

                # 绘制教室
                for classroom in level.get('classrooms', []):
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
            for stair in level.get('stairs', []):
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
        
        # 建筑标签位置
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

    # 绘制建筑名称标签
    for building_name, (x, y, z) in building_label_positions.items():
        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", 
                        facecolor=COLORS['building'].get(building_name, 'lightgray'), alpha=0.7)
        ax.text(
            x, y, z, f"Building {building_name}", 
            color=COLORS['building_label'].get(building_name, 'black'), 
            fontweight='bold', fontsize=30,
            ha='center', va='center', bbox=bbox_props
        )

    # 绘制导航路径
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
            st.warning(f"路径绘制警告: {str(e)}")

    # 图表配置
    ax.set_xlabel('X Coordinate', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y Coordinate', fontsize=18, fontweight='bold')
    ax.set_zlabel('Floor Height (Z Value)', fontsize=18, fontweight='bold')
    ax.set_title('SCIS Campus 3D Navigation Map (A/B/C/Gate)', fontsize=24, fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=2)
    return fig, ax

# --------------------------
# 导航图类
# --------------------------
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}

    def add_node(self, building_key, building_name, node_type, name, level, coordinates):
        """添加节点"""
        if node_type == 'corridor':
            node_id = f"{building_name}-corr-{name}@{level}"
        else:
            node_id = f"{building_name}-{node_type}-{name}@{level}"
        
        self.nodes[node_id] = {
            'building_key': building_key,
            'building': building_name,
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        # 节点映射
        map_key = (building_key, node_type, name, level)
        self.node_id_map[map_key] = node_id
        if node_type == 'classroom':
            class_key = (building_name, name, level)
            self.node_id_map[class_key] = node_id
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        """添加边（双向）"""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight
    
    def remove_gate_connections(self):
        """移除所有与Gate相关的连接（保留Gate节点但断开连接）"""
        # 找出所有Gate节点
        gate_nodes = [node_id for node_id, node_info in self.nodes.items() if node_info['building'] == 'Gate']
        
        # 断开所有指向Gate节点的连接
        for node_id in self.nodes:
            if node_id in gate_nodes:
                continue
            # 过滤掉邻居中的Gate节点
            self.nodes[node_id]['neighbors'] = {
                neighbor_id: weight 
                for neighbor_id, weight in self.nodes[node_id]['neighbors'].items()
                if neighbor_id not in gate_nodes
            }
        
        # 清空Gate节点的所有邻居
        for gate_node_id in gate_nodes:
            self.nodes[gate_node_id]['neighbors'] = {}

# --------------------------
# 路径算法相关
# --------------------------
def euclidean_distance(coords1, coords2):
    """欧式距离计算"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(coords1, coords2)))

def build_navigation_graph(school_data, include_gate_connections=False):
    """
    构建导航图
    :param include_gate_connections: 是否包含与Gate的连接（仅当起点/终点是Gate时设为True）
    """
    graph = Graph()
    # 建筑映射
    building_mapping = {
        'buildingA': 'A',
        'buildingB': 'B',
        'buildingC': 'C',
        'gate': 'Gate'
    }
    
    # 第一步：添加所有节点
    for building_key in school_data.keys():
        if building_key not in building_mapping:
            continue
        building_name = building_mapping[building_key]
        building_data = school_data[building_key]
        
        for level in building_data['levels']:
            level_name = level['name']
            # 添加教室节点
            for classroom in level.get('classrooms', []):
                graph.add_node(
                    building_key,
                    building_name,
                    'classroom',
                    classroom['name'],
                    level_name,
                    classroom['coordinates']
                )
            # 添加楼梯节点
            for stair in level.get('stairs', []):
                graph.add_node(
                    building_key,
                    building_name,
                    'stair',
                    stair['name'],
                    level_name,
                    stair['coordinates']
                )
            # 添加走廊节点
            for corr_idx, corridor in enumerate(level.get('corridors', [])):
                corr_name = corridor.get('name', f'corr{corr_idx}')
                for p_idx, point in enumerate(corridor['points']):
                    point_name = f"{corr_name}-p{p_idx}"
                    graph.add_node(
                        building_key,
                        building_name,
                        'corridor',
                        point_name,
                        level_name,
                        point
                    )

    # 第二步：添加建筑内部边
    for building_key in school_data.keys():
        if building_key not in building_mapping:
            continue
        building_name = building_mapping[building_key]
        building_data = school_data[building_key]

        for level in building_data['levels']:
            level_name = level['name']
            # 筛选当前楼层走廊节点
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]
            # 连接同走廊的点
            for corr_idx, corridor in enumerate(level.get('corridors', [])):
                corr_name = corridor.get('name', f'corr{corr_idx}')
                corr_points = corridor['points']
                for p_idx in range(len(corr_points) - 1):
                    current_point_name = f"{corr_name}-p{p_idx}"
                    next_point_name = f"{corr_name}-p{p_idx + 1}"
                    current_node_id = graph.node_id_map.get((building_key, 'corridor', current_point_name, level_name))
                    next_node_id = graph.node_id_map.get((building_key, 'corridor', next_point_name, level_name))
                    if current_node_id and next_node_id:
                        distance = euclidean_distance(corr_points[p_idx], corr_points[p_idx + 1])
                        graph.add_edge(current_node_id, next_node_id, distance)
            # 连接相邻走廊节点
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2)
                    if distance < 3.0:
                        graph.add_edge(node1_id, node2_id, distance)
            # 连接教室到最近走廊
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
                    st.warning(f"警告：{building_name}楼{level_name}的{graph.nodes[class_node_id]['name']}没有找到相邻走廊")
            # 连接楼梯到最近走廊
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
        
        # 建筑内部跨楼层连接
        for connection in building_data.get('connections', []):
            from_obj_name, from_level = connection['from']
            to_obj_name, to_level = connection['to']
            
            # 确定节点类型
            from_obj_type = 'stair' if from_obj_name.startswith(('Stairs', 'GateStairs')) else 'corridor'
            to_obj_type = 'stair' if to_obj_name.startswith(('Stairs', 'GateStairs')) else 'corridor'
            
            # 处理走廊点命名
            if from_obj_type == 'corridor' and not from_obj_name.endswith('-p0'):
                from_obj_name = f"{from_obj_name}-p0"
            if to_obj_type == 'corridor' and not to_obj_name.endswith('-p0'):
                to_obj_name = f"{to_obj_name}-p0"
            
            # 获取节点ID
            from_node_id = graph.node_id_map.get((building_key, from_obj_type, from_obj_name, from_level))
            to_node_id = graph.node_id_map.get((building_key, to_obj_type, to_obj_name, to_level))
            
            if from_node_id and to_node_id:
                graph.add_edge(from_node_id, to_node_id, 5.0)

    # 第三步：建立ABC楼之间的直接连接（核心修改：AC连接权重与AB/BC一致）
    # 获取各建筑level1的走廊节点
    a_corridors = [node_id for node_id, node_info in graph.nodes.items() 
                   if node_info['building'] == 'A' and node_info['type'] == 'corridor' and node_info['level'] == 'level1']
    b_corridors = [node_id for node_id, node_info in graph.nodes.items() 
                   if node_info['building'] == 'B' and node_info['type'] == 'corridor' and node_info['level'] == 'level1']
    c_corridors = [node_id for node_id, node_info in graph.nodes.items() 
                   if node_info['building'] == 'C' and node_info['type'] == 'corridor' and node_info['level'] == 'level1']
    
    # A-B直接连接（强制优先）
    if a_corridors and b_corridors:
        # 选择A楼最靠近B楼的节点
        a_node = min(a_corridors, key=lambda x: graph.nodes[x]['coordinates'][1])
        # 选择B楼最靠近A楼的节点
        b_node = min(b_corridors, key=lambda x: abs(graph.nodes[x]['coordinates'][0] - graph.nodes[a_node]['coordinates'][0]))
        # 设置极小的权重，确保优先选择
        graph.add_edge(a_node, b_node, 1.0)
    
    # B-C直接连接（强制优先）
    if b_corridors and c_corridors:
        # 选择B楼最靠近C楼的节点
        b_node = max(b_corridors, key=lambda x: graph.nodes[x]['coordinates'][1])
        # 选择C楼最靠近B楼的节点
        c_node = min(c_corridors, key=lambda x: abs(graph.nodes[x]['coordinates'][0] - graph.nodes[b_node]['coordinates'][0]))
        # 设置极小的权重，确保优先选择
        graph.add_edge(b_node, c_node, 1.0)
    
    # A-C直接连接（核心修改：权重改为1.0，与AB/BC一致，确保AC优先直连）
    if a_corridors and c_corridors:
        a_node = max(a_corridors, key=lambda x: graph.nodes[x]['coordinates'][0])
        c_node = min(c_corridors, key=lambda x: graph.nodes[x]['coordinates'][0])
        graph.add_edge(a_node, c_node, 1.0)  # 权重从1.5改为1.0，确保AC直连优先
    
    # 第四步：仅在需要时添加Gate连接
    if include_gate_connections:
        gate_corridors = [node_id for node_id, node_info in graph.nodes.items() 
                          if node_info['building'] == 'Gate' and node_info['type'] == 'corridor' and node_info['level'] == 'level1']
        
        # Gate到A的连接（高权重，仅备用）
        if gate_corridors and a_corridors:
            gate_node = gate_corridors[0]
            a_node = min(a_corridors, key=lambda x: euclidean_distance(graph.nodes[x]['coordinates'], graph.nodes[gate_node]['coordinates']))
            graph.add_edge(gate_node, a_node, 100.0)  # 极高权重，仅当无其他路径时选择
        
        # Gate到B的连接（高权重，仅备用）
        if gate_corridors and b_corridors:
            gate_node = gate_corridors[0]
            b_node = min(b_corridors, key=lambda x: euclidean_distance(graph.nodes[x]['coordinates'], graph.nodes[gate_node]['coordinates']))
            graph.add_edge(gate_node, b_node, 100.0)
        
        # Gate到C的连接（高权重，仅备用）
        if gate_corridors and c_corridors:
            gate_node = gate_corridors[0]
            c_node = min(c_corridors, key=lambda x: euclidean_distance(graph.nodes[x]['coordinates'], graph.nodes[gate_node]['coordinates']))
            graph.add_edge(gate_node, c_node, 100.0)
    else:
        # 移除所有与Gate相关的连接
        graph.remove_gate_connections()

    return graph

def dijkstra(graph, start_node, ignore_gate_nodes=True):
    """
    Dijkstra最短路径算法
    :param ignore_gate_nodes: 是否忽略Gate节点（ABC导航时设为True）
    """
    # 初始化距离
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    
    # 如果需要忽略Gate节点，提前标记
    if ignore_gate_nodes:
        gate_nodes = set(node_id for node_id, node_info in graph.nodes.items() if node_info['building'] == 'Gate')
        # 将Gate节点的距离设为无穷大，确保不会被选中
        for gate_node in gate_nodes:
            distances[gate_node] = float('inf')
    
    nodes = set(graph.nodes.keys())
    
    while nodes:
        # 找到距离最小的节点
        min_node = min(nodes, key=lambda node: distances[node])
        nodes.remove(min_node)
        
        if distances[min_node] == float('inf'):
            break
        
        # 更新邻居节点距离
        for neighbor, weight in graph.nodes[min_node]['neighbors'].items():
            # 如果忽略Gate节点，跳过Gate邻居
            if ignore_gate_nodes and graph.nodes[neighbor]['building'] == 'Gate':
                continue
                
            alternative_route = distances[min_node] + weight
            if alternative_route < distances[neighbor]:
                distances[neighbor] = alternative_route
                previous_nodes[neighbor] = min_node
    
    return distances, previous_nodes

def construct_path(previous_nodes, end_node):
    """构建路径"""
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path if len(path) > 1 else None

def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    """导航核心逻辑"""
    valid_buildings = ['A', 'B', 'C', 'Gate']
    if start_building not in valid_buildings or end_building not in valid_buildings:
        return None, "无效的建筑选择！仅支持A/B/C/Gate", None, None
    
    # 判断是否需要包含Gate
    need_gate = (start_building == 'Gate') or (end_building == 'Gate')
    
    try:
        # 教室节点查询
        start_key = (start_building, start_classroom, start_level)
        end_key = (end_building, end_classroom, end_level)
        start_node = graph.node_id_map.get(start_key)
        end_node = graph.node_id_map.get(end_key)
        
        # 校验节点是否存在
        if not start_node or start_node not in graph.nodes:
            return None, f"起点不存在：{start_building}楼{start_classroom} @ {start_level}", None, None
        if not end_node or end_node not in graph.nodes:
            return None, f"终点不存在：{end_building}楼{end_classroom} @ {end_level}", None, None

        # 计算最短路径（关键：ABC导航时忽略Gate节点）
        distances, previous_nodes = dijkstra(graph, start_node, ignore_gate_nodes=not need_gate)
        path = construct_path(previous_nodes, end_node)
        
        if path:
            # 验证路径中是否包含Gate节点（ABC导航时不允许）
            if not need_gate:
                path_buildings = set(graph.nodes[node_id]['building'] for node_id in path)
                if 'Gate' in path_buildings:
                    return None, "路径规划错误：ABC楼导航不应经过Gate", None, None
            
            total_distance = distances[end_node]
            simplified_path = []
            path_stairs = set()
            prev_building = None
            
            # 简化路径描述
            for node_id in path:
                node_info = graph.nodes[node_id]
                node_type = node_info['type']
                node_name = node_info['name']
                node_level = node_info['level']
                node_building = node_info['building']
                
                if node_type == 'stair':
                    path_stairs.add((node_building, node_name, node_level))
                    simplified_path.append(f"{node_building}楼{node_name}({node_level})")
                elif node_type == 'classroom':
                    simplified_path.append(f"{node_building}楼{node_name}({node_level})")
                elif node_type == 'corridor' and prev_building and prev_building != node_building:
                    simplified_path.append(f"从{prev_building}楼前往{node_building}楼({node_level})")
                
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
            return path, f"总距离：{total_distance:.2f} 单位", full_path_str, display_options
        else:
            return None, "未找到可行路径", None, None
    except Exception as e:
        return None, f"导航错误：{str(e)}", None, None

# --------------------------
# 数据提取工具
# --------------------------
def get_classroom_info(school_data):
    """提取所有建筑的楼层和教室信息"""
    try:
        # 建筑映射
        building_mapping = {
            'buildingA': 'A',
            'buildingB': 'B',
            'buildingC': 'C',
            'gate': 'Gate'
        }
        # 强制显示顺序
        building_order = ['buildingA', 'buildingB', 'buildingC', 'gate']
        
        building_names = []
        classrooms_by_building = {}
        levels_by_building = {}
        
        for building_key in building_order:
            if building_key not in school_data:
                continue
            building_name = building_mapping[building_key]
            building_names.append(building_name)
            
            building_data = school_data[building_key]
            levels = []
            classrooms_by_level = {}
            
            for level in building_data['levels']:
                level_name = level['name']
                levels.append(level_name)
                # 提取教室名称
                classrooms = [classroom['name'] for classroom in level.get('classrooms', [])]
                classrooms_by_level[level_name] = classrooms
            
            levels_by_building[building_name] = levels
            classrooms_by_building[building_name] = classrooms_by_level
        
        return building_names, levels_by_building, classrooms_by_building
    except Exception as e:
        st.error(f"提取教室信息失败: {str(e)}")
        return [], {}, {}

# --------------------------
# 状态管理
# --------------------------
def reset_app_state():
    """重置应用状态"""
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
    st.session_state['graph'] = None
    if 'path_result' in st.session_state:
        del st.session_state['path_result']

# --------------------------
# 页面布局
# --------------------------
def welcome_page():
    """欢迎页面"""
    if 'worksheet' not in st.session_state:
        st.session_state['worksheet'] = init_google_sheet()
    total_accesses = get_total_accesses(st.session_state['worksheet'])
    
    # 样式美化
    st.markdown("""
        <style>
        .welcome-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 80vh;
            text-align: center;
        }
        .welcome-title {
            font-size: 3.5rem;
            color: #2c3e50;
            margin-bottom: 2rem;
        }
        .enter-button {
            font-size: 1.5rem;
            padding: 1rem 3rem;
            border-radius: 10px;
        }
        .access-count {
            margin-top: 2rem;
            font-size: 1.2rem;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 页面内容
    st.markdown('<div class="welcome-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="welcome-title">SCIS 校园导航系统</h1>', unsafe_allow_html=True)
    st.markdown('<h3>3D可视化 · 跨建筑路径规划 · A/B/C/Gate全覆盖</h3>', unsafe_allow_html=True)
    
    if st.button('进入系统', key='enter_btn', use_container_width=False, 
                help='点击进入3D导航界面', type='primary'):
        update_access_count(st.session_state['worksheet'])
        st.session_state['page'] = 'main'
        st.rerun()
    
    st.markdown(f'<div class="access-count">总访问次数：{total_accesses}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def main_interface():
    """主界面"""
    # 样式优化
    st.markdown("""
        <style>
        .stApp {padding-top: 1rem !important;}
        .sidebar .sidebar-content {padding: 1rem;}
        .block-container {padding: 1rem;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏫 SCIS 校园3D导航系统")
    st.markdown("### 支持A/B/C/Gate建筑互导航 · ABC楼优先内部连接 · AC楼直连不经过B楼")

    # 初始化状态
    if 'display_options' not in st.session_state:
        reset_app_state()

    # 加载数据
    school_data = load_school_data_detailed('school_data_detailed.json')
    if not school_data:
        return
    
    # 构建导航图（根据是否需要Gate动态调整）
    if 'graph' not in st.session_state or st.session_state['graph'] is None:
        # 默认不包含Gate连接
        st.session_state['graph'] = build_navigation_graph(school_data, include_gate_connections=False)
    
    global graph
    graph = st.session_state['graph']
    
    # 获取教室信息
    building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
    
    # 侧边栏导航控制
    with st.sidebar:
        st.header("📍 导航设置")
        
        # 起点设置
        st.subheader("起点")
        start_building = st.selectbox("建筑", building_names, key='start_building')
        start_level = None
        start_classroom = None
        if start_building in levels_by_building:
            start_level = st.selectbox("楼层", levels_by_building[start_building], key='start_level')
            if start_level in classrooms_by_building[start_building]:
                start_classrooms = classrooms_by_building[start_building][start_level]
                start_classroom = st.selectbox("教室/位置", start_classrooms, key='start_classroom')
        
        # 终点设置
        st.subheader("终点")
        end_building = st.selectbox("建筑", building_names, key='end_building')
        end_level = None
        end_classroom = None
        if end_building in levels_by_building:
            end_level = st.selectbox("楼层", levels_by_building[end_building], key='end_level')
            if end_level in classrooms_by_building[end_building]:
                end_classrooms = classrooms_by_building[end_building][end_level]
                end_classroom = st.selectbox("教室/位置", end_classrooms, key='end_classroom')
        
        # 操作按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧭 开始导航", type='primary'):
                try:
                    # 重新构建图（根据是否需要Gate）
                    need_gate = (start_building == 'Gate') or (end_building == 'Gate')
                    graph = build_navigation_graph(school_data, include_gate_connections=need_gate)
                    st.session_state['graph'] = graph
                    
                    path, msg, path_str, display_options = navigate(
                        graph,
                        start_building,
                        start_classroom,
                        start_level,
                        end_building,
                        end_classroom,
                        end_level
                    )
                    if path:
                        st.session_state['current_path'] = path
                        st.session_state['display_options'] = display_options
                        st.session_state['path_result'] = (msg, path_str)
                        st.success(f"导航成功！{msg}")
                    else:
                        st.error(msg)
                        reset_app_state()
                except Exception as e:
                    st.error(f"导航失败：{str(e)}")
                    reset_app_state()
        
        with col2:
            if st.button("🗺️ 显示全图"):
                reset_app_state()
                st.success("已切换到全图视图")
        
        # 重置按钮
        if st.button("🔄 重置"):
            reset_app_state()
            st.rerun()
    
    # 显示路径信息
    if 'path_result' in st.session_state:
        msg, path_str = st.session_state['path_result']
        st.markdown("### 📝 导航路径")
        st.info(f"{path_str}")
        st.markdown(f"### 📏 {msg}")
    
    # 绘制3D地图
    st.markdown("### 3D校园地图")
    fig, ax = plot_3d_map(school_data, st.session_state['display_options'])
    st.pyplot(fig, use_container_width=True)

# --------------------------
# 应用入口
# --------------------------
def main():
    """应用主函数"""
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'
    
    if st.session_state['page'] == 'welcome':
        welcome_page()
    elif st.session_state['page'] == 'main':
        main_interface()

if __name__ == "__main__":
    main()
