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
    'connect_corridor': 'gold',  # 连廊专用颜色
    'building_label': {'A': 'darkblue', 'B': 'darkgreen', 'C': 'darkred', 'Gate': 'darkgoldenrod'}
}

# --------------------------
# 连廊配置（核心：定义各建筑连廊的物理连接关系）
# --------------------------
CORRIDOR_CONNECTIONS = {
    # 格式: (建筑1, 连廊节点名, 建筑2, 连廊节点名, 权重/距离)
    ('A', 'connect_to_B', 'B', 'connect_to_A', 5.0),
    ('A', 'connect_to_Gate', 'Gate', 'connect_to_A', 8.0),
    ('B', 'connect_to_C', 'C', 'connect_to_B', 5.0),
    ('B', 'connect_to_Gate', 'Gate', 'connect_to_B', 7.0),
    ('C', 'connect_to_Gate', 'Gate', 'connect_to_C', 9.0),
    ('A', 'connect_to_C', 'C', 'connect_to_A', 12.0),  # 备用连廊
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
    
    # 存储连廊节点坐标（用于绘制连廊）
    corridor_junction_coords = {}
    
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
                    show_level = (start_building == 'Gate' or end_building == 'Gate') or any((building_name, s_name, level_name) in path_stairs for s_name in ['GateStairs'])
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

                # 绘制走廊（区分普通走廊和连廊）
                for corr_idx, corridor in enumerate(level.get('corridors', [])):
                    points = corridor['points']
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    z_coords = [p[2] for p in points]
                    
                    is_external = corridor.get('type') == 'external'
                    # 判断是否是连廊
                    is_connecting = 'name' in corridor and any(
                        conn in corridor['name'] for conn in ['connect_to_A', 'connect_to_B', 'connect_to_C', 'connect_to_Gate']
                    )
                    
                    if is_external:
                        ext_style = corridor.get('style', {})
                        corr_line_color = ext_style.get('color', 'gray')
                        corr_line_style = ext_style.get('lineType', '--')
                        corr_line_width = 10
                        corr_label = f"External Corridor ({building_name}-{corridor.get('name', f'corr{corr_idx}')})"
                    elif is_connecting:
                        # 连廊专用样式
                        corr_line_color = COLORS['connect_corridor']
                        corr_line_style = '-'
                        corr_line_width = 15  # 更粗的线条突出连廊
                        corr_label = f"Connecting Corridor ({building_name} ↔ {corridor['name'].split('_')[-1]})"
                        # 记录连廊节点坐标
                        for i, (px, py, pz) in enumerate(points):
                            node_name = f"{corridor['name']}-p{i}"
                            corridor_junction_coords[(building_name, node_name)] = (px, py, pz)
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
                        ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=80 if is_connecting else 40, 
                                  marker='s' if is_connecting else 'o', alpha=0.9)
                        if is_connecting:
                            # 连廊节点添加标签
                            ax.text(px, py, pz+0.5, corridor['name'], color=COLORS['corridor_label'], 
                                    fontweight='bold', fontsize=12, ha='center')

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

    # 绘制建筑间连廊连接线（核心：可视化连廊连接）
    for conn in CORRIDOR_CONNECTIONS:
        b1, n1, b2, n2, _ = conn
        # 找连廊起点和终点坐标
        start_key = (b1, f"{n1}-p0")
        end_key = (b2, f"{n2}-p0")
        if start_key in corridor_junction_coords and end_key in corridor_junction_coords:
            x1, y1, z1 = corridor_junction_coords[start_key]
            x2, y2, z2 = corridor_junction_coords[end_key]
            # 绘制连廊连接线
            ax.plot([x1, x2], [y1, y2], [z1, z2], 
                   color=COLORS['connect_corridor'], linewidth=10, linestyle='-', 
                   alpha=0.7, label=f"Corridor {b1}-{b2}" if f"Corridor {b1}-{b2}" not in ax.get_legend_handles_labels()[1] else "")

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
                elif node_type == 'corridor' and 'connect_to' in graph.nodes[node_id]['name']:
                    labels.append(f"连廊({graph.nodes[node_id]['name'].split('_')[-1]})")
                else:
                    labels.append("")
            # 绘制路径
            ax.plot(x, y, z, color=COLORS['path'], linewidth=8, linestyle='-', marker='o', markersize=12, label='Navigation Path')
            # 起点终点标记
            ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=1200, marker='*', label='Start', edgecolors='black', linewidth=3)
            ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=1200, marker='*', label='End', edgecolors='black', linewidth=3)
            # 起点终点标签
            ax.text(x[0], y[0], z[0]+1, f"起点\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=18, ha='center')
            ax.text(x[-1], y[-1], z[-1]+1, f"终点\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=18, ha='center')
            # 连廊节点高亮
            for i, node_id in enumerate(path):
                if graph.nodes[node_id]['type'] == 'corridor' and 'connect_to' in graph.nodes[node_id]['name']:
                    ax.scatter(x[i], y[i], z[i], color='yellow', s=800, marker='D', edgecolors='red', linewidth=2, alpha=0.8)
        except Exception as e:
            st.warning(f"路径绘制警告: {str(e)}")

    # 图表配置
    ax.set_xlabel('X Coordinate (m)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize=18, fontweight='bold')
    ax.set_zlabel('Floor Height (Z Value)', fontsize=18, fontweight='bold')
    ax.set_title('SCIS Campus 3D Navigation Map (A/B/C/Gate with Corridors)', fontsize=26, fontweight='bold', pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=2)
    # 调整视角，更好地展示连廊
    ax.view_init(elev=20, azim=45)
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

# --------------------------
# 路径算法相关
# --------------------------
def euclidean_distance(coords1, coords2):
    """欧式距离计算"""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(coords1, coords2)))

def build_navigation_graph(school_data):
    """构建导航图（完整的连廊网络）"""
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
            # 添加走廊节点（包括连廊）
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

    # 第三步：添加建筑间连廊连接（核心）
    # 先获取各建筑的连廊节点
    corridor_nodes = {}
    for building_key in ['buildingA', 'buildingB', 'buildingC', 'gate']:
        building_name = building_mapping[building_key]
        corridor_nodes[building_name] = {}
        # 获取level1的连廊节点
        for node_id, node_info in graph.nodes.items():
            if (node_info['building'] == building_name and 
                node_info['type'] == 'corridor' and 
                node_info['level'] == 'level1' and
                'connect_to' in node_info['name']):
                corridor_nodes[building_name][node_info['name']] = node_id
    
    # 添加预定义的连廊连接
    for conn in CORRIDOR_CONNECTIONS:
        b1, n1, b2, n2, weight = conn
        # 构建完整的节点名（加-p0后缀）
        n1_full = f"{n1}-p0"
        n2_full = f"{n2}-p0"
        
        # 获取节点ID
        node1_id = corridor_nodes[b1].get(n1_full)
        node2_id = corridor_nodes[b2].get(n2_full)
        
        if node1_id and node2_id:
            graph.add_edge(node1_id, node2_id, weight)
            st.success(f"已建立连廊连接：{b1}({n1}) ↔ {b2}({n2}) (距离：{weight}m)")
        else:
            st.warning(f"无法建立连廊连接：{b1}({n1}) ↔ {b2}({n2}) - 节点不存在")

    return graph

def dijkstra(graph, start_node):
    """Dijkstra最短路径算法（自然选择最优连廊路径）"""
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    nodes = set(graph.nodes.keys())
    
    while nodes:
        # 找到距离最小的节点
        min_node = min(nodes, key=lambda node: distances[node])
        nodes.remove(min_node)
        
        if distances[min_node] == float('inf'):
            break
        
        # 更新邻居节点距离
        for neighbor, weight in graph.nodes[min_node]['neighbors'].items():
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
    """导航核心逻辑（走连廊）"""
    valid_buildings = ['A', 'B', 'C', 'Gate']
    if start_building not in valid_buildings or end_building not in valid_buildings:
        return None, "无效的建筑选择！仅支持A/B/C/Gate", None, None
        
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

        # 计算最短路径（自然选择最优连廊）
        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)
        
        if path:
            total_distance = distances[end_node]
            simplified_path = []
            path_stairs = set()
            prev_building = None
            prev_node_type = None
            
            # 简化路径描述（突出连廊）
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
                elif node_type == 'corridor':
                    if 'connect_to' in node_name:
                        # 连廊节点
                        target_building = node_name.split('_')[-1]
                        if prev_building and prev_building != node_building:
                            simplified_path.append(f"通过连廊从{prev_building}楼前往{node_building}楼")
                        elif prev_node_type != 'corridor':
                            simplified_path.append(f"进入{node_building}楼{target_building}方向连廊")
                    elif prev_building == node_building and prev_node_type == 'corridor':
                        continue  # 跳过连续的普通走廊节点
                    else:
                        simplified_path.append(f"{node_building}楼{node_level}走廊")
                
                prev_building = node_building
                prev_node_type = node_type
            
            # 去重并优化路径描述
            simplified_path = [p for i, p in enumerate(simplified_path) if i == 0 or p != simplified_path[i-1]]
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
            return path, f"总距离：{total_distance:.2f} 米 (优先走连廊)", full_path_str, display_options
        else:
            return None, "未找到可行路径（请检查连廊连接）", None, None
    except Exception as e:
        st.error(f"导航错误详情：{str(e)}")
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
        .welcome-subtitle {
            font-size: 1.5rem;
            color: #666;
            margin-bottom: 3rem;
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
    st.markdown('<div class="welcome-subtitle">3D可视化 · 连廊路径规划 · A/B/C/Gate全覆盖</div>', unsafe_allow_html=True)
    
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
        .stAlert {padding: 0.5rem; margin: 0.5rem 0;}
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🏫 SCIS 校园3D导航系统")
    st.markdown("### 🛤️ 连廊路径规划 · A/B/C/Gate建筑互通")

    # 初始化状态
    if 'display_options' not in st.session_state:
        reset_app_state()

    # 加载数据
    school_data = load_school_data_detailed('school_data_detailed.json')
    if not school_data:
        return
    
    # 构建导航图（连廊网络）
    if 'graph' not in st.session_state or st.session_state['graph'] is None:
        with st.spinner("正在构建连廊导航网络..."):
            st.session_state['graph'] = build_navigation_graph(school_data)
    
    global graph
    graph = st.session_state['graph']
    
    # 获取教室信息
    building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
    
    # 侧边栏导航控制
    with st.sidebar:
        st.header("📍 导航设置")
        
        # 显示连廊信息
        st.info("📌 连廊连接：\n• A ↔ B (5m)\n• B ↔ C (5m)\n• A ↔ Gate (8m)\n• B ↔ Gate (7m)\n• C ↔ Gate (9m)")
        
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
                    with st.spinner("正在规划最优连廊路径..."):
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
                        st.success(f"✅ 导航成功！{msg}")
                    else:
                        st.error(f"❌ {msg}")
                        reset_app_state()
                except Exception as e:
                    st.error(f"❌ 导航失败：{str(e)}")
                    reset_app_state()
        
        with col2:
            if st.button("🗺️ 显示全图"):
                reset_app_state()
                st.success("已切换到全图视图（显示所有连廊）")
        
        # 重置按钮
        if st.button("🔄 重置"):
            reset_app_state()
            st.rerun()
    
    # 显示路径信息
    if 'path_result' in st.session_state:
        msg, path_str = st.session_state['path_result']
        st.markdown("### 📝 导航路径（优先走连廊）")
        st.info(f"""
        <div style="font-size: 16px; line-height: 1.8;">
        {path_str}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"### 📏 {msg}")
    
    # 绘制3D地图
    st.markdown("### 3D校园地图（连廊高亮显示）")
    with st.spinner("正在绘制3D地图和导航路径..."):
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
