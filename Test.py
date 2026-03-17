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
# 地图与导航核心逻辑（更新以支持Gate）
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

    # 遍历所有建筑（包括Gate）
    for building_id in school_data.keys():
        if building_id == 'gate' or building_id.startswith('building'):
            if building_id == 'gate':
                building_name = 'Gate'
            else:
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
                    # Gate建筑特殊处理
                    if building_name == 'Gate':
                        show_level = (level_name == start_level) or (level_name == end_level)
                    elif building_name == 'B':
                        show_level = any((building_name, s_name, level_name) in path_stairs for s_name in ['StairsB1', 'StairsB2'])
                        if (start_building == 'B' or end_building == 'B') or (start_building in ['A','C','Gate'] and end_building in ['A','C','Gate'] and 'B' in [start_building, end_building]):
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

    # 添加建筑标签
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
        # 处理Gate建筑
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

def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b)**2 for a, b in zip(coords1, coords2)))

def calculate_relative_direction(from_point, to_point, facing_direction):
    """
    通用化相对方向计算：
    - from_point: 当前位置 (x,y)
    - to_point: 目标位置 (x,y)
    - facing_direction: 当前面朝方向向量 (dx, dy)
    - 返回: 向左/向右/直行
    """
    # 计算目标方向向量
    target_vec = np.array([to_point[0] - from_point[0], to_point[1] - from_point[1]])
    len_target = np.linalg.norm(target_vec)
    if len_target < 1e-6:
        return ""
    target_vec = target_vec / len_target
    
    # 归一化面朝方向
    len_facing = np.linalg.norm(facing_direction)
    if len_facing < 1e-6:
        return ""
    facing_vec = facing_direction / len_facing
    
    # 计算叉积（判断左右）
    cross = np.cross(np.append(facing_vec, 0), np.append(target_vec, 0))[2]
    
    # 计算点积（判断前后）
    dot = np.dot(facing_vec, target_vec)
    
    # 角度阈值（可调整）
    angle_threshold = np.cos(np.radians(15))  # 15度以内视为直行
    
    if dot > angle_threshold:
        return "直行"
    elif cross > 0.01:
        return "向左"
    elif cross < -0.01:
        return "向右"
    else:
        return "直行"

def get_corridor_main_direction(corridor_points):
    """
    计算走廊的主延伸方向：
    - corridor_points: 走廊的坐标点列表
    - 返回: 走廊主方向向量 (dx, dy)
    """
    if len(corridor_points) < 2:
        return np.array([0, 1])  # 默认Y轴正方向
    
    # 计算走廊的首尾点向量（主方向）
    start = np.array([corridor_points[0][0], corridor_points[0][1]])
    end = np.array([corridor_points[-1][0], corridor_points[-1][1]])
    main_dir = end - start
    
    # 如果长度过短，取中间段
    if np.linalg.norm(main_dir) < 1e-6 and len(corridor_points) > 2:
        mid_idx = len(corridor_points) // 2
        start = np.array([corridor_points[mid_idx-1][0], corridor_points[mid_idx-1][1]])
        end = np.array([corridor_points[mid_idx+1][0], corridor_points[mid_idx+1][1]])
        main_dir = end - start
    
    return main_dir

def get_real_world_direction(graph, current_node_id, next_node_id, prev_node_id=None):
    """
    真实场景的方向判断主函数（适配所有场景）：
    1. 教室→走廊：面朝走廊主方向，判断左右
    2. 走廊→走廊：面朝当前行走方向，判断左右转
    3. 走廊→楼梯/教室：基于走廊主方向判断左右
    """
    current_node = graph.nodes[current_node_id]
    next_node = graph.nodes[next_node_id]
    
    # 提取平面坐标
    curr_x, curr_y, _ = current_node['coordinates']
    next_x
