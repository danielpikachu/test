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

def get_walking_direction(prev_coords, curr_coords, next_coords):
    """
    调整后的第一人称方向判断：
    1. 楼梯优先：向上/向下
    2. 平面移动：只返回左转/右转/向左/向右（取消向前/向后/直行）
    3. 模拟用户面朝走廊时的左右判断
    """
    # 先判断上下（楼梯优先）
    dz = next_coords[2] - curr_coords[2]
    if abs(dz) > 0.1:
        return "向上" if dz > 0 else "向下"
    
    # 只看平面XY
    x1, y1, _ = prev_coords
    x2, y2, _ = curr_coords
    x3, y3, _ = next_coords

    # 计算来向和去向向量
    dx_in = x2 - x1
    dy_in = y2 - y1
    dx_out = x3 - x2
    dy_out = y3 - y2

    # 极小位移视为无方向
    eps = 1e-6
    if (abs(dx_in) < eps and abs(dy_in) < eps) or (abs(dx_out) < eps and abs(dy_out) < eps):
        return ""

    # 单位向量（归一化，只保留方向）
    len_in = np.hypot(dx_in, dy_in)
    ux_in = dx_in / len_in
    uy_in = dy_in / len_in

    len_out = np.hypot(dx_out, dy_out)
    ux_out = dx_out / len_out
    uy_out = dy_out / len_out

    # 计算夹角（弧度）
    dot = ux_in * ux_out + uy_in * uy_out
    cross = ux_in * uy_out - uy_in * ux_out
    
    # 转换为角度
    angle = np.arccos(np.clip(dot, -1.0, 1.0)) * 180 / np.pi
    
    # 调整逻辑：取消直行，全部用左/右描述
    if cross > 0:
        # 左侧转向
        if angle < 30:
            return "向左"  # 小角度左转 → 向左
        else:
            return "左转"  # 大角度左转 → 左转
    else:
        # 右侧转向
        if angle < 30:
            return "向右"  # 小角度右转 → 向右
        else:
            return "右转"  # 大角度右转 → 右转

def get_real_world_start_direction(classroom_coords, corridor_coords):
    """
    结合实际场景的起点方向判断：
    1. 计算从教室到走廊的向量（出门方向）
    2. 模拟用户出门后自然"面朝走廊延伸方向"
    3. 基于这个真实朝向判断左右
    """
    # 提取平面坐标
    cx, cy, _ = classroom_coords
    cor_x, cor_y, _ = corridor_coords
    
    # 步骤1：计算从教室到走廊的向量（出门向量）
    exit_vec = np.array([cor_x - cx, cor_y - cy])
    len_exit = np.hypot(exit_vec[0], exit_vec[1])
    if len_exit < 1e-6:
        return ""
    exit_vec = exit_vec / len_exit
    
    # 步骤2：模拟走廊的延伸方向（实际场景：用户出门后面朝走廊深处）
    # 假设走廊是直线，这里取垂直于出门方向的向量作为走廊延伸方向
    # 真实场景可根据走廊实际走向调整，这里用90度旋转模拟
    corridor_forward = np.array([-exit_vec[1], exit_vec[0]])  # 左转90度 = 走廊延伸方向
    
    # 步骤3：计算目标方向（走廊节点）相对于走廊延伸方向的左右
    target_vec = exit_vec  # 目标是走到走廊节点
    cross = corridor_forward[0] * target_vec[1] - corridor_forward[1] * target_vec[0]
    
    # 步骤4：根据实际场景校准左右（可根据校园实际布局调整符号）
    if cross > 0.01:
        return "向左"
    elif cross < -0.01:
        return "向右"
    else:
        # 如果正对走廊，根据实际场景返回合理方向
        # 针对AA303场景，强制返回"向左"（可根据实际需要调整）
        return "向左"

def filter_important_nodes(path, graph):
    """
    过滤路径中的重要节点（只保留教室、楼梯、跨建筑走廊）
    """
    important_nodes = []
    
    for node_id in path:
        node_info = graph.nodes[node_id]
        node_type = node_info['type']
        node_name = node_info['name']
        
        # 保留教室和楼梯节点
        if node_type in ['classroom', 'stair']:
            important_nodes.append(node_id)
        # 保留跨建筑走廊节点
        elif node_type == 'corridor' and ('connectToBuilding' in node_name or 'gateTo' in node_name):
            # 去重：避免连续的相同类型跨建筑走廊节点
            if not important_nodes or graph.nodes[important_nodes[-1]]['type'] != 'corridor':
                important_nodes.append(node_id)
    
    return important_nodes

def build_navigation_graph(school_data):
    graph = Graph()

    # 添加所有建筑节点（包括Gate）
    for building_id in school_data.keys():
        if building_id == 'gate' or building_id.startswith('building'):
            if building_id == 'gate':
                building_name = 'Gate'
            else:
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
                        name=stair_name,
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

    # 添加内部边连接
    for building_id in school_data.keys():
        if building_id == 'gate' or building_id.startswith('building'):
            if building_id == 'gate':
                building_name = 'Gate'
            else:
                building_name = building_id.replace('building', '')
            
            building_data = school_data[building_id]

            for level in building_data['levels']:
                level_name = level['name']
                
                # 获取当前楼层的走廊节点
                corr_nodes = [
                    node_id for node_id, node_info in graph.nodes.items()
                    if node_info['building'] == building_name 
                    and node_info['type'] == 'corridor' 
                    and node_info['level'] == level_name
                ]

                # 连接走廊点
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
                        st.warning(f"Warning: Classroom {graph.nodes[class_node_id]['name']} in Building {building_name}{level_name} has no corridor connection")

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

            # 添加建筑内部连接
            for connection in building_data['connections']:
                from_obj_name, from_level = connection['from']
                to_obj_name, to_level = connection['to']
                
                from_obj_type = 'stair' if from_obj_name.startswith('Stairs') or from_obj_name.startswith('GateStairs') else 'corridor'
                to_obj_type = 'stair' if to_obj_name.startswith('Stairs') or to_obj_name.startswith('GateStairs') else 'corridor'
                
                if from_obj_type == 'corridor':
                    from_obj_name = f"{from_obj_name}-p0"
                if to_obj_type == 'corridor':
                    to_obj_name = f"{to_obj_name}-p0"
                
                from_node_id = graph.node_id_map.get((building_id, from_obj_type, from_obj_name, from_level))
                to_node_id = graph.node_id_map.get((building_id, to_obj_type, to_obj_name, to_level))
                
                if from_node_id and to_node_id:
                    graph.add_edge(from_node_id, to_node_id, 5.0)

    # A-B 连接
    a_building_id = 'buildingA'
    b_building_id = 'buildingB'
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
    c_building_id = 'buildingC'
    c_b_corr_name = 'connectToBuildingB-p1'
    c_b_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_b_corr_name, bc_connect_level))
    
    if b_c_node_id and c_b_node_id:
        coords_b = graph.nodes[b_c_node_id]['coordinates']
        coords_c = graph.nodes[c_b_node_id]['coordinates']
        distance = euclidean_distance(coords_b, coords_c)
        graph.add_edge(b_c_node_id, c_b_node_id, distance)
    else:
        st.warning("Could not find B-C level1 inter-building corridor connection nodes")
    
    # A-C level1 连接
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
    
    # A-C level3 连接
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
    
    # 添加 Gate 与 A/B/C 的连接
    gate_building_id = 'gate'
    gate_level = 'level1'
    
    # Gate-A 连接
    gate_a_corr_name = 'gateToA-p1'
    gate_a_node_id = graph.node_id_map.get((gate_building_id, 'corridor', gate_a_corr_name, gate_level))
    a_gate_corr_name = 'gateToA-p1'
    a_gate_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_gate_corr_name, connect_level1))
    
    if gate_a_node_id and a_gate_node_id:
        coords_gate = graph.nodes[gate_a_node_id]['coordinates']
        coords_a = graph.nodes[a_gate_node_id]['coordinates']
        distance = euclidean_distance(coords_gate, coords_a)
        graph.add_edge(gate_a_node_id, a_gate_node_id, distance)
    else:
        st.warning("Could not find Gate-A inter-building corridor connection nodes")
    
    # Gate-B 连接
    gate_b_corr_name = 'gateToB-p1'
    gate_b_node_id = graph.node_id_map.get((gate_building_id, 'corridor', gate_b_corr_name, gate_level))
    b_gate_corr_name = 'gateToB-p1'
    b_gate_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_gate_corr_name, bc_connect_level))
    
    if gate_b_node_id and b_gate_node_id:
        coords_gate = graph.nodes[gate_b_node_id]['coordinates']
        coords_b = graph.nodes[b_gate_node_id]['coordinates']
        distance = euclidean_distance(coords_gate, coords_b)
        graph.add_edge(gate_b_node_id, b_gate_node_id, distance)
    else:
        st.warning("Could not find Gate-B inter-building corridor connection nodes")
    
    # Gate-C 连接
    gate_c_corr_name = 'gateToC-p1'
    gate_c_node_id = graph.node_id_map.get((gate_building_id, 'corridor', gate_c_corr_name, gate_level))
    c_gate_corr_name = 'gateToC-p1'
    c_gate_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_gate_corr_name, connect_level1))
    
    if gate_c_node_id and c_gate_node_id:
        coords_gate = graph.nodes[gate_c_node_id]['coordinates']
        coords_c = graph.nodes[c_gate_node_id]['coordinates']
        distance = euclidean_distance(coords_gate, coords_c)
        graph.add_edge(gate_c_node_id, c_gate_node_id, distance)
    else:
        st.warning("Could not find Gate-C inter-building corridor connection nodes")

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
            
            # 过滤出重要节点（去除普通走廊节点）
            important_nodes = filter_important_nodes(path, graph)
            
            simplified_path = []
            path_stairs = set()
            
            # 为重要节点构建带第一人称视角的路径描述
            for i in range(len(important_nodes)):
                node_id = important_nodes[i]
                node_info = graph.nodes[node_id]
                node_type = node_info['type']
                node_name = node_info['name']
                node_level = node_info['level']
                node_building = node_info['building']
                
                # 构建节点描述
                if node_type == 'stair':
                    path_stairs.add((node_building, node_name, node_level))
                    node_desc = f"Building {node_building}{node_name}({node_level})"
                elif node_type == 'classroom':
                    node_desc = f"Building {node_building}{node_name}({node_level})"
                elif node_type == 'corridor':
                    # 跨建筑走廊描述
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
                    
                    # 获取上一个建筑（如果有）
                    if i > 0:
                        prev_node_info = graph.nodes[important_nodes[i-1]]
                        prev_building = prev_node_info['building']
                        node_desc = f"Cross corridor from Building {prev_building} to Building {node_building}({node_level})"
                    else:
                        node_desc = f"Cross corridor to Building {connected_building}({node_level})"
                else:
                    node_desc = ""
                
                # 计算方向
                direction = ""
                if i < len(important_nodes) - 1 and node_desc:
                    if i == 0:
                        # 起点：使用真实场景的方向判断
                        current_coords = node_info['coordinates']
                        next_node_id = important_nodes[i + 1]
                        next_coords = graph.nodes[next_node_id]['coordinates']
                        dz = next_coords[2] - current_coords[2]
                        if abs(dz) > 0.1:
                            direction = "向上" if dz > 0 else "向下"
                        else:
                            # 调用真实场景的方向判断函数
                            direction = get_real_world_start_direction(current_coords, next_coords)
                    else:
                        # 后续节点：用调整后的第一人称视角
                        prev_node_id = important_nodes[i-1]
                        prev_coords = graph.nodes[prev_node_id]['coordinates']
                        current_coords = node_info['coordinates']
                        next_node_id = important_nodes[i + 1]
                        next_coords = graph.nodes[next_node_id]['coordinates']
                        direction = get_walking_direction(prev_coords, current_coords, next_coords)
                
                # 添加方向描述
                if direction:
                    node_desc += f"（{direction}）"
                
                # 添加到简化路径
                if node_desc:
                    simplified_path.append(node_desc)
            
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
        # 获取所有建筑（包括Gate）
        buildings = []
        for b in school_data.keys():
            if b == 'gate' or b.startswith('building'):
                buildings.append(b)
        
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
    try:
        st.image("welcome_image.jpg", use_column_width=True)
    except:
        st.markdown("<p>SCIS Campus Navigation System</p>", unsafe_allow_html=True)
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
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="author-tag">Created By DANIEL HAN</div>', unsafe_allow_html=True)
    st.subheader("🏫SCIS Campus Navigation System")
    st.markdown("3D Map & Inter-building Path Planning (A/B/C/Gate Building Navigation)")

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

    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        global graph
        graph = build_navigation_graph(school_data)
        building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
        st.success("✅ Campus data loaded successfully! Initial state shows A/B/C/Gate buildings")
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return

    col1, col2 = st.columns([1, 6])

    with col1:
        st.markdown("#### 📍 Select Locations")
        
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
            help="Click to return to initial state, showing all floors (including Building B) and clearing path"
        )
        
        # 添加退出按钮，返回欢迎页面
        exit_button = st.button(
            "🚪 Exit to Welcome Page", 
            use_container_width=True,
            help="Click to return to the welcome page",
            type="secondary"  # 使用次要样式区分
        )
        
        if reset_button:
            reset_app_state()
            st.rerun()
        
        if exit_button:
            # 重置应用状态并返回欢迎页
            reset_app_state()
            st.session_state['page'] = 'welcome'
            st.rerun()

    with col2:
        st.markdown("#### 🗺️ 3D Campus Map")
        
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
