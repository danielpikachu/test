import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. 基本配置 --------------------------
plt.switch_backend('Agg')  # 解决Streamlit matplotlib渲染问题

# 定义颜色常量：包含A/B/C楼
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral'},  # 建筑填充色
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},  # 楼层边框色
    'corridor_line': {'A': 'cyan', 'B': 'forestgreen', 'C': 'salmon'},  # 走廊线条色
    'corridor_node': 'navy',
    'corridor_label': 'darkblue',
    # 楼梯颜色配置（包含B楼楼梯）
    'stair': {
        'Stairs1': '#FF5733',   # A楼 - 橙红
        'Stairs2': '#33FF57',   # C楼 - 绿
        'Stairs3': '#3357FF',   # C楼 - 蓝
        'Stairs4': '#FF33F5',   # C楼 - 粉紫
        'Stairs5': '#F5FF33',   # C楼 - 黄
        'StairsB1': '#33FFF5',  # B楼 - 青
        'StairsB2': '#FF9933',  # B楼 - 橙
    },
    'stair_label': 'darkred',  # 楼梯标签颜色
    'classroom_label': 'black',
    'path': 'darkred',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold',
    'building_label': {'A': 'darkblue', 'B': 'darkgreen', 'C': 'darkred'}  # 楼宇标签颜色
}

# -------------------------- 2. 核心功能实现 --------------------------
# 读取JSON数据
def load_school_data_detailed(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载数据文件失败: {str(e)}")
        return None

# 绘制3D地图 - B楼在初始状态显示，路径规划时隐藏
def plot_3d_map(school_data, display_options=None):
    # 放大图形尺寸
    fig = plt.figure(figsize=(35, 30))
    ax = fig.add_subplot(111, projection='3d')

    # 放大坐标轴刻度标签
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # 默认显示所有内容（包括B楼）
    if display_options is None:
        display_options = {
            'start_level': None,
            'end_level': None,
            'path_stairs': set(),
            'show_all': True,
            'path': [],
            'start_building': None,  # 起点楼宇信息
            'end_building': None     # 终点楼宇信息
        }
    
    show_all = display_options['show_all']
    start_level = display_options['start_level']
    end_level = display_options['end_level']
    path_stairs = display_options['path_stairs']
    path = display_options.get('path', [])
    start_building = display_options.get('start_building')
    end_building = display_options.get('end_building')

    # 存储每栋楼的标识位置信息
    building_label_positions = {}

    # 遍历所有建筑物
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        # 路径规划状态下隐藏B楼
        if not show_all and building_name == 'B':
            continue
        
        building_data = school_data[building_id]
        
        # 记录建筑物的显示楼层中最高楼层和最大Y值
        displayed_levels = []  # 存储显示的楼层
        max_displayed_z = -float('inf')
        max_displayed_y = -float('inf')
        corresponding_x = 0  # 最大Y值对应的X坐标
        level_count = 0
        
        # 处理建筑物的每个楼层
        for level in building_data['levels']:
            level_name = level['name']
            z = level['z']
            
            # 判断是否需要显示当前楼层
            show_level = show_all
            if not show_all:
                # 只显示起点楼层、终点楼层
                show_level = (level_name == start_level) or (level_name == end_level)
            
            # 如果楼层会被显示，记录相关信息
            if show_level:
                displayed_levels.append(level)
                if z > max_displayed_z:
                    max_displayed_z = z
                
                # 获取楼层平面信息
                fp = level['floorPlane']
                current_max_y = fp['maxY']
                if current_max_y > max_displayed_y:
                    max_displayed_y = current_max_y
                    corresponding_x = (fp['minX'] + fp['maxX']) / 2
            
            level_count += 1
            
            # 适配各楼楼层颜色
            floor_border_color = COLORS['floor_z'].get(z, 'gray')
            building_fill_color = COLORS['building'].get(building_name, 'lightgray')

            # 绘制楼层平面（仅在需要显示的楼层）
            if show_level:
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
                
                # 避免重复添加图例
                legend_label = f"Building {building_name}-{level_name}"
                if legend_label not in ax.get_legend_handles_labels()[1]:
                    ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=4, label=legend_label)
                else:
                    ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=4)
                ax.plot_trisurf(x_plane[:-1], y_plane[:-1], z_plane[:-1], 
                                color=building_fill_color, alpha=0.3)

                # 绘制走廊（新增外部走廊判断，读取JSON样式）
                for corr_idx, corridor in enumerate(level['corridors']):
                    points = corridor['points']
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    z_coords = [p[2] for p in points]
                    
                    # 1. 优先判断是否为外部走廊（读取JSON的type和style）
                    is_external = corridor.get('type') == 'external'
                    if is_external:
                        # 读取JSON中的样式，无配置则用默认灰色虚线
                        ext_style = corridor.get('style', {})
                        corr_line_color = ext_style.get('color', 'gray')  # 优先JSON颜色，默认灰色
                        corr_line_style = ext_style.get('lineType', '--')  # 优先JSON线型，默认虚线
                        corr_line_width = 10  # 外部走廊线宽放大，突出显示
                        corr_label = f"External Corridor ({building_name}-{corridor.get('name', f'corr{corr_idx}')})"
                    
                    # 2. 非外部走廊：判断是否为连廊
                    elif 'name' in corridor and ('connectToBuilding' in corridor['name']):
                        corr_line_color = COLORS['connect_corridor']
                        corr_line_style = '-'  # 连廊用实线
                        corr_line_width = 12  # 放大楼宇间走廊线宽
                        corr_label = f"Connect Corridor ({building_name}-{level_name})"
                    
                    # 3. 普通内部走廊
                    else:
                        corr_line_color = COLORS['corridor_line'].get(building_name, 'gray')
                        corr_line_style = '-'  # 内部走廊用实线
                        corr_line_width = 8  # 放大普通走廊线宽
                        corr_label = None
                    
                    # 绘制走廊（添加linestyle参数，应用虚线样式）
                    if corr_label and corr_label not in ax.get_legend_handles_labels()[1]:
                        ax.plot(x, y, z_coords, 
                                color=corr_line_color, 
                                linestyle=corr_line_style,  # 应用线型（虚线/实线）
                                linewidth=corr_line_width, 
                                alpha=0.8, 
                                label=corr_label)
                    else:
                        ax.plot(x, y, z_coords, 
                                color=corr_line_color, 
                                linestyle=corr_line_style,  # 应用线型（虚线/实线）
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
                    # 教室边框
                    ax.plot([x, x + width, x + width, x, x],
                            [y, y, y + depth, y + depth, y],
                            [z, z, z, z, z],
                            color=floor_border_color, linestyle='--', alpha=0.6, linewidth=2)

            # 绘制楼梯
            for stair in level['stairs']:
                stair_name = stair['name']
                # 检查是否是路径中经过的楼梯
                is_path_stair = (building_name, stair_name, level_name) in path_stairs
                
                if show_all or show_level or is_path_stair:
                    x, y, _ = stair['coordinates']
                    stair_label = f"Building {building_name}-{stair_name}"
                    
                    # 适配各楼楼梯颜色
                    stair_color = COLORS['stair'].get(stair_name, 'red')
                    
                    # 为路径中的楼梯使用更醒目的样式
                    marker_size = 800 if is_path_stair else 600
                    marker_edge_width = 3 if is_path_stair else 1
                    
                    # 避免重复添加图例
                    if stair_label not in ax.get_legend_handles_labels()[1]:
                        ax.scatter(x, y, z, color=stair_color, s=marker_size, marker='^', 
                                  label=stair_label, edgecolors='black', linewidths=marker_edge_width)
                    else:
                        ax.scatter(x, y, z, color=stair_color, s=marker_size, marker='^',
                                  edgecolors='black', linewidths=marker_edge_width)
                    
                    # 楼梯标签
                    ax.text(x, y, z, stair_name, color=COLORS['stair_label'], fontweight='bold', fontsize=14)
        
        # 存储楼宇标识位置
        if level_count > 0 and len(displayed_levels) > 0:
            # 对于A楼和C楼，在路径规划状态下将标签放在显示的最高楼层的最大Y轴旁边
            if (building_name in ['A', 'C']) and (not show_all):
                # 获取显示的最高楼层的floorPlane数据
                highest_displayed_level = None
                for level in displayed_levels:
                    if level['z'] == max_displayed_z:
                        highest_displayed_level = level
                        break
                
                if highest_displayed_level:
                    fp = highest_displayed_level['floorPlane']
                    # 计算最高显示楼层的中心X坐标
                    center_x = (fp['minX'] + fp['maxX']) / 2
                    # 根据楼宇调整Y轴偏移方向（A楼向右，C楼向左）
                    y_offset = 2.0 if building_name == 'A' else -2.0
                    label_y = max_displayed_y + y_offset
                    label_z = max_displayed_z  # 标签Z值与最高显示楼层一致
            else:
                # B楼和全显示状态下的位置
                if building_name == 'B':
                    label_y = max_displayed_y - 2.0  # B楼标签放在Y外侧（更负的位置）
                else:
                    label_y = max_displayed_y + 2.0  # 其他楼标签放在Y外侧
                label_z = max_displayed_z + 1.0
                center_x = corresponding_x
            
            building_label_positions[building_name] = (center_x, label_y, label_z)

    # 添加楼宇标识
    for building_name, (x, y, z) in building_label_positions.items():
        # 路径规划状态下不显示B楼标签
        if not show_all and building_name == 'B':
            continue
            
        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", 
                        facecolor=COLORS['building'].get(building_name, 'lightgray'), alpha=0.7)
        ax.text(
            x, y, z, 
            f"{building_name}楼", 
            color=COLORS['building_label'].get(building_name, 'black'), 
            fontweight='bold', 
            fontsize=30,
            ha='center', 
            va='center', 
            bbox=bbox_props
        )

    # 绘制路径（当有路径且不显示全部时）
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

            # 路径线条
            ax.plot(x, y, z, color=COLORS['path'], linewidth=6, linestyle='-', marker='o', markersize=10, label='导航路径')
            # 起点标记
            ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=1000, marker='*', label='起点', edgecolors='black')
            # 终点标记
            ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=1000, marker='*', label='终点', edgecolors='black')
            # 起点标签
            ax.text(x[0], y[0], z[0], f"起点\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=16)
            # 终点标签
            ax.text(x[-1], y[-1], z[-1], f"终点\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=16)
        except Exception as e:
            st.warning(f"路径绘制警告: {str(e)}")

    # 轴标签和标题
    ax.set_xlabel('X坐标', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y坐标', fontsize=18, fontweight='bold')
    ax.set_zlabel('楼层高度 (Z值)', fontsize=18, fontweight='bold')
    ax.set_title('校园3D导航地图 (A/C楼导航，B楼初始可见)', fontsize=24, fontweight='bold', pad=20)
    
    # 图例
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=2)

    return fig, ax

# 自定义图数据结构
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}  # 多重映射关系

    def add_node(self, building_id, node_type, name, level, coordinates):
        building_name = building_id.replace('building', '')
        
        # 路径规划只处理A/C楼节点，B楼节点不添加到导航图
        if building_name == 'B':
            return None
        
        # 简洁的节点ID
        if node_type == 'corridor':
            node_id = f"{building_name}-corr-{name}@{level}"
        else:
            node_id = f"{building_name}-{node_type}-{name}@{level}"
        
        # 存储节点信息
        self.nodes[node_id] = {
            'building': building_name,
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        
        # 建立多重映射
        map_key = (building_id, node_type, name, level)
        self.node_id_map[map_key] = node_id
        # 为教室添加额外映射
        if node_type == 'classroom':
            class_key = (building_name, name, level)
            self.node_id_map[class_key] = node_id
            
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（只处理A/C楼）
def build_navigation_graph(school_data):
    graph = Graph()

    # 步骤1: 添加所有建筑物节点（教室、楼梯、走廊）- 只处理A/C楼
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        # 跳过B楼节点
        if building_name == 'B':
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

            # 2. 添加楼梯节点（只处理A/C楼楼梯）
            for stair in level['stairs']:
                # 过滤掉B楼楼梯
                if stair['name'].startswith('StairsB'):
                    continue
                graph.add_node(
                    building_id=building_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

            # 3. 添加走廊节点
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

    # 步骤2: 添加所有连接关系 - 只处理A/C楼
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        # 跳过B楼
        if building_name == 'B':
            continue
        
        building_data = school_data[building_id]

        for level in building_data['levels']:
            level_name = level['name']
            
            # 获取当前建筑物和楼层的所有走廊节点
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]

            # 1. 连接同一走廊内的节点
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

            # 2. 连接不同走廊之间的节点
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2)
                    
                    if distance < 3.0:  # 距离较近的走廊节点认为是连通的
                        graph.add_edge(node1_id, node2_id, distance)

            # 3. 连接教室到最近的走廊节点
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
                    st.warning(f"警告: 建筑物 {building_name}{level_name} 中的教室 {graph.nodes[class_node_id]['name']} 没有走廊连接")

            # 4. 连接楼梯到最近的走廊节点
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

        # 5. 连接同一建筑物内不同楼层的节点
        for connection in building_data['connections']:
            from_obj_name, from_level = connection['from']
            to_obj_name, to_level = connection['to']
            
            # 过滤B楼相关连接
            if 'B' in from_obj_name or 'B' in to_obj_name:
                continue
                
            from_obj_type = 'stair' if from_obj_name.startswith('Stairs') else 'corridor'
            to_obj_type = 'stair' if to_obj_name.startswith('Stairs') else 'corridor'
            
            if from_obj_type == 'corridor':
                from_obj_name = f"{from_obj_name}-p0"
            if to_obj_type == 'corridor':
                to_obj_name = f"{to_obj_name}-p0"
            
            from_node_id = graph.node_id_map.get((building_id, from_obj_type, from_obj_name, from_level))
            to_node_id = graph.node_id_map.get((building_id, to_obj_type, to_obj_name, to_level))
            
            if from_node_id and to_node_id:
                graph.add_edge(from_node_id, to_node_id, 5.0)  # 楼梯连接权重固定为5

    # 6. 连接A/C楼之间的节点
    a_building_id = 'buildingA'
    c_building_id = 'buildingC'
    
    # level1楼宇间连接
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
        st.warning("未找到level1楼宇间走廊连接节点")
    
    # level3楼宇间连接
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
        st.warning("未找到level3楼宇间走廊连接节点")

    return graph

# 迪杰斯特拉算法
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

# 生成最短路径
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path if len(path) > 1 else None

# 导航功能（只支持A/C楼）
def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    # 验证建筑物是否有效（只允许A/C楼）
    valid_buildings = ['A', 'C']
    if start_building not in valid_buildings or end_building not in valid_buildings:
        return None, "无效的建筑物选择，只支持A楼和C楼", None, None
        
    try:
        # 使用多重映射查找节点
        start_key = (start_building, start_classroom, start_level)
        end_key = (end_building, end_classroom, end_level)
        
        start_node = graph.node_id_map.get(start_key)
        end_node = graph.node_id_map.get(end_key)
        
        # 备选方案：构造节点ID
        if not start_node:
            start_node = f"{start_building}-classroom-{start_classroom}@{start_level}"
        if not end_node:
            end_node = f"{end_building}-classroom-{end_classroom}@{end_level}"

        # 验证节点是否存在
        if start_node not in graph.nodes:
            return None, f"起始教室不存在: {start_building}{start_classroom}@{start_level}", None, None
        if end_node not in graph.nodes:
            return None, f"目标教室不存在: {end_building}{end_classroom}@{end_level}", None, None

        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)

        if path:
            total_distance = distances[end_node]
            simplified_path = []
            # 收集路径中经过的楼梯
            path_stairs = set()
            # 跟踪上一个节点的建筑物
            prev_building = None
            
            for node_id in path:
                node_type = graph.nodes[node_id]['type']
                node_name = graph.nodes[node_id]['name']
                node_level = graph.nodes[node_id]['level']
                node_building = graph.nodes[node_id]['building']
                
                # 记录路径中经过的楼梯
                if node_type == 'stair':
                    path_stairs.add((node_building, node_name, node_level))
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
                
                # 处理教室节点
                elif node_type == 'classroom':
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
                
                # 处理走廊节点，检测是否是连廊
                elif node_type == 'corridor':
                    # 检查是否是连接两栋楼的连廊
                    if 'connectToBuilding' in node_name:
                        # 确定连廊连接的建筑物
                        if 'connectToBuildingA' in node_name:
                            connected_building = 'A'
                        elif 'connectToBuildingC' in node_name:
                            connected_building = 'C'
                        else:
                            connected_building = '其他'
                            
                        # 只有当建筑物发生变化时才添加连廊信息
                        if prev_building and prev_building != node_building:
                            simplified_path.append(f"通过连廊从Building {prev_building}到Building {node_building}({node_level})")
                
                # 更新上一个节点的建筑物
                if node_type in ['classroom', 'stair', 'corridor']:
                    prev_building = node_building
            
            full_path_str = " → ".join(simplified_path)
            # 返回显示选项，路径规划时隐藏B楼
            display_options = {
                'start_level': start_level,
                'end_level': end_level,
                'path_stairs': path_stairs,
                'show_all': False,
                'path': path,
                'start_building': start_building,
                'end_building': end_building
            }
            return path, f"总距离: {total_distance:.2f} 单位", full_path_str, display_options
        else:
            return None, "两个教室之间没有可用路径", None, None
    except Exception as e:
        return None, f"导航错误: {str(e)}", None, None

# 在3D地图上绘制路径
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

        # 路径线条
        ax.plot(x, y, z, color=COLORS['path'], linewidth=6, linestyle='-', marker='o', markersize=10)
        # 起点和终点标记
        ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=1000, marker='*', label='起点', edgecolors='black')
        ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=1000, marker='*', label='终点', edgecolors='black')
        ax.text(x[0], y[0], z[0], f"起点\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=16)
        ax.text(x[-1], y[-1], z[-1], f"终点\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=16)

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    except Exception as e:
        st.error(f"绘制路径失败: {str(e)}")

# 获取所有建筑物、楼层和教室信息（UI只显示A/C楼）
def get_classroom_info(school_data):
    try:
        # UI只显示A/C楼
        buildings = [b for b in school_data.keys() if b.startswith('building') and b.replace('building', '') in ['A', 'C']]
        building_names = [b.replace('building', '') for b in buildings]
        
        classrooms_by_building = {}
        levels_by_building = {}
        
        for building_id in buildings:
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
        st.error(f"获取教室信息失败: {str(e)}")
        return [], {}, {}

# 重置应用状态到初始状态（显示B楼）
def reset_app_state():
    st.session_state['display_options'] = {
        'start_level': None,
        'end_level': None,
        'path_stairs': set(),
        'show_all': True,  # 重置后显示所有楼宇，包括B楼
        'path': [],
        'start_building': None,
        'end_building': None
    }
    st.session_state['current_path'] = None
    # 清除路径结果显示
    if 'path_result' in st.session_state:
        del st.session_state['path_result']

# -------------------------- 3. Streamlit界面逻辑 --------------------------
def main():
    # 调整边距
    st.markdown("""
        <style>
            .block-container {
                padding-left: 1rem;    /* 减少左 margin */
                padding-right: 1rem;   /* 减少右 margin */
                max-width: 100%;       /* 移除最大宽度限制 */
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.subheader("🏫 校园导航系统")
    st.markdown("3D地图 & 楼宇间路径规划 (A/C楼导航，B楼初始可见)")

    # 初始化会话状态变量
    if 'display_options' not in st.session_state:
        st.session_state['display_options'] = {
            'start_level': None,
            'end_level': None,
            'path_stairs': set(),
            'show_all': True,  # 初始状态显示所有楼宇，包括B楼
            'path': [],
            'start_building': None,
            'end_building': None
        }
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = None

    # 加载JSON数据
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        # 全局变量graph，供plot_3d_map使用
        global graph
        graph = build_navigation_graph(school_data)
        building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
        st.success("✅ 校园数据加载成功! 初始状态显示A/B/C楼，路径规划时仅显示A/C楼")
    except Exception as e:
        st.error(f"初始化错误: {str(e)}")
        return

    # 布局调整: 左侧1/3为交互界面，右侧2/3为地图
    col1, col2 = st.columns([1, 5])

    with col1:
        st.markdown("#### 📍 选择位置")
        
        # 起点选择（只显示A/C楼）
        st.markdown("#### 起点")
        start_building = st.selectbox("建筑物", building_names, key="start_building")
        start_levels = levels_by_building.get(start_building, [])
        start_level = st.selectbox("楼层", start_levels, key="start_level")
        start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
        start_classroom = st.selectbox("教室", start_classrooms, key="start_classroom")

        # 终点选择（只显示A/C楼）
        st.markdown("#### 终点")
        end_building = st.selectbox("建筑物", building_names, key="end_building")
        end_levels = levels_by_building.get(end_building, [])
        end_level = st.selectbox("楼层", end_levels, key="end_level")
        end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
        end_classroom = st.selectbox("教室", end_classrooms, key="end_classroom")

        # 导航按钮和重置按钮
        nav_button = st.button("🔍 查找最短路径", use_container_width=True)
        
        # 添加重置视图按钮（重置后显示B楼）
        reset_button = st.button(
            "🔄 重置视图", 
            use_container_width=True,
            help="点击恢复到初始状态，显示所有楼层（包括B楼）并清除路径"
        )
        
        # 处理重置按钮点击
        if reset_button:
            reset_app_state()
            st.rerun()  # 重新运行应用以刷新界面

    with col2:
        st.markdown("#### 🗺️ 3D校园地图")
        
        # 处理导航按钮点击（路径规划时隐藏B楼）
        if nav_button:
            try:
                path, message, simplified_path, display_options = navigate(
                    graph, 
                    start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                
                if path and display_options:
                    st.success(f"📊 导航结果: {message}")
                    st.markdown("##### 🛤️ 路径详情")
                    st.info(simplified_path)
                    
                    # 保存路径和显示选项到会话状态（此时show_all=False，B楼隐藏）
                    st.session_state['current_path'] = path
                    st.session_state['display_options'] = display_options
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"导航过程错误: {str(e)}")
        
        # 绘制地图
        try:
            # 如果有路径规划结果，使用保存的显示选项（B楼隐藏）
            if st.session_state['current_path'] is not None:
                fig, ax = plot_3d_map(school_data, st.session_state['display_options'])
                # 绘制路径
                plot_path(ax, graph, st.session_state['current_path'])
            else:
                # 初始状态和重置后显示全部楼层（包括B楼）
                fig, ax = plot_3d_map(school_data)
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"显示地图失败: {str(e)}")

if __name__ == "__main__":
    main()
