import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. 基础配置 --------------------------
plt.switch_backend('Agg')  # 解决Streamlit matplotlib渲染问题

# 定义颜色常量（区分A/C楼，统一楼层Z值颜色）
COLORS = {
    'building': {'A': 'lightblue', 'C': 'lightcoral'},  # A楼浅蓝色，C楼浅红色
    'floor_z': {-3: 'blue', 2: 'green', 7: 'orange', 12: 'purple'},  # 按Z值定义楼层边框色
    'corridor_line': {'A': 'cyan', 'C': 'salmon'},  # A楼走廊青色，C楼走廊橙红色
    'corridor_node': 'navy',
    'corridor_label': 'darkblue',
    'stair': 'red',
    'stair_label': 'darkred',
    'classroom_label': 'black',
    'path': 'darkred',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold'  # 跨楼连通走廊金色
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

# 绘制3D地图（支持A/C楼）
def plot_3d_map(school_data):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 遍历所有建筑（A楼和C楼）
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue  # 跳过非建筑节点
        building_name = building_id.replace('building', '')  # 提取"A"/"C"
        building_data = school_data[building_id]
        
        # 处理建筑内每个楼层
        for level in building_data['levels']:
            z = level['z']
            level_name = level['name']
            floor_border_color = COLORS['floor_z'].get(z, 'gray')  # 楼层边框色按Z值
            building_fill_color = COLORS['building'][building_name]  # 建筑填充色

            # 1. 绘制楼层平面（半透明填充，区分建筑）
            fp = level['floorPlane']
            # 生成楼层平面顶点（矩形）
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
            
            # 绘制边框
            ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=2, 
                    label=f"{building_name}楼-{level_name}" if f"{building_name}楼-{level_name}" not in ax.get_legend_handles_labels()[1] else "")
            # 绘制半透明填充面
            ax.plot_trisurf(x_plane[:-1], y_plane[:-1], z_plane[:-1], 
                            color=building_fill_color, alpha=0.3)

            # 2. 绘制走廊（区分普通走廊和跨楼连通走廊）
            for corr_idx, corridor in enumerate(level['corridors']):
                points = corridor['points']
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                z_coords = [p[2] for p in points]
                
                # 判断是否为跨楼连通走廊
                if 'name' in corridor and ('connectToBuilding' in corridor['name']):
                    corr_line_color = COLORS['connect_corridor']
                    corr_line_width = 6
                    corr_label = f"跨楼走廊-{building_name}楼"
                else:
                    corr_line_color = COLORS['corridor_line'][building_name]
                    corr_line_width = 4
                    corr_label = None
                
                # 绘制走廊线条
                ax.plot(x, y, z_coords, color=corr_line_color, linewidth=corr_line_width, 
                        alpha=0.8, label=corr_label if (corr_label and corr_label not in ax.get_legend_handles_labels()[1]) else "")
                
                # 标记走廊节点
                for p_idx, (px, py, pz) in enumerate(points):
                    ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=20, marker='s', alpha=0.9)
                    ax.text(px, py, pz, f'{building_name}C{corr_idx}-P{p_idx}', 
                            color=COLORS['corridor_label'], fontsize=7)

            # 3. 绘制楼梯
            for stair in level['stairs']:
                x, y, _ = stair['coordinates']
                stair_label = f"{building_name}楼-{stair['name']}"
                # 避免重复添加图例
                if stair_label not in ax.get_legend_handles_labels()[1]:
                    ax.scatter(x, y, z, color=COLORS['stair'], s=300, marker='^', label=stair_label)
                else:
                    ax.scatter(x, y, z, color=COLORS['stair'], s=300, marker='^')
                ax.text(x, y, z, stair['name'], color=COLORS['stair_label'], fontweight='bold', fontsize=9)

            # 4. 绘制教室
            for classroom in level['classrooms']:
                x, y, _ = classroom['coordinates']
                width, depth = classroom['size']
                class_name = classroom['name']

                # 教室标签
                ax.text(x, y, z, class_name, color=COLORS['classroom_label'], fontweight='bold', fontsize=8)

                # 教室位置标记点
                ax.scatter(x, y, z, color=building_fill_color, s=80, edgecolors=floor_border_color)

                # 教室边界（虚线）
                ax.plot([x, x + width, x + width, x, x],
                        [y, y, y + depth, y + depth, y],
                        [z, z, z, z, z],
                        color=floor_border_color, linestyle='--', alpha=0.6)

    # 设置坐标轴
    ax.set_xlabel('X 坐标', fontsize=12)
    ax.set_ylabel('Y 坐标', fontsize=12)
    ax.set_zlabel('楼层高度 (Z值)', fontsize=12)
    ax.set_title('校园3D导航地图（支持A/C楼跨楼导航）', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)  # 图例靠右显示
    ax.grid(True, alpha=0.3)

    return fig, ax

# 自定义图数据结构（支持多建筑）
class Graph:
    def __init__(self):
        self.nodes = {}  # 节点ID: 节点信息
        self.node_id_map = {}  # 辅助映射：(建筑,类型,名称,楼层) → 节点ID

    def add_node(self, building_id, node_type, name, level, coordinates):
        """添加节点，生成唯一ID（包含建筑标识）"""
        building_name = building_id.replace('building', '')  # 提取"A"/"C"
        # 生成唯一节点ID（格式：建筑-类型-名称@楼层）
        if node_type == 'corridor':
            # 走廊节点ID：A-Corr-C0-P0@level3（建筑-类型-走廊索引-节点索引@楼层）
            node_id = f"{building_name}-Corr-{name}@{level}"
        else:
            # 教室/楼梯ID：A-Class-A303@level3 / A-Stair-Stairs1@level3
            node_id = f"{building_name}-{node_type.capitalize()}-{name}@{level}"
        
        # 存储节点信息
        self.nodes[node_id] = {
            'building': building_name,
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        # 建立辅助映射（便于后续查找）
        map_key = (building_id, node_type, name, level)
        self.node_id_map[map_key] = node_id
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        """添加双向边"""
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（支持A/C楼及跨楼连通）
def build_navigation_graph(school_data):
    graph = Graph()

    # 第一步：添加所有建筑的节点（教室+楼梯+走廊）
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_data = school_data[building_id]
        
        for level in building_data['levels']:
            level_name = level['name']
            z = level['z']

            # 1. 添加教室节点
            for classroom in level['classrooms']:
                graph.add_node(
                    building_id=building_id,
                    node_type='classroom',
                    name=classroom['name'],
                    level=level_name,
                    coordinates=classroom['coordinates']
                )

            # 2. 添加楼梯节点
            for stair in level['stairs']:
                graph.add_node(
                    building_id=building_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

            # 3. 添加走廊节点（区分普通走廊和跨楼走廊）
            for corr_idx, corridor in enumerate(level['corridors']):
                corr_name = corridor.get('name', f'C{corr_idx}')  # 优先使用自定义名称（如connectToBuildingA）
                for p_idx, point in enumerate(corridor['points']):
                    # 走廊节点名称：普通走廊=C0-P0，跨楼走廊=connectToBuildingA-P0
                    corridor_point_name = f"{corr_name}-P{p_idx}"
                    graph.add_node(
                        building_id=building_id,
                        node_type='corridor',
                        name=corridor_point_name,
                        level=level_name,
                        coordinates=point
                    )

    # 第二步：添加所有连接关系（分建筑内和跨建筑）
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_data = school_data[building_id]
        building_name = building_id.replace('building', '')

        for level in building_data['levels']:
            level_name = level['name']
            
            # 获取当前建筑当前楼层的所有走廊节点
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]

            # 1. 同一走廊内的节点连接（沿走廊路径）
            for corr_idx, corridor in enumerate(level['corridors']):
                corr_name = corridor.get('name', f'C{corr_idx}')
                corr_points = corridor['points']
                # 连接走廊内相邻节点
                for p_idx in range(len(corr_points) - 1):
                    # 获取当前节点和下一个节点的ID
                    current_point_name = f"{corr_name}-P{p_idx}"
                    next_point_name = f"{corr_name}-P{p_idx + 1}"
                    current_node_id = graph.node_id_map.get((building_id, 'corridor', current_point_name, level_name))
                    next_node_id = graph.node_id_map.get((building_id, 'corridor', next_point_name, level_name))
                    
                    if current_node_id and next_node_id:
                        # 计算距离并添加边
                        coords1 = graph.nodes[current_node_id]['coordinates']
                        coords2 = graph.nodes[next_node_id]['coordinates']
                        distance = euclidean_distance(coords1, coords2)
                        graph.add_edge(current_node_id, next_node_id, distance)

            # 2. 不同走廊间的节点连接（距离<3视为交叉点）
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2)
                    
                    if distance < 3.0:  # 距离阈值，视为可互通
                        graph.add_edge(node1_id, node2_id, distance)

            # 3. 教室 → 最近的走廊节点连接
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
                
                # 查找最近的走廊节点
                for corr_node_id in corr_nodes:
                    corr_coords = graph.nodes[corr_node_id]['coordinates']
                    dist = euclidean_distance(class_coords, corr_coords)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(class_node_id, nearest_corr_node_id, min_dist)

            # 4. 楼梯 → 最近的走廊节点连接
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
                
                # 查找最近的走廊节点
                for corr_node_id in corr_nodes:
                    corr_coords = graph.nodes[corr_node_id]['coordinates']
                    dist = euclidean_distance(stair_coords, corr_coords)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(stair_node_id, nearest_corr_node_id, min_dist)

        # 5. 建筑内跨楼层连接（楼梯节点之间，读取connections配置）
        for connection in building_data['connections']:
            from_obj_name, from_level = connection['from']
            to_obj_name, to_level = connection['to']
            
            # 判断连接对象类型（楼梯或跨楼走廊）
            from_obj_type = 'stair' if from_obj_name.startswith('Stairs') else 'corridor'
            to_obj_type = 'stair' if to_obj_name.startswith('Stairs') else 'corridor'
            
            # 处理走廊节点名称（跨楼走廊需添加-P0，默认取第一个节点）
            if from_obj_type == 'corridor':
                from_obj_name = f"{from_obj_name}-P0"
            if to_obj_type == 'corridor':
                to_obj_name = f"{to_obj_name}-P0"
            
            # 获取连接的两个节点ID
            from_node_id = graph.node_id_map.get((building_id, from_obj_type, from_obj_name, from_level))
            to_node_id = graph.node_id_map.get((building_id, to_obj_type, to_obj_name, to_level))
            
            if from_node_id and to_node_id:
                # 跨楼层连接权重固定为5.0（模拟楼梯通行成本）
                graph.add_edge(from_node_id, to_node_id, 5.0)

    # 6. 跨建筑连接（A楼和C楼的连通走廊节点）
    # 查找A楼level3的跨楼走廊最后一个节点和C楼level3的跨楼走廊第一个节点
    a_building_id = 'buildingA'
    c_building_id = 'buildingC'
    connect_level = 'level3'  # 跨楼连接在三楼
    
    # A楼连接C楼的走廊最后一个节点（P2）
    a_corr_name = 'connectToBuildingC-P2'
    a_connect_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr_name, connect_level))
    
    # C楼连接A楼的走廊第一个节点（P0）
    c_corr_name = 'connectToBuildingA-P0'
    c_connect_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr_name, connect_level))
    
    if a_connect_node_id and c_connect_node_id:
        # 计算跨楼节点之间的距离并连接
        coords_a = graph.nodes[a_connect_node_id]['coordinates']
        coords_c = graph.nodes[c_connect_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_c)
        graph.add_edge(a_connect_node_id, c_connect_node_id, distance)

    return graph

# Dijkstra算法
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
    return path if len(path) > 1 else None  # 确保路径有效

# 导航函数（支持跨建筑导航）
def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    try:
        # 生成起点和终点节点ID
        start_node = f"{start_building}-Class-{start_classroom}@{start_level}"
        end_node = f"{end_building}-Class-{end_classroom}@{end_level}"

        if start_node not in graph.nodes or end_node not in graph.nodes:
            return None, "无效的教室或楼层（节点未找到）", None

        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)

        if path:
            total_distance = distances[end_node]
            # 简化路径显示（只保留教室和楼梯）
            simplified_path = []
            for node_id in path:
                node_type = graph.nodes[node_id]['type']
                node_name = graph.nodes[node_id]['name']
                node_level = graph.nodes[node_id]['level']
                node_building = graph.nodes[node_id]['building']
                if node_type in ['classroom', 'stair']:
                    simplified_path.append(f"{node_building}楼-{node_name}（楼层: {node_level}）")
            return path, f"总距离: {total_distance:.2f} 单位", simplified_path
        else:
            return None, "两个教室之间没有可用路径", None
    except Exception as e:
        return None, f"导航错误: {str(e)}", None

# 在3D图上绘制路径
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

        # 绘制完整路径
        ax.plot(x, y, z, color=COLORS['path'], linewidth=3, linestyle='-', marker='o', markersize=5)

        # 标记起点和终点
        ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=500, marker='*', label='起点', edgecolors='black')
        ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=500, marker='*', label='终点', edgecolors='black')
        
        # 添加起点终点标签
        ax.text(x[0], y[0], z[0], f"起点\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=10)
        ax.text(x[-1], y[-1], z[-1], f"终点\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=10)

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    except Exception as e:
        st.error(f"绘制路径失败: {str(e)}")

# 获取所有建筑、楼层和教室信息
def get_classroom_info(school_data):
    try:
        buildings = [b for b in school_data.keys() if b.startswith('building')]
        building_names = [b.replace('building', '') for b in buildings]  # 提取"A"/"C"
        
        # 按建筑组织信息：{建筑: {楼层: [教室列表]}}
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

# -------------------------- 3. Streamlit界面逻辑 --------------------------
def main():
    # 页面标题和数据加载
    st.title("🏫 校园导航系统")
    st.subheader("3D地图与跨楼路径规划")

    # 加载JSON数据
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        nav_graph = build_navigation_graph(school_data)
        building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
        st.success("✅ 校园数据加载成功！")
    except Exception as e:
        st.error(f"初始化错误: {str(e)}")
        return

    # 布局：左右分栏
    col1, col2 = st.columns([1, 2])

    with col1:
        # 左侧：起点和终点选择（支持选择建筑）
        st.markdown("### 📍 选择位置")
        
        # 起点选择
        st.markdown("#### 起点")
        start_building = st.selectbox("建筑", building_names, key="start_building")
        start_levels = levels_by_building.get(start_building, [])
        start_level = st.selectbox("楼层", start_levels, key="start_level")
        start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
        start_classroom = st.selectbox("教室", start_classrooms, key="start_classroom")

        # 终点选择
        st.markdown("#### 终点")
        end_building = st.selectbox("建筑", building_names, key="end_building")
        end_levels = levels_by_building.get(end_building, [])
        end_level = st.selectbox("楼层", end_levels, key="end_level")
        end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
        end_classroom = st.selectbox("教室", end_classrooms, key="end_classroom")

        # 导航按钮
        nav_button = st.button("🔍 查找最短路径", use_container_width=True)

    with col2:
        # 右侧：显示3D地图和导航结果
        st.markdown("### 🗺️ 3D校园地图")
        
        # 初始显示3D地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        # 点击导航按钮后，计算路径并更新地图
        if nav_button:
            try:
                path, message, simplified_path = navigate(
                    nav_graph, 
                    start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                
                # 显示导航结果
                if path:
                    st.success(f"📊 导航结果: {message}")
                    # 显示简化路径详情
                    st.markdown("#### 🛤️ 路径详情（教室和楼梯）")
                    for i, step in enumerate(simplified_path, 1):
                        st.write(f"{i}. {step}")
                    
                    # 重新绘制带路径的3D图
                    fig, ax = plot_3d_map(school_data)
                    plot_path(ax, nav_graph, path)
                    st.session_state['fig'] = fig
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"导航过程出错: {str(e)}")
        
        # 显示3D图
        try:
            st.pyplot(st.session_state['fig'])
        except Exception as e:
            st.error(f"显示地图失败: {str(e)}")

# -------------------------- 4. 运行主函数 --------------------------
if __name__ == "__main__":
    main()
    
