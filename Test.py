import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. 基础配置：解决Streamlit matplotlib渲染问题 --------------------------
plt.switch_backend('Agg')

# -------------------------- 2. 核心功能：数据读取、3D绘图、路径计算 --------------------------
# 读取JSON数据
def load_school_data_detailed(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# 绘制3D地图（返回fig用于Streamlit显示）
def plot_3d_map(school_data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同楼层使用不同颜色
    floor_colors = {0: 'blue', 2: 'green', 5: 'orange'}  

    # 处理每个楼层
    for level in school_data['buildingA']['levels']:
        z = level['z']
        color = floor_colors.get(z, 'gray')

        # 收集当前楼层所有走廊的坐标点（用于计算平面范围）
        all_corridor_points = []
        for corridor in level['corridors']:
            all_corridor_points.extend(corridor['points'])
        if not all_corridor_points:  # 避免无走廊时报错
            continue  

        # 计算平面的X/Y轴范围（取走廊坐标的最大/最小值）
        xs = [p[0] for p in all_corridor_points]
        ys = [p[1] for p in all_corridor_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 构造平面的4个顶点（闭合矩形）
        plane_vertices = [
            [min_x, min_y, z], [max_x, min_y, z], 
            [max_x, max_y, z], [min_x, max_y, z], [min_x, min_y, z]
        ]
        x_plane = [p[0] for p in plane_vertices]
        y_plane = [p[1] for p in plane_vertices]
        z_plane = [p[2] for p in plane_vertices]

        # 绘制楼层平面边框
        ax.plot(x_plane, y_plane, z_plane, color=color, linewidth=2, label=level['name'])

        # 绘制走廊（加粗显示，突出路径必经区域）
        for corridor in level['corridors']:
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            ax.plot(x, y, z_coords, color=color, linewidth=5, alpha=0.8)

        # 绘制楼梯
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stairs' if z == 0 else "")
            ax.text(x, y, z+0.1, stair['name'], color='red', fontweight='bold')  # 楼梯标签

        # 绘制教室（用立方体表示）
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']

            # 教室标签
            ax.text(x, y, z, classroom['name'], color='black', fontweight='bold')
            # 教室位置点
            ax.scatter(x, y, z, color=color, s=50)
            # 教室边界（虚线）
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color=color, linestyle='--')

    # 设置坐标轴
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Floor')
    ax.set_title('School 3D Map (Must Pass Corridor First)')
    ax.legend()

    return fig, ax

# 自定义图数据结构（支持节点类型区分）
class Graph:
    def __init__(self):
        self.nodes = {}  # key: node_id, value: node_info（含type/coordinates/neighbors等）

    def add_node(self, node_id, node_type, name, level, coordinates):
        """添加节点：node_type支持'classroom'/'corridor'/'stair'"""
        self.nodes[node_id] = {
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}  # key: neighbor_node_id, value: weight(distance)
        }

    def add_edge(self, node1, node2, weight):
        """添加双向边（确保节点存在）"""
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1]['neighbors'][node2] = weight
            self.nodes[node2]['neighbors'][node1] = weight

# 计算欧氏距离（两点间直线距离）
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（核心修改：强制教室仅连接走廊，删除教室间直接连接）
def build_navigation_graph(school_data):
    graph = Graph()

    # -------------------------- 步骤1：添加所有节点（教室、楼梯、走廊） --------------------------
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']  # 楼层高度（Z轴坐标）

        # 1.1 添加教室节点（ID格式：教室名@楼层名，如"Class1@Floor0"）
        for classroom in level['classrooms']:
            node_id = f"{classroom['name']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='classroom',
                name=classroom['name'],
                level=level_name,
                coordinates=classroom['coordinates']
            )

        # 1.2 添加楼梯节点（ID格式：楼梯名@楼层名，如"Stair1@Floor0"）
        for stair in level['stairs']:
            node_id = f"{stair['name']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='stair',
                name=stair['name'],
                level=level_name,
                coordinates=stair['coordinates']
            )

        # 1.3 添加走廊节点（按走廊的每个坐标点创建，ID格式：corridor_X_Y_Z，确保唯一性）
        for corridor_idx, corridor in enumerate(level['corridors']):
            for point_idx, point in enumerate(corridor['points']):
                # 走廊节点ID：包含坐标和楼层，避免不同楼层走廊节点冲突
                node_id = f"corridor_{point[0]}_{point[1]}_{z}"
                # 仅添加不存在的走廊节点（避免重复）
                if node_id not in graph.nodes:
                    graph.add_node(
                        node_id=node_id,
                        node_type='corridor',
                        name=f"Corridor_{corridor_idx+1}_Point_{point_idx+1}",
                        level=level_name,
                        coordinates=point
                    )

    # -------------------------- 步骤2：添加边（核心逻辑：强制教室仅连走廊，楼梯仅连走廊） --------------------------
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 2.1 教室 ↔ 走廊：教室仅能连接到同一楼层的走廊（强制第一步进走廊）
        for classroom in level['classrooms']:
            classroom_node_id = f"{classroom['name']}@{level_name}"
            # 遍历当前楼层所有走廊节点，计算距离并添加边
            for node_id in graph.nodes:
                node = graph.nodes[node_id]
                if node['type'] == 'corridor' and node['level'] == level_name:
                    # 计算教室到走廊节点的距离（作为边的权重）
                    distance = euclidean_distance(classroom['coordinates'], node['coordinates'])
                    graph.add_edge(classroom_node_id, node_id, distance)

        # 2.2 楼梯 ↔ 走廊：楼梯仅能连接到同一楼层的走廊（跨楼层需通过楼梯节点）
        for stair in level['stairs']:
            stair_node_id = f"{stair['name']}@{level_name}"
            # 遍历当前楼层所有走廊节点，计算距离并添加边
            for node_id in graph.nodes:
                node = graph.nodes[node_id]
                if node['type'] == 'corridor' and node['level'] == level_name:
                    distance = euclidean_distance(stair['coordinates'], node['coordinates'])
                    graph.add_edge(stair_node_id, node_id, distance)

        # 2.3 走廊 ↔ 走廊：同一楼层的走廊节点间相互连接（模拟走廊通路）
        # 先收集当前楼层所有走廊节点
        corridor_nodes = [
            node_id for node_id in graph.nodes 
            if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
        ]
        # 走廊节点间两两连接（按距离权重）
        for i in range(len(corridor_nodes)):
            for j in range(i + 1, len(corridor_nodes)):
                node1 = corridor_nodes[i]
                node2 = corridor_nodes[j]
                distance = euclidean_distance(
                    graph.nodes[node1]['coordinates'], 
                    graph.nodes[node2]['coordinates']
                )
                graph.add_edge(node1, node2, distance)

    # 2.4 楼梯 ↔ 楼梯：跨楼层连接（基于JSON中的connections配置）
    for connection in school_data['buildingA']['connections']:
        from_stair_name, from_level = connection['from']
        to_stair_name, to_level = connection['to']
        # 生成跨楼层的楼梯节点ID
        from_stair_node = f"{from_stair_name}@{from_level}"
        to_stair_node = f"{to_stair_name}@{to_level}"
        # 验证节点存在后添加边（跨楼层权重设为1.0，模拟楼梯通行成本）
        if from_stair_node in graph.nodes and to_stair_node in graph.nodes:
            graph.add_edge(from_stair_node, to_stair_node, 1.0)

    return graph

# 自定义Dijkstra算法（计算最短路径）
def dijkstra(graph, start_node):
    # 初始化距离：起点为0，其他为无穷大
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    # 记录前驱节点（用于回溯路径）
    previous_nodes = {node: None for node in graph.nodes}
    # 未访问节点集合
    unvisited_nodes = set(graph.nodes.keys())

    while unvisited_nodes:
        # 选择当前距离最小的节点
        current_node = min(unvisited_nodes, key=lambda x: distances[x])
        unvisited_nodes.remove(current_node)

        # 若当前节点距离为无穷大，说明无通路，终止
        if distances[current_node] == float('inf'):
            break

        # 遍历当前节点的邻居，更新距离
        for neighbor, weight in graph.nodes[current_node]['neighbors'].items():
            new_distance = distances[current_node] + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node

    return distances, previous_nodes

# 生成最短路径（回溯前驱节点）
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    # 从终点回溯到起点
    while current_node is not None:
        path.insert(0, current_node)  # 插入到列表头部（保证路径顺序：起点→终点）
        current_node = previous_nodes[current_node]
    return path

# 导航函数（核心：验证路径是否"先到走廊"，不符合则重新计算）
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    # 1. 生成起点和终点节点ID
    start_node = f"{start_classroom}@{start_level}"
    end_node = f"{end_classroom}@{end_level}"

    # 2. 验证节点是否存在
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "❌ 无效的教室或楼层（节点不存在）"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 3. 计算最短路径
    distances, previous_nodes = dijkstra(graph, start_node)
    path = construct_path(previous_nodes, end_node)

    # 4. 验证路径是否"先到走廊"（核心校验）
    if len(path) >= 2:
        first_step_node = path[1]  # 起点后的第一个节点（必须是走廊）
        if graph.nodes[first_step_node]['type'] != 'corridor':
            # 若第一步不是走廊，尝试排除非走廊路径（重新计算时屏蔽非走廊的第一步）
            return force_corridor_first_path(graph, start_node, end_node)
    
    # 5. 返回有效路径
    if path:
        total_distance = distances[end_node]
        return path, f"✅ 路径规划成功！总距离：{total_distance:.2f} 单位"
    else:
        return None, "❌ 起点和终点间无有效路径"

# 强制"先到走廊"的路径计算（若初始路径不符合，屏蔽非走廊第一步后重新计算）
def force_corridor_first_path(graph, start_node, end_node):
    # 1. 收集起点（教室）的所有非走廊邻居（需要屏蔽）
    non_corridor_neighbors = [
        neighbor for neighbor in graph.nodes[start_node]['neighbors']
        if graph.nodes[neighbor]['type'] != 'corridor'
    ]

    # 2. 临时屏蔽非走廊邻居（复制原图，避免修改原图）
    temp_graph = Graph()
    # 复制所有节点
    for node_id, node_info in graph.nodes.items():
        temp_graph.add_node(
            node_id=node_id,
            node_type=node_info['type'],
            name=node_info['name'],
            level=node_info['level'],
            coordinates=node_info['coordinates']
        )
    # 复制边（跳过起点到非走廊邻居的边）
    for node1 in graph.nodes:
        for node2, weight in graph.nodes[node1]['neighbors'].items():
            # 屏蔽起点→非走廊邻居的边
            if node1 == start_node and node2 in non_corridor_neighbors:
                continue
            # 其他边正常添加
            temp_graph.add_edge(node1, node2, weight)

    # 3. 用临时图重新计算路径
    distances, previous_nodes = dijkstra(temp_graph, start_node)
    path = construct_path(previous_nodes, end_node)

    if path and len(path) >= 2 and temp_graph.nodes[path[1]]['type'] == 'corridor':
        total_distance = distances[end_node]
        return path, f"✅ 强制先到走廊！总距离：{total_distance:.2f} 单位"
    else:
        return None, "❌ 无法找到'先到走廊'的有效路径（可能走廊未连通）"

# 在3D图上绘制路径（优化：区分不同节点类型的显示样式）
def plot_path(ax, graph, path):
    # 提取路径的坐标（区分节点类型）
    x_coords = []
    y_coords = []
    z_coords = []
    node_types = []  # 记录每个节点的类型（用于后续标注）

    for node_id in path:
        coords = graph.nodes[node_id]['coordinates']
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        z_coords.append(coords[2])
        node_types.append(graph.nodes[node_id]['type'])

    # 1. 绘制路径主线（红色粗线，突出显示）
    ax.plot(
        x_coords, y_coords, z_coords,
        color='red', linewidth=3, linestyle='-', marker='o', markersize=6
    )

    # 2. 标记特殊节点（起点、终点、走廊、楼梯）
    for i, (x, y, z, node_type) in enumerate(zip(x_coords, y_coords, z_coords, node_types)):
        if i == 0:  # 起点（绿色星形）
            ax.scatter(x, y, z, color='green', s=300, marker='*', label='Start')
        elif i == len(path) - 1:  # 终点（紫色星形）
            ax.scatter(x, y, z, color='purple', s=300, marker='*', label='End')
        elif node_type == 'stair':  # 楼梯（红色三角形）
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stair' if i == 1 else "")
        elif node_type == 'corridor':  # 走廊（蓝色圆形）
            ax.scatter(x, y, z, color='blue', s=100, marker='o', label='Corridor' if i == 1 else "")

    ax.legend()

# 获取所有楼层和教室信息（适配Streamlit下拉框）
def get_classroom_info(school_data):
    levels = []
    classrooms_by_level = {}
    
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        levels.append(level_name)
        classrooms = [classroom['name'] for classroom in level['classrooms']]
        classrooms_by_level[level_name] = classrooms
        
    return levels, classrooms_by_level

# -------------------------- 3. Streamlit界面逻辑 --------------------------
def main():
    # 1. 页面标题和数据加载
    st.title("🏫 校园导航系统")
    st.subheader("3D地图与最短路径规划（强制先经过走廊）")

    # 加载JSON数据
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_navigation_graph(school_data)
        levels, classrooms_by_level = get_classroom_info(school_data)
        st.success("✅ 校园数据加载成功！")
    except FileNotFoundError:
        st.error("❌ 错误：未找到'school_data_detailed.json'文件，请检查文件路径。")
        return  # 数据加载失败，终止程序

    # 2. 布局：左右分栏（左侧选择器，右侧结果显示）
    col1, col2 = st.columns([1, 2])  # 左侧占1份，右侧占2份

    with col1:
        # 左侧：起点和终点选择（下拉框）
        st.markdown("### 📍 选择位置")
        
        # 起点选择（楼层→教室联动）
        st.markdown("#### 起点")
        start_level = st.selectbox("楼层", levels, key="start_level")
        start_classrooms = classrooms_by_level[start_level]
        start_classroom = st.selectbox("教室", start_classrooms, key="start_classroom")

        # 终点选择（楼层→教室联动）
        st.markdown("#### 终点")
        end_level = st.selectbox("楼层", levels, key="end_level")
        end_classrooms = classrooms_by_level[end_level]
        end_classroom = st.selectbox("教室", end_classrooms, key="end_classroom")

        # 导航按钮（点击触发路径计算）
        nav_button = st.button("🔍 查找最短路径", use_container_width=True)

    with col2:
        # 右侧：显示3D地图和导航结果
        st.markdown("### 🗺️ 3D校园地图")
        
        # 初始显示空的3D地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig  # 用session_state保存图，避免重复绘制
        
        # 点击导航按钮后，计算路径并更新地图
        if nav_button:
            # 调用导航函数
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            # 显示导航结果
            if path:
                st.success(message)
                # 显示路径详情
                st.markdown("#### 🛤️ 路径详情")
                for i, node in enumerate(path, 1):
                    if node.startswith("corridor_"):
                        st.write(f"{i}. 走廊")
                    else:
                        room, floor = node.split('@')
                        st.write(f"{i}. {room}（楼层：{floor}）")
                
                # 重新绘制带路径的3D图
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, nav_graph, path)
                st.session_state['fig'] = fig  # 更新保存的图
            else:
                st.error(message)
        
        # 显示3D图（Streamlit用st.pyplot()渲染matplotlib图）
        st.pyplot(st.session_state['fig'])

# -------------------------- 4. 运行主函数 --------------------------
if __name__ == "__main__":
    main()
