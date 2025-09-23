import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. 基础配置 --------------------------
plt.switch_backend('Agg')  # 解决Streamlit matplotlib渲染问题

# 定义安全的颜色常量（使用Matplotlib确认支持的颜色）
COLORS = {
    'floor': {-2: 'blue', 2: 'green', 5: 'orange', 10: 'red'},
    'corridor_node': 'cyan',
    'corridor_label': 'navy',
    'stair': 'red',
    'stair_label': 'darkred',
    'classroom_label': 'black',
    'path': 'red',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple'
}

# -------------------------- 2. 核心功能实现 --------------------------
# 读取JSON数据
def load_school_data_detailed(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading data file: {str(e)}")
        return None

# 绘制3D地图
def plot_3d_map(school_data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 处理每个楼层
    for level in school_data['buildingA']['levels']:
        z = level['z']
        color = COLORS['floor'].get(z, 'gray')
        level_name = level['name']

        # 绘制楼层平面边框
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
        ax.plot(x_plane, y_plane, z_plane, color=color, linewidth=2, label=level_name)

        # 绘制走廊及走廊节点
        for corr_idx, corridor in enumerate(level['corridors']):
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            
            # 绘制走廊线条
            ax.plot(x, y, z_coords, color=color, linewidth=5, alpha=0.7)
            
            # 标记走廊节点
            for p_idx, (px, py, pz) in enumerate(points):
                ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=100, marker='s', alpha=0.8)
                ax.text(px, py, pz, f'C{corr_idx}-{p_idx}', color=COLORS['corridor_label'], fontsize=8)

        # 绘制楼梯
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color=COLORS['stair'], s=200, marker='^', label='Stairs' if z == -2 else "")
            ax.text(x, y, z, stair['name'], color=COLORS['stair_label'], fontweight='bold')

        # 绘制教室
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']

            # 教室标签
            ax.text(x, y, z, classroom['name'], color=COLORS['classroom_label'], fontweight='bold')

            # 教室位置点
            ax.scatter(x, y, z, color=color, s=50)

            # 教室边界
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color=color, linestyle='--', alpha=0.5)

    # 设置坐标轴
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Floor (Z)')
    ax.set_title('School 3D Map (Fully Connected Paths)')
    ax.legend()

    return fig, ax

# 自定义图数据结构
class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_id, node_type, name, level, coordinates):
        self.nodes[node_id] = {
            'type': node_type,  # 类型：classroom/stair/corridor
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }

    def add_edge(self, node1, node2, weight):
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1]['neighbors'][node2] = weight
            self.nodes[node2]['neighbors'][node1] = weight

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图
def build_navigation_graph(school_data):
    graph = Graph()

    # 第一步：添加所有节点（教室+楼梯+走廊）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 添加教室节点
        for classroom in level['classrooms']:
            node_id = f"Class_{classroom['name']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='classroom',
                name=classroom['name'],
                level=level_name,
                coordinates=classroom['coordinates']
            )

        # 添加楼梯节点
        for stair in level['stairs']:
            node_id = f"Stair_{stair['name']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='stair',
                name=stair['name'],
                level=level_name,
                coordinates=stair['coordinates']
            )

        # 添加走廊节点
        for corr_idx, corridor in enumerate(level['corridors']):
            for p_idx, point in enumerate(corridor['points']):
                node_id = f"Corr_{level_name}_C{corr_idx}_P{p_idx}"
                graph.add_node(
                    node_id=node_id,
                    node_type='corridor',
                    name=f"Corridor_{corr_idx}_Point_{p_idx}",
                    level=level_name,
                    coordinates=point
                )

    # 第二步：添加连接关系
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        
        # 获取当前楼层所有走廊节点
        corr_nodes = [n for n in graph.nodes if 
                      graph.nodes[n]['type'] == 'corridor' and 
                      graph.nodes[n]['level'] == level_name]

        # 1. 同一走廊内的节点连接（沿走廊路径）
        for corr_idx, corridor in enumerate(level['corridors']):
            corr_points = corridor['points']
            for p_idx in range(len(corr_points) - 1):
                node1_id = f"Corr_{level_name}_C{corr_idx}_P{p_idx}"
                node2_id = f"Corr_{level_name}_C{corr_idx}_P{p_idx + 1}"
                coords1 = graph.nodes[node1_id]['coordinates']
                coords2 = graph.nodes[node2_id]['coordinates']
                distance = euclidean_distance(coords1, coords2)
                graph.add_edge(node1_id, node2_id, distance)

        # 2. 不同走廊间的节点连接
        for i in range(len(corr_nodes)):
            node1 = corr_nodes[i]
            coords1 = graph.nodes[node1]['coordinates']
            for j in range(i + 1, len(corr_nodes)):
                node2 = corr_nodes[j]
                coords2 = graph.nodes[node2]['coordinates']
                distance = euclidean_distance(coords1, coords2)
                
                # 距离小于3.0的走廊节点视为交叉点，建立连接
                if distance < 3.0:
                    graph.add_edge(node1, node2, distance)

        # 3. 教室 → 最近的走廊节点连接
        class_nodes = [n for n in graph.nodes if 
                       graph.nodes[n]['type'] == 'classroom' and 
                       graph.nodes[n]['level'] == level_name]
        for class_node_id in class_nodes:
            class_coords = graph.nodes[class_node_id]['coordinates']
            min_dist = float('inf')
            nearest_corr_node = None
            for corr_node_id in corr_nodes:
                corr_coords = graph.nodes[corr_node_id]['coordinates']
                dist = euclidean_distance(class_coords, corr_coords)
                if dist < min_dist:
                    min_dist = dist
                    nearest_corr_node = corr_node_id
            if nearest_corr_node:
                graph.add_edge(class_node_id, nearest_corr_node, min_dist)

        # 4. 楼梯 → 最近的走廊节点连接
        stair_nodes = [n for n in graph.nodes if 
                       graph.nodes[n]['type'] == 'stair' and 
                       graph.nodes[n]['level'] == level_name]
        for stair_node_id in stair_nodes:
            stair_coords = graph.nodes[stair_node_id]['coordinates']
            min_dist = float('inf')
            nearest_corr_node = None
            for corr_node_id in corr_nodes:
                corr_coords = graph.nodes[corr_node_id]['coordinates']
                dist = euclidean_distance(stair_coords, corr_coords)
                if dist < min_dist:
                    min_dist = dist
                    nearest_corr_node = corr_node_id
            if nearest_corr_node:
                graph.add_edge(stair_node_id, nearest_corr_node, min_dist)

    # 5. 跨楼层连接（仅楼梯节点之间）
    for connection in school_data['buildingA']['connections']:
        from_stair_name, from_level = connection['from']
        to_stair_name, to_level = connection['to']

        from_stair_node = f"Stair_{from_stair_name}@{from_level}"
        to_stair_node = f"Stair_{to_stair_name}@{to_level}"

        if from_stair_node in graph.nodes and to_stair_node in graph.nodes:
            graph.add_edge(from_stair_node, to_stair_node, 5.0)

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

# 导航函数
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    try:
        start_node = f"Class_{start_classroom}@{start_level}"
        end_node = f"Class_{end_classroom}@{end_level}"

        if start_node not in graph.nodes or end_node not in graph.nodes:
            return None, "Invalid classroom or level (node not found)", None

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
                if node_type in ['classroom', 'stair']:
                    simplified_path.append(f"{node_name} (Floor: {node_level})")
            return path, f"Total distance: {total_distance:.2f} units", simplified_path
        else:
            return None, "No path exists between these classrooms", None
    except Exception as e:
        return None, f"Navigation error: {str(e)}", None

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

        # 标记起点和终点（使用Matplotlib确认支持的颜色）
        ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=500, marker='*', label='Start', edgecolors='black')
        ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=500, marker='*', label='End', edgecolors='black')
        
        # 使用安全的颜色名称
        ax.text(x[0], y[0], z[0], f"Start\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=10)
        ax.text(x[-1], y[-1], z[-1], f"End\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=10)

        ax.legend()
    except Exception as e:
        st.error(f"Error plotting path: {str(e)}")

# 获取所有楼层和教室信息
def get_classroom_info(school_data):
    try:
        levels = []
        classrooms_by_level = {}
        
        for level in school_data['buildingA']['levels']:
            level_name = level['name']
            levels.append(level_name)
            classrooms = [classroom['name'] for classroom in level['classrooms']]
            classrooms_by_level[level_name] = classrooms
            
        return levels, classrooms_by_level
    except Exception as e:
        st.error(f"Error getting classroom info: {str(e)}")
        return [], {}

# -------------------------- 3. Streamlit界面逻辑 --------------------------
def main():
    # 页面标题和数据加载
    st.title("🏫 School Campus Navigation System")
    st.subheader("3D Map & Fully Connected Path Finder")

    # 加载JSON数据
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        nav_graph = build_navigation_graph(school_data)
        levels, classrooms_by_level = get_classroom_info(school_data)
        st.success("✅ School data loaded successfully!")
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return

    # 布局：左右分栏
    col1, col2 = st.columns([1, 2])

    with col1:
        # 左侧：起点和终点选择
        st.markdown("### 📍 Select Locations")
        
        # 起点选择
        st.markdown("#### Start Point")
        start_level = st.selectbox("Floor", levels, key="start_level")
        start_classrooms = classrooms_by_level.get(start_level, [])
        start_classroom = st.selectbox("Classroom", start_classrooms, key="start_classroom")

        # 终点选择
        st.markdown("#### End Point")
        end_level = st.selectbox("Floor", levels, key="end_level")
        end_classrooms = classrooms_by_level.get(end_level, [])
        end_classroom = st.selectbox("Classroom", end_classrooms, key="end_classroom")

        # 导航按钮
        nav_button = st.button("🔍 Find Shortest Path", use_container_width=True)

    with col2:
        # 右侧：显示3D地图和导航结果
        st.markdown("### 🗺️ 3D Campus Map")
        
        # 初始显示3D地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        # 点击导航按钮后，计算路径并更新地图
        if nav_button:
            try:
                path, message, simplified_path = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
                
                # 显示导航结果
                if path:
                    st.success(f"📊 Navigation Result: {message}")
                    # 显示简化路径详情
                    st.markdown("#### 🛤️ Path Details (Classrooms & Stairs)")
                    for i, step in enumerate(simplified_path, 1):
                        st.write(f"{i}. {step}")
                    
                    # 重新绘制带路径的3D图
                    fig, ax = plot_3d_map(school_data)
                    plot_path(ax, nav_graph, path)
                    st.session_state['fig'] = fig
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"Error during navigation: {str(e)}")
        
        # 显示3D图
        try:
            st.pyplot(st.session_state['fig'])
        except Exception as e:
            st.error(f"Error displaying map: {str(e)}")

# -------------------------- 4. 运行主函数 --------------------------
if __name__ == "__main__":
    main()
