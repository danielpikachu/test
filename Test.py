import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. 基础配置：解决Streamlit matplotlib渲染问题 --------------------------
# Streamlit 不支持 matplotlib 交互后端，必须切换为非交互后端
plt.switch_backend('Agg')

# -------------------------- 2. 原有核心功能：数据读取、3D绘图、路径计算（无修改） --------------------------
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
        xs = [p[0] for p in all_corridor_points]  # 所有走廊点的X坐标
        ys = [p[1] for p in all_corridor_points]  # 所有走廊点的Y坐标
        min_x, max_x = min(xs), max(xs)  # 平面X轴范围
        min_y, max_y = min(ys), max(ys)  # 平面Y轴范围

        # 构造平面的4个顶点（闭合矩形，确保3D中显示为完整平面边框）
        plane_vertices = [
            [min_x, min_y, z],   # 左下角
            [max_x, min_y, z],   # 右下角
            [max_x, max_y, z],   # 右上角
            [min_x, max_y, z],   # 左上角
            [min_x, min_y, z]    # 回到起点，闭合图形
        ]
        # 提取顶点的X/Y/Z坐标，用于绘图
        x_plane = [p[0] for p in plane_vertices]
        y_plane = [p[1] for p in plane_vertices]
        z_plane = [p[2] for p in plane_vertices]

        # 绘制楼层平面边框（与楼层颜色一致，添加楼层标签）
        ax.plot(x_plane, y_plane, z_plane, color=color, linewidth=2, label=level['name'])

        # 绘制走廊
        for corridor in level['corridors']:
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            ax.plot(x, y, z_coords, color=color, linewidth=5)

        # 绘制楼梯
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stairs' if z == 0 else "")

        # 绘制教室（用立方体表示）
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']

            # 绘制教室标签
            ax.text(x, y, z, classroom['name'], color='black', fontweight='bold')

            # 绘制教室位置点
            ax.scatter(x, y, z, color=color, s=50)

            # 绘制教室边界（简化为矩形）
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color=color, linestyle='--')

    # 设置坐标轴
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Floor')
    ax.set_title('School 3D Map with Navigation')
    ax.legend()

    return fig, ax

# 自定义图数据结构
class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_id, node_type, name, level, coordinates):
        self.nodes[node_id] = {
            'type': node_type,
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

    # 添加所有位置节点
    for level in school_data['buildingA']['levels']:
        z = level['z']

        # 添加教室
        for classroom in level['classrooms']:
            node_id = f"{classroom['name']}@{level['name']}"
            graph.add_node(node_id,
                           'classroom',
                           classroom['name'],
                           level['name'],
                           classroom['coordinates'])

        # 添加楼梯
        for stair in level['stairs']:
            node_id = f"{stair['name']}@{level['name']}"
            graph.add_node(node_id,
                          'stair',
                           stair['name'],
                           level['name'],
                           stair['coordinates'])

    # 添加连接关系
    # 1. 同一楼层内的连接（基于走廊）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 获取该楼层所有节点
        level_nodes = list(graph.nodes.keys())
        level_nodes = [n for n in level_nodes if graph.nodes[n]['level'] == level_name]

        for i in range(len(level_nodes)):
            for j in range(i + 1, len(level_nodes)):
                coords1 = graph.nodes[level_nodes[i]]['coordinates']
                coords2 = graph.nodes[level_nodes[j]]['coordinates']
                distance = euclidean_distance(coords1, coords2)
                graph.add_edge(level_nodes[i], level_nodes[j], distance)

    # 2. 跨楼层连接（楼梯）
    for connection in school_data['buildingA']['connections']:
        from_stair, from_level = connection['from']
        to_stair, to_level = connection['to']

        from_node = f"{from_stair}@{from_level}"
        to_node = f"{to_stair}@{to_level}"

        if from_node in graph.nodes and to_node in graph.nodes:
            graph.add_edge(from_node, to_node, 1.0)

    return graph

# 自定义Dijkstra算法
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
    return path

# 导航函数
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    start_node = f"{start_classroom}@{start_level}"
    end_node = f"{end_classroom}@{end_level}"

    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "Invalid classroom or level"

    distances, previous_nodes = dijkstra(graph, start_node)
    path = construct_path(previous_nodes, end_node)

    if path:
        total_distance = distances[end_node]
        return path, f"Total distance: {total_distance:.2f} units"
    else:
        return None, "No path exists between these classrooms"

# 在3D图上绘制路径
def plot_path(ax, graph, path):
    x = []
    y = []
    z = []

    for node in path:
        coords = graph.nodes[node]['coordinates']
        x.append(coords[0])
        y.append(coords[1])
        z.append(coords[2])

    # 绘制路径
    ax.plot(x, y, z, color='red', linewidth=3, linestyle='-', marker='o')

    # 标记起点和终点
    ax.scatter(x[0], y[0], z[0], color='green', s=300, marker='*', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='purple', s=300, marker='*', label='End')

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

# -------------------------- 3. Streamlit界面逻辑（替换原Tkinter界面） --------------------------
def main():
    # 1. 页面标题和数据加载
    st.title("🏫 School Campus Navigation System")
    st.subheader("3D Map & Shortest Path Finder")

    # 加载JSON数据（注意：确保 school_data_detailed.json 和代码在同一目录）
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_navigation_graph(school_data)
        levels, classrooms_by_level = get_classroom_info(school_data)
        st.success("✅ School data loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Error: 'school_data_detailed.json' not found. Please check the file path.")
        return  # 数据加载失败，终止程序

    # 2. 布局：左右分栏（左侧选择器，右侧结果显示）
    col1, col2 = st.columns([1, 2])  # 左侧占1份，右侧占2份

    with col1:
        # 左侧：起点和终点选择（下拉框）
        st.markdown("### 📍 Select Locations")
        
        # 起点选择（楼层→教室联动）
        st.markdown("#### Start Point")
        start_level = st.selectbox("Floor", levels, key="start_level")
        start_classrooms = classrooms_by_level[start_level]
        start_classroom = st.selectbox("Classroom", start_classrooms, key="start_classroom")

        # 终点选择（楼层→教室联动）
        st.markdown("#### End Point")
        end_level = st.selectbox("Floor", levels, key="end_level")
        end_classrooms = classrooms_by_level[end_level]
        end_classroom = st.selectbox("Classroom", end_classrooms, key="end_classroom")

        # 导航按钮（点击触发路径计算）
        nav_button = st.button("🔍 Find Shortest Path", use_container_width=True)

    with col2:
        # 右侧：显示3D地图和导航结果
        st.markdown("### 🗺️ 3D Campus Map")
        
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
                st.success(f"📊 Navigation Result: {message}")
                # 显示路径详情
                st.markdown("#### 🛤️ Path Details")
                for i, node in enumerate(path, 1):
                    room, floor = node.split('@')
                    st.write(f"{i}. {room} (Floor: {floor})")
                
                # 重新绘制带路径的3D图
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, nav_graph, path)
                st.session_state['fig'] = fig  # 更新保存的图
            else:
                st.error(f"❌ {message}")
        
        # 显示3D图（Streamlit用st.pyplot()渲染matplotlib图）
        st.pyplot(st.session_state['fig'])

# -------------------------- 4. 运行主函数 --------------------------
if __name__ == "__main__":
    main()
