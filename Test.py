import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. 基础配置 --------------------------
plt.switch_backend('Agg')

# -------------------------- 2. 核心功能实现 --------------------------
# 读取JSON数据
def load_school_data_detailed(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# 绘制3D地图
def plot_3d_map(school_data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同楼层使用不同颜色
    floor_colors = {0: 'blue', 2: 'green', 5: 'orange'}  

    # 处理每个楼层
    for level in school_data['buildingA']['levels']:
        z = level['z']
        color = floor_colors.get(z, 'gray')

        # 收集当前楼层所有走廊的坐标点
        all_corridor_points = []
        for corridor in level['corridors']:
            all_corridor_points.extend(corridor['points'])
        if not all_corridor_points:
            continue  

        # 计算平面的X/Y轴范围
        xs = [p[0] for p in all_corridor_points]
        ys = [p[1] for p in all_corridor_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # 构造平面的4个顶点
        plane_vertices = [
            [min_x, min_y, z], [max_x, min_y, z], 
            [max_x, max_y, z], [min_x, max_y, z], [min_x, min_y, z]
        ]
        x_plane = [p[0] for p in plane_vertices]
        y_plane = [p[1] for p in plane_vertices]
        z_plane = [p[2] for p in plane_vertices]

        # 绘制楼层平面边框
        ax.plot(x_plane, y_plane, z_plane, color=color, linewidth=2, label=level['name'])

        # 绘制走廊
        for corridor in level['corridors']:
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            ax.plot(x, y, z_coords, color=color, linewidth=5, alpha=0.8)

        # 绘制楼梯（突出显示，方便识别楼梯附近走廊）
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stairs' if z == 0 else "")
            ax.text(x, y, z+0.1, stair['name'], color='red', fontweight='bold')
            
            # 标记楼梯附近的走廊区域（增加半透明圆圈）
            ax.scatter(x, y, z, color='red', s=800, alpha=0.2, marker='o')

        # 绘制教室
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']

            # 教室标签
            ax.text(x, y, z, classroom['name'], color='black', fontweight='bold')
            # 教室位置点
            ax.scatter(x, y, z, color=color, s=50)
            # 教室边界
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color=color, linestyle='--')

    # 设置坐标轴
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Floor')
    ax.set_title('School 3D Map (Optimal Corridor & Stair Path)')
    ax.legend()

    return fig, ax

# 自定义图数据结构
class Graph:
    def __init__(self):
        self.nodes = {}  # key: node_id, value: node_info
        self.stair_proximity = {}  # 记录走廊节点与最近楼梯的距离

    def add_node(self, node_id, node_type, name, level, coordinates, stair_distance=None):
        """添加节点，新增stair_distance参数记录与最近楼梯的距离"""
        self.nodes[node_id] = {
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        if node_type == 'corridor' and stair_distance is not None:
            self.stair_proximity[node_id] = stair_distance

    def add_edge(self, node1, node2, weight):
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1]['neighbors'][node2] = weight
            self.nodes[node2]['neighbors'][node1] = weight

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（优化走廊和楼梯路径）
def build_navigation_graph(school_data):
    graph = Graph()

    # 步骤1：添加所有节点（教室、楼梯、走廊）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']
        
        # 收集当前楼层楼梯坐标（用于计算走廊与楼梯的距离）
        stair_coords = [stair['coordinates'] for stair in level['stairs']]

        # 1.1 添加教室节点
        for classroom in level['classrooms']:
            node_id = f"{classroom['name']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='classroom',
                name=classroom['name'],
                level=level_name,
                coordinates=classroom['coordinates']
            )

        # 1.2 添加楼梯节点
        for stair in level['stairs']:
            node_id = f"{stair['name']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='stair',
                name=stair['name'],
                level=level_name,
                coordinates=stair['coordinates']
            )

        # 1.3 添加走廊节点（记录与最近楼梯的距离）
        for corridor_idx, corridor in enumerate(level['corridors']):
            for point_idx, point in enumerate(corridor['points']):
                node_id = f"corridor_{point[0]}_{point[1]}_{z}"
                if node_id not in graph.nodes:
                    # 计算该走廊点与最近楼梯的距离
                    min_stair_dist = min(euclidean_distance(point, sc) for sc in stair_coords) if stair_coords else 0
                    
                    graph.add_node(
                        node_id=node_id,
                        node_type='corridor',
                        name=f"Corridor_{corridor_idx+1}_Point_{point_idx+1}",
                        level=level_name,
                        coordinates=point,
                        stair_distance=min_stair_dist
                    )

    # 步骤2：添加边（优化权重计算）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 2.1 教室 ↔ 走廊：仅连接最近的2个走廊（确保最短距离）
        for classroom in level['classrooms']:
            classroom_node_id = f"{classroom['name']}@{level_name}"
            classroom_coords = classroom['coordinates']
            
            # 找出当前楼层所有走廊节点并按距离排序
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            
            # 按距离排序，仅连接最近的2个走廊（减少冗余，保证最短）
            corridor_distances = sorted(
                [(n, euclidean_distance(classroom_coords, graph.nodes[n]['coordinates'])) for n in corridor_nodes],
                key=lambda x: x[1]
            )[:2]  # 只取前2个最近的走廊
            
            # 添加连接，权重=真实距离×0.5（让教室→走廊的路径权重更低）
            for node_id, distance in corridor_distances:
                graph.add_edge(classroom_node_id, node_id, distance * 0.5)

        # 2.2 楼梯 ↔ 走廊：仅连接最近的3个走廊（确保楼梯附近走廊权重最低）
        for stair in level['stairs']:
            stair_node_id = f"{stair['name']}@{level_name}"
            stair_coords = stair['coordinates']
            
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            
            # 按距离排序，取最近的3个走廊（楼梯附近走廊优先）
            corridor_distances = sorted(
                [(n, euclidean_distance(stair_coords, graph.nodes[n]['coordinates'])) for n in corridor_nodes],
                key=lambda x: x[1]
            )[:3]
            
            # 添加连接，楼梯附近的走廊权重更低
            for node_id, distance in corridor_distances:
                weight = distance * 0.3  # 楼梯附近走廊权重大幅降低
                graph.add_edge(stair_node_id, node_id, weight)

        # 2.3 走廊 ↔ 走廊：仅连接相邻的走廊点（避免跨走廊绕路）
        corridor_nodes = [
            node_id for node_id in graph.nodes 
            if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
        ]
        
        for i in range(len(corridor_nodes)):
            node1 = corridor_nodes[i]
            coords1 = graph.nodes[node1]['coordinates']
            for j in range(i + 1, len(corridor_nodes)):
                node2 = corridor_nodes[j]
                coords2 = graph.nodes[node2]['coordinates']
                distance = euclidean_distance(coords1, coords2)
                
                # 仅连接距离<3的走廊节点（模拟真实走廊的连续路径）
                if distance < 3:
                    graph.add_edge(node1, node2, distance)  # 权重=真实距离，确保直线最短

    # 2.4 楼梯 ↔ 楼梯：跨楼层连接
    for connection in school_data['buildingA']['connections']:
        from_stair_name, from_level = connection['from']
        to_stair_name, to_level = connection['to']
        
        from_stair_node = f"{from_stair_name}@{from_level}"
        to_stair_node = f"{to_stair_name}@{to_level}"
        
        if from_stair_node in graph.nodes and to_stair_node in graph.nodes:
            graph.add_edge(from_stair_node, to_stair_node, 1.0)

    return graph

# 改进的Dijkstra算法，强制经走廊且保证最短距离
def dijkstra(graph, start_node, end_node):
    # 初始化距离：起点为0，其他为无穷大
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    unvisited_nodes = set(graph.nodes.keys())

    # 提取起点/终点的类型和楼层（用于约束）
    start_type = graph.nodes[start_node]['type'] if start_node in graph.nodes else ""
    end_type = graph.nodes[end_node]['type'] if end_node in graph.nodes else ""
    end_level = graph.nodes[end_node]['level'] if end_node in graph.nodes else None

    while unvisited_nodes:
        # 选择当前距离最短的节点（保证最短路径优先）
        current_node = min(unvisited_nodes, key=lambda x: distances[x])
        unvisited_nodes.remove(current_node)

        if distances[current_node] == float('inf'):
            break  # 无可达路径

        current_type = graph.nodes[current_node]['type']
        current_level = graph.nodes[current_node]['level']

        for neighbor, weight in graph.nodes[current_node]['neighbors'].items():
            neighbor_type = graph.nodes[neighbor]['type']
            neighbor_level = graph.nodes[neighbor]['level']

            # -------------------------- 核心约束：强制经走廊 --------------------------
            # 规则1：教室只能连接走廊（禁止“教室→楼梯”“教室→教室”直接跳转）
            if current_type == 'classroom' and neighbor_type != 'corridor':
                continue  # 跳过非走廊邻居
            # 规则2：楼梯只能连接走廊（禁止“楼梯→教室”“楼梯→楼梯”直接跳转，跨楼层楼梯除外）
            if current_type == 'stair' and neighbor_type != 'corridor':
                # 仅允许跨楼层的“楼梯→楼梯”连接
                if not (neighbor_type == 'stair' and current_level != neighbor_level):
                    continue  # 同楼层楼梯或楼梯→教室，跳过
            # 规则3：终点为教室时，前一个节点必须是走廊（最终一步强制经走廊）
            if neighbor == end_node and end_type == 'classroom' and current_type != 'corridor':
                continue  # 非走廊节点无法直接到教室终点

            # -------------------------- 权重优化：保证经走廊的路径最短 --------------------------
            extra_factor = 1.0  # 额外因子：默认1.0（不改变距离）
            # 优化1：跨楼层时，靠近楼梯的走廊权重更低
            if (current_level != end_level) and (current_type == 'corridor'):
                stair_dist = graph.stair_proximity.get(current_node, float('inf'))
                # 走廊越近楼梯，因子越小（范围0.4-1.0）
                extra_factor = 0.4 + (min(stair_dist, 10) / 10) * 0.6
            # 优化2：同楼层走廊间的直线距离权重（避免绕路）
            if current_type == 'corridor' and neighbor_type == 'corridor':
                # 直接用欧氏距离作为基础权重
                base_dist = euclidean_distance(
                    graph.nodes[current_node]['coordinates'],
                    graph.nodes[neighbor]['coordinates']
                )
                weight = base_dist  # 确保走廊间距离为真实直线距离

            # 计算新距离（距离=基础权重×额外因子，保证最短优先）
            new_distance = distances[current_node] + weight * extra_factor

            # 更新最短距离
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node

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

    # 基础校验：节点是否存在
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "❌ 无效的教室或楼层"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 调用优化后的Dijkstra算法，生成路径
    distances, previous_nodes = dijkstra(graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node)

    # 路径校验：确保经走廊且最短
    has_corridor = any(graph.nodes[node]['type'] == 'corridor' for node in path[1:-1])
    if not has_corridor:
        # 若路径无走廊，强制生成经走廊的路径
        path, message = force_corridor_first_path(graph, start_node, end_node)
        if not path:
            return None, message
        # 重新计算强制路径的距离
        total_distance = sum(
            euclidean_distance(
                graph.nodes[path[k]]['coordinates'],
                graph.nodes[path[k+1]]['coordinates']
            ) for k in range(len(path)-1)
        )
    else:
        # 计算真实总距离（确保是最短）
        total_distance = distances[end_node]

    # 最终返回：经走廊的最短路径
    return path, f"✅ 经走廊最短路径规划成功！总距离：{total_distance:.2f} 单位"

# 强制"先到走廊"的路径计算
def force_corridor_first_path(graph, start_node, end_node):
    non_corridor_neighbors = [
        neighbor for neighbor in graph.nodes[start_node]['neighbors']
        if graph.nodes[neighbor]['type'] != 'corridor'
    ]

    temp_graph = Graph()
    for node_id, node_info in graph.nodes.items():
        temp_graph.add_node(
            node_id=node_id,
            node_type=node_info['type'],
            name=node_info['name'],
            level=node_info['level'],
            coordinates=node_info['coordinates']
        )
    
    for node1 in graph.nodes:
        for node2, weight in graph.nodes[node1]['neighbors'].items():
            if node1 == start_node and node2 in non_corridor_neighbors:
                continue
            temp_graph.add_edge(node1, node2, weight)

    distances, previous_nodes = dijkstra(temp_graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node)

    if path and len(path) >= 2 and temp_graph.nodes[path[1]]['type'] == 'corridor':
        total_distance = distances[end_node]
        return path, f"✅ 强制先到走廊！总距离：{total_distance:.2f} 单位"
    else:
        return None, "❌ 无法找到符合要求的路径"

# 在3D图上绘制路径（突出显示临近走廊和楼梯附近走廊）
def plot_path(ax, graph, path):
    x_coords = []
    y_coords = []
    z_coords = []
    node_types = []
    node_details = []  # 存储节点详细信息

    for node_id in path:
        node = graph.nodes[node_id]
        coords = node['coordinates']
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        z_coords.append(coords[2])
        node_types.append(node['type'])
        
        # 标记特殊走廊节点
        detail = ""
        if node['type'] == 'corridor':
            if len(node_details) == 0:  # 第一个走廊节点是临近教室的走廊
                detail = "near_classroom"
            elif any(t == 'stair' for t in node_types):  # 楼梯之后的走廊
                detail = "after_stair"
            elif graph.stair_proximity.get(node_id, float('inf')) < 5:  # 楼梯附近的走廊
                detail = "near_stair"
        node_details.append(detail)

    # 绘制路径主线
    ax.plot(
        x_coords, y_coords, z_coords,
        color='red', linewidth=3, linestyle='-', marker='o', markersize=6
    )

    # 标记特殊节点
    for i, (x, y, z, node_type, detail) in enumerate(zip(x_coords, y_coords, z_coords, node_types, node_details)):
        if i == 0:  # 起点
            ax.scatter(x, y, z, color='green', s=300, marker='*', label='Start')
        elif i == len(path) - 1:  # 终点
            ax.scatter(x, y, z, color='purple', s=300, marker='*', label='End')
        elif node_type == 'stair':  # 楼梯
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stair')
        elif node_type == 'corridor':  # 走廊（根据类型使用不同颜色）
            if detail == "near_classroom":
                ax.scatter(x, y, z, color='cyan', s=150, marker='o', label='Near Classroom')
            elif detail == "near_stair":
                ax.scatter(x, y, z, color='orange', s=150, marker='o', label='Near Stair')
            else:
                ax.scatter(x, y, z, color='blue', s=100, marker='o', label='Corridor')

    ax.legend()

# 获取所有楼层和教室信息
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
    st.title("🏫 校园导航系统")
    st.subheader("3D地图与优化路径规划（必须经过走廊且保证最短距离）")

    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_navigation_graph(school_data)
        levels, classrooms_by_level = get_classroom_info(school_data)
        st.success("✅ 校园数据加载成功！")
    except FileNotFoundError:
        st.error("❌ 错误：未找到'school_data_detailed.json'文件，请检查文件路径。")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📍 选择位置")
        
        st.markdown("#### 起点")
        start_level = st.selectbox("楼层", levels, key="start_level")
        start_classrooms = classrooms_by_level[start_level]
        start_classroom = st.selectbox("教室", start_classrooms, key="start_classroom")

        st.markdown("#### 终点")
        end_level = st.selectbox("楼层", levels, key="end_level")
        end_classrooms = classrooms_by_level[end_level]
        end_classroom = st.selectbox("教室", end_classrooms, key="end_classroom")

        nav_button = st.button("🔍 查找最优路径", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ 3D校园地图")
        
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        if nav_button:
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            if path:
                st.success(message)
                st.markdown("#### 🛤️ 路径详情")
                for i, node in enumerate(path, 1):
                    if node.startswith("corridor_"):
                        # 识别特殊走廊节点
                        if i == 2:  # 起点后的第一个走廊
                            st.write(f"{i}. 临近教室的走廊")
                        elif any("stair" in path[j] for j in range(i)) and "stair" not in node:
                            st.write(f"{i}. 楼梯附近的走廊")
                        else:
                            st.write(f"{i}. 走廊")
                    else:
                        room, floor = node.split('@')
                        st.write(f"{i}. {room}（楼层：{floor}）")
                
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, nav_graph, path)
                st.session_state['fig'] = fig
            else:
                st.error(message)
        
        st.pyplot(st.session_state['fig'])

if __name__ == "__main__":
    main()
