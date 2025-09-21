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

        # 绘制走廊（加粗显示，突出走廊网络）
        for corridor in level['corridors']:
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            ax.plot(x, y, z_coords, color=color, linewidth=8, alpha=0.8)
            
            # 标记走廊节点
            ax.scatter(x, y, z_coords, color='white', s=50, alpha=0.7)

        # 绘制楼梯
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=300, marker='^', label='Stairs' if z == 0 else "")
            ax.text(x, y, z+0.1, stair['name'], color='red', fontweight='bold')

        # 绘制教室
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']

            # 教室标签
            ax.text(x, y, z, classroom['name'], color='black', fontweight='bold')
            # 教室位置点
            ax.scatter(x, y, z, color=color, s=100)
            # 教室边界
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color=color, linestyle='--', linewidth=2)

    # 设置坐标轴
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Floor')
    ax.set_title('School 3D Map (Strictly Along Corridors)')
    ax.legend()

    return fig, ax

# 自定义图数据结构，强化走廊网络概念
class Graph:
    def __init__(self):
        self.nodes = {}  # key: node_id, value: node_info
        self.corridor_network = set()  # 仅包含走廊节点ID，强化走廊网络概念
        self.stair_connections = {}  # 楼梯与走廊的连接关系

    def add_node(self, node_id, node_type, name, level, coordinates):
        self.nodes[node_id] = {
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        if node_type == 'corridor':
            self.corridor_network.add(node_id)

    def add_edge(self, node1, node2, weight, is_corridor_edge=False):
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1]['neighbors'][node2] = {
                'weight': weight,
                'is_corridor_edge': is_corridor_edge
            }
            self.nodes[node2]['neighbors'][node1] = {
                'weight': weight,
                'is_corridor_edge': is_corridor_edge
            }
            
            # 记录楼梯与走廊的连接
            if (self.nodes[node1]['type'] == 'stair' and 
                self.nodes[node2]['type'] == 'corridor'):
                self.stair_connections.setdefault(node1, []).append(node2)
            if (self.nodes[node2]['type'] == 'stair' and 
                self.nodes[node1]['type'] == 'corridor'):
                self.stair_connections.setdefault(node2, []).append(node1)

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（强化走廊网络，确保路径必须沿走廊）
def build_navigation_graph(school_data):
    graph = Graph()

    # 步骤1：添加所有节点（教室、楼梯、走廊）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

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

        # 1.3 添加走廊节点（走廊网络的核心节点）
        for corridor_idx, corridor in enumerate(level['corridors']):
            for point_idx, point in enumerate(corridor['points']):
                node_id = f"corridor_{corridor_idx+1}_point_{point_idx+1}@{level_name}"
                graph.add_node(
                    node_id=node_id,
                    node_type='corridor',
                    name=f"Corridor {corridor_idx+1} Point {point_idx+1}",
                    level=level_name,
                    coordinates=point
                )

    # 步骤2：添加边（严格遵循走廊网络逻辑）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 2.1 教室 ↔ 走廊：仅允许教室连接到最近的1个走廊节点（确保从教室直接进入走廊）
        for classroom in level['classrooms']:
            classroom_node_id = f"{classroom['name']}@{level_name}"
            classroom_coords = classroom['coordinates']
            
            # 找出当前楼层所有走廊节点
            corridor_nodes = [
                node_id for node_id in graph.corridor_network 
                if graph.nodes[node_id]['level'] == level_name
            ]
            
            # 按距离排序，只连接最近的1个走廊（确保唯一入口）
            if corridor_nodes:
                corridor_distances = sorted(
                    [(n, euclidean_distance(classroom_coords, graph.nodes[n]['coordinates'])) 
                     for n in corridor_nodes],
                    key=lambda x: x[1]
                )
                closest_corridor, distance = corridor_distances[0]
                # 权重设置为较小值，确保优先选择
                graph.add_edge(classroom_node_id, closest_corridor, distance * 0.3)

        # 2.2 走廊 ↔ 走廊：构建连续的走廊网络（核心改进）
        # 同一走廊内的点依次连接（确保走廊是连续路径）
        for corridor_idx, corridor in enumerate(level['corridors']):
            points = corridor['points']
            for i in range(len(points) - 1):
                node1_id = f"corridor_{corridor_idx+1}_point_{i+1}@{level_name}"
                node2_id = f"corridor_{corridor_idx+1}_point_{i+2}@{level_name}"
                
                if node1_id in graph.nodes and node2_id in graph.nodes:
                    distance = euclidean_distance(points[i], points[i+1])
                    # 标记为走廊边缘，权重为实际距离（确保沿走廊走是最短路径）
                    graph.add_edge(node1_id, node2_id, distance, is_corridor_edge=True)
        
        # 2.3 不同走廊之间的连接：仅在走廊交汇处连接
        all_corridors = level['corridors']
        for i in range(len(all_corridors)):
            for j in range(i + 1, len(all_corridors)):
                # 检查两个走廊是否有交汇点（距离小于阈值）
                for p1 in all_corridors[i]['points']:
                    for p2 in all_corridors[j]['points']:
                        if euclidean_distance(p1, p2) < 1.5:  # 阈值表示走廊交汇处
                            node1_id = f"corridor_{i+1}_point_{all_corridors[i]['points'].index(p1)+1}@{level_name}"
                            node2_id = f"corridor_{j+1}_point_{all_corridors[j]['points'].index(p2)+1}@{level_name}"
                            
                            if node1_id in graph.nodes and node2_id in graph.nodes:
                                distance = euclidean_distance(p1, p2)
                                graph.add_edge(node1_id, node2_id, distance, is_corridor_edge=True)

        # 2.4 楼梯 ↔ 走廊：楼梯只能连接到直接相邻的走廊节点
        for stair in level['stairs']:
            stair_node_id = f"{stair['name']}@{level_name}"
            stair_coords = stair['coordinates']
            
            corridor_nodes = [
                node_id for node_id in graph.corridor_network 
                if graph.nodes[node_id]['level'] == level_name
            ]
            
            # 只连接距离楼梯最近的1-2个走廊节点（模拟楼梯与走廊的实际连接）
            if corridor_nodes:
                corridor_distances = sorted(
                    [(n, euclidean_distance(stair_coords, graph.nodes[n]['coordinates'])) 
                     for n in corridor_nodes],
                    key=lambda x: x[1]
                )[:2]  # 最多连接2个最近的走廊节点
                
                for corridor_node, distance in corridor_distances:
                    if distance < 4:  # 确保楼梯确实与走廊相邻
                        graph.add_edge(stair_node_id, corridor_node, distance * 0.2)

    # 2.5 楼梯 ↔ 楼梯：跨楼层连接（保持原逻辑）
    for connection in school_data['buildingA']['connections']:
        from_stair_name, from_level = connection['from']
        to_stair_name, to_level = connection['to']
        
        from_stair_node = f"{from_stair_name}@{from_level}"
        to_stair_node = f"{to_stair_name}@{to_level}"
        
        if from_stair_node in graph.nodes and to_stair_node in graph.nodes:
            graph.add_edge(from_stair_node, to_stair_node, 1.0)

    return graph

# 改进的Dijkstra算法，严格限制只能沿走廊行走
def dijkstra(graph, start_node, end_node):
    # 初始化距离：起点为0，其他为无穷大
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    unvisited_nodes = set(graph.nodes.keys())

    # 提取起点/终点信息
    start_type = graph.nodes[start_node]['type'] if start_node in graph.nodes else ""
    end_type = graph.nodes[end_node]['type'] if end_node in graph.nodes else ""
    end_level = graph.nodes[end_node]['level'] if end_node in graph.nodes else None

    while unvisited_nodes:
        # 选择当前距离最短的节点
        current_node = min(unvisited_nodes, key=lambda x: distances[x])
        unvisited_nodes.remove(current_node)

        if distances[current_node] == float('inf'):
            break  # 无可达路径

        current_type = graph.nodes[current_node]['type']
        current_level = graph.nodes[current_node]['level']

        for neighbor, edge_info in graph.nodes[current_node]['neighbors'].items():
            weight = edge_info['weight']
            is_corridor_edge = edge_info['is_corridor_edge']
            neighbor_type = graph.nodes[neighbor]['type']

            # -------------------------- 核心约束：必须沿走廊行走 --------------------------
            # 规则1：除了起点和终点，所有中间节点必须是走廊节点
            # 检查如果这不是路径的第一步或最后一步，是否在走廊上
            is_first_step = (previous_nodes[current_node] is None and 
                           distances[current_node] == 0)  # 当前是起点
            is_last_step = (neighbor == end_node)  # 下一步是终点
            
            if not is_first_step and not is_last_step:
                # 中间节点必须是走廊
                if neighbor_type != 'corridor':
                    continue
                
                # 中间步骤必须沿着走廊边缘走
                if not is_corridor_edge:
                    continue

            # 规则2：从非走廊节点（教室/楼梯）出发，下一步必须是走廊
            if current_type != 'corridor' and neighbor_type != 'corridor' and not is_last_step:
                continue

            # 规则3：到达非走廊节点（教室/楼梯）前，必须来自走廊
            if neighbor_type != 'corridor' and current_type != 'corridor' and not is_first_step:
                continue

            # 计算新距离
            new_distance = distances[current_node] + weight

            # 更新最短距离
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node

    return distances, previous_nodes

# 生成路径并验证是否沿走廊
def construct_path(previous_nodes, end_node, graph):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    
    # 验证路径是否符合沿走廊行走的规则
    if len(path) > 2:  # 排除起点和终点相同的情况
        for i in range(1, len(path)-1):  # 检查所有中间节点
            node_type = graph.nodes[path[i]]['type']
            if node_type != 'corridor':
                # 中间节点不是走廊，尝试修复
                return None
    
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

    # 调用改进的Dijkstra算法，生成路径
    distances, previous_nodes = dijkstra(graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node, graph)

    # 如果路径不符合走廊规则，尝试修复
    if not path:
        path, message = force_corridor_path(graph, start_node, end_node)
        if not path:
            return None, message
        total_distance = sum(
            graph.nodes[path[k]]['neighbors'][path[k+1]]['weight']
            for k in range(len(path)-1)
        )
    else:
        total_distance = distances[end_node]

    # 最终验证：确保所有中间节点都是走廊
    if len(path) > 2:
        for node in path[1:-1]:
            if graph.nodes[node]['type'] != 'corridor':
                return None, "❌ 无法找到完全沿走廊的路径"

    return path, f"✅ 沿走廊路径规划成功！总距离：{total_distance:.2f} 单位"

# 强制生成沿走廊的路径
def force_corridor_path(graph, start_node, end_node):
    # 创建只包含走廊、起点和终点的临时图
    temp_graph = Graph()
    
    # 添加起点和终点
    temp_graph.add_node(
        start_node, 
        graph.nodes[start_node]['type'],
        graph.nodes[start_node]['name'],
        graph.nodes[start_node]['level'],
        graph.nodes[start_node]['coordinates']
    )
    temp_graph.add_node(
        end_node, 
        graph.nodes[end_node]['type'],
        graph.nodes[end_node]['name'],
        graph.nodes[end_node]['level'],
        graph.nodes[end_node]['coordinates']
    )
    
    # 添加所有走廊节点
    for node_id in graph.corridor_network:
        node = graph.nodes[node_id]
        temp_graph.add_node(
            node_id, node['type'], node['name'], node['level'], node['coordinates']
        )
    
    # 添加边：只保留走廊相关的边和起点/终点到走廊的边
    for node1 in graph.nodes:
        # 只处理起点、终点和走廊节点
        if node1 != start_node and node1 != end_node and node1 not in graph.corridor_network:
            continue
            
        for node2, edge_info in graph.nodes[node1]['neighbors'].items():
            # 只连接到走廊节点、起点或终点
            if node2 != start_node and node2 != end_node and node2 not in graph.corridor_network:
                continue
                
            temp_graph.add_edge(
                node1, node2, edge_info['weight'], edge_info['is_corridor_edge']
            )
    
    # 在临时图上重新计算路径
    distances, previous_nodes = dijkstra(temp_graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node, temp_graph)
    
    if path and len(path) >= 2:
        # 验证路径是否有效
        valid = True
        for i in range(1, len(path)-1):
            if temp_graph.nodes[path[i]]['type'] != 'corridor':
                valid = False
                break
                
        if valid:
            total_distance = distances[end_node]
            return path, f"✅ 强制沿走廊路径！总距离：{total_distance:.2f} 单位"
    
    return None, "❌ 无法找到沿走廊的有效路径"

# 在3D图上绘制路径（突出显示走廊路径）
def plot_path(ax, graph, path):
    x_coords = []
    y_coords = []
    z_coords = []
    node_types = []

    for node_id in path:
        node = graph.nodes[node_id]
        coords = node['coordinates']
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        z_coords.append(coords[2])
        node_types.append(node['type'])

    # 绘制路径主线（加粗显示沿走廊的路径）
    ax.plot(
        x_coords, y_coords, z_coords,
        color='red', linewidth=5, linestyle='-', marker='o', markersize=8
    )

    # 标记特殊节点
    for i, (x, y, z, node_type) in enumerate(zip(x_coords, y_coords, z_coords, node_types)):
        if i == 0:  # 起点
            ax.scatter(x, y, z, color='green', s=400, marker='*', label='Start')
        elif i == len(path) - 1:  # 终点
            ax.scatter(x, y, z, color='purple', s=400, marker='*', label='End')
        elif node_type == 'stair':  # 楼梯
            ax.scatter(x, y, z, color='red', s=300, marker='^', label='Stair')
        elif node_type == 'corridor':  # 走廊节点（重点突出）
            ax.scatter(x, y, z, color='yellow', s=200, marker='o', label='Corridor')

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
    st.subheader("严格沿走廊行走的路径规划")

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

        nav_button = st.button("🔍 查找沿走廊的路径", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ 3D校园地图")
        
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        if nav_button:
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            if path:
                st.success(message)
                st.markdown("#### 🛤️ 路径详情（所有中间节点均为走廊）")
                for i, node in enumerate(path, 1):
                    if node.startswith("corridor_"):
                        st.write(f"{i}. 走廊节点")
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
