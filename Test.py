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

        # 绘制楼梯（突出显示）
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stairs' if z == 0 else "")
            ax.text(x, y, z+0.1, stair['name'], color='red', fontweight='bold')
            
            # 标记楼梯附近的走廊区域
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
    ax.set_title('School 3D Map (Optimized Destination Path)')
    ax.legend()

    return fig, ax

# 自定义图数据结构
class Graph:
    def __init__(self):
        self.nodes = {}  # key: node_id, value: node_info
        self.stair_proximity = {}  # 走廊与最近楼梯的距离
        self.classroom_proximity = {}  # 走廊与特定教室的距离

    def add_node(self, node_id, node_type, name, level, coordinates, stair_distance=None):
        """添加节点"""
        self.nodes[node_id] = {
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        if node_type == 'corridor' and stair_distance is not None:
            self.stair_proximity[node_id] = stair_distance

    def add_classroom_proximity(self, corridor_id, classroom_id, distance):
        """记录走廊与教室的距离"""
        if corridor_id not in self.classroom_proximity:
            self.classroom_proximity[corridor_id] = {}
        self.classroom_proximity[corridor_id][classroom_id] = distance

    def add_edge(self, node1, node2, weight):
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1]['neighbors'][node2] = weight
            self.nodes[node2]['neighbors'][node1] = weight

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（重点优化终点侧路径）
def build_navigation_graph(school_data):
    graph = Graph()

    # 步骤1：添加所有节点（教室、楼梯、走廊）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']
        
        # 收集当前楼层楼梯坐标
        stair_coords = [stair['coordinates'] for stair in level['stairs']]
        stair_names = [stair['name'] for stair in level['stairs']]

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

        # 1.4 记录走廊与教室的距离（用于终点侧路径优化）
        for classroom in level['classrooms']:
            classroom_id = f"{classroom['name']}@{level_name}"
            classroom_coords = classroom['coordinates']
            
            for corridor_id in graph.nodes:
                corridor_node = graph.nodes[corridor_id]
                if corridor_node['type'] == 'corridor' and corridor_node['level'] == level_name:
                    dist = euclidean_distance(classroom_coords, corridor_node['coordinates'])
                    graph.add_classroom_proximity(corridor_id, classroom_id, dist)

    # 步骤2：添加边（优化权重计算）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 2.1 教室 ↔ 走廊：优先连接最近的走廊
        for classroom in level['classrooms']:
            classroom_node_id = f"{classroom['name']}@{level_name}"
            classroom_coords = classroom['coordinates']
            
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            
            corridor_distances = [
                (node_id, euclidean_distance(classroom_coords, graph.nodes[node_id]['coordinates']))
                for node_id in corridor_nodes
            ]
            corridor_distances.sort(key=lambda x: x[1])
            
            # 对最近的走廊给予权重优势
            for i, (node_id, distance) in enumerate(corridor_distances):
                weight = distance * (0.5 if i < 2 else 1.0)  # 最近的2个走廊权重减半
                graph.add_edge(classroom_node_id, node_id, weight)

        # 2.2 楼梯 ↔ 走廊：优先连接楼梯附近的走廊
        for stair in level['stairs']:
            stair_node_id = f"{stair['name']}@{level_name}"
            stair_coords = stair['coordinates']
            
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            
            corridor_distances = [
                (node_id, euclidean_distance(stair_coords, graph.nodes[node_id]['coordinates']))
                for node_id in corridor_nodes
            ]
            
            # 楼梯附近的走廊权重更低
            for node_id, distance in corridor_distances:
                weight = distance * (0.3 if distance < 5 else 1.0)  # 楼梯5单位内的走廊权重降低
                graph.add_edge(stair_node_id, node_id, weight)

        # 2.3 走廊 ↔ 走廊：优化权重
        corridor_nodes = [
            node_id for node_id in graph.nodes 
            if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
        ]
        
        for i in range(len(corridor_nodes)):
            for j in range(i + 1, len(corridor_nodes)):
                node1 = corridor_nodes[i]
                node2 = corridor_nodes[j]
                
                distance = euclidean_distance(
                    graph.nodes[node1]['coordinates'], 
                    graph.nodes[node2]['coordinates']
                )
                
                # 楼梯附近走廊连接权重降低
                stair_factor = 0.7 if (graph.stair_proximity[node1] < 5 or graph.stair_proximity[node2] < 5) else 1.0
                weight = distance * stair_factor
                graph.add_edge(node1, node2, weight)

    # 2.4 楼梯 ↔ 楼梯：跨楼层连接
    for connection in school_data['buildingA']['connections']:
        from_stair_name, from_level = connection['from']
        to_stair_name, to_level = connection['to']
        
        from_stair_node = f"{from_stair_name}@{from_level}"
        to_stair_node = f"{to_stair_name}@{to_level}"
        
        if from_stair_node in graph.nodes and to_stair_node in graph.nodes:
            graph.add_edge(from_stair_node, to_stair_node, 1.0)

    return graph

# 改进的Dijkstra算法，重点优化终点侧路径
def dijkstra(graph, start_node, end_node):
    # 初始化距离
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    unvisited_nodes = set(graph.nodes.keys())

    # 终点信息
    end_level = graph.nodes[end_node]['level'] if end_node in graph.nodes else None
    end_type = graph.nodes[end_node]['type'] if end_node in graph.nodes else None

    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=lambda x: distances[x])
        unvisited_nodes.remove(current_node)

        if distances[current_node] == float('inf'):
            break

        for neighbor, weight in graph.nodes[current_node]['neighbors'].items():
            # 基础权重因子
            extra_factor = 1.0
            current_node_type = graph.nodes[current_node]['type']
            current_level = graph.nodes[current_node]['level']
            
            # 终点侧路径优化逻辑
            if end_type == 'classroom' and current_level == end_level:
                # 1. 优先到达终点楼层的楼梯
                if current_node_type == 'corridor' and any(
                    'stair' in n for n in graph.nodes[current_node]['neighbors']
                ) and not any(
                    graph.nodes[n]['type'] == 'stair' for n in previous_nodes.values() if n is not None
                ):
                    extra_factor *= 0.6  # 接近楼梯时权重降低
                
                # 2. 到达楼梯后，优先到楼梯临近走廊
                if current_node_type == 'stair':
                    extra_factor *= 0.5  # 楼梯节点权重降低
                
                # 3. 楼梯附近走廊后，优先到终点教室临近走廊
                if current_node_type == 'corridor' and graph.stair_proximity.get(current_node, float('inf')) < 5:
                    # 离终点教室越近的走廊权重越低
                    class_dist = graph.classroom_proximity.get(current_node, {}).get(end_node, float('inf'))
                    if class_dist < float('inf'):
                        extra_factor *= 0.5 + (min(class_dist, 10) / 10) * 0.5  # 0.5-1.0之间
            
            new_distance = distances[current_node] + weight * extra_factor
            
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

# 验证终点侧路径是否符合要求
def validate_destination_path(graph, path, end_node):
    if len(path) < 4 or end_node != path[-1]:
        return False
        
    end_level = graph.nodes[end_node]['level']
    # 找到进入终点楼层的位置
    dest_entry_idx = None
    for i in range(1, len(path)):
        if graph.nodes[path[i]]['level'] == end_level and graph.nodes[path[i-1]]['level'] != end_level:
            dest_entry_idx = i
            break
            
    if dest_entry_idx is None:  # 同楼层
        dest_entry_idx = 0
        
    # 从进入终点楼层开始的子路径
    dest_subpath = path[dest_entry_idx:]
    
    # 检查是否符合: 楼梯 → 楼梯临近走廊 → 终点教室临近走廊 → 终点教室
    stair_found = False
    stair_corridor_found = False
    class_corridor_found = False
    
    for node in dest_subpath:
        node_type = graph.nodes[node]['type']
        
        if not stair_found:
            if node_type == 'stair':
                stair_found = True
        elif not stair_corridor_found:
            if node_type == 'corridor' and graph.stair_proximity.get(node, float('inf')) < 5:
                stair_corridor_found = True
        elif not class_corridor_found:
            if node_type == 'corridor':
                class_dist = graph.classroom_proximity.get(node, {}).get(end_node, float('inf'))
                if class_dist < 5:  # 终点教室5单位内的走廊
                    class_corridor_found = True
                    # 下一个节点应该是终点教室
                    if dest_subpath.index(node) + 1 < len(dest_subpath) and dest_subpath[dest_subpath.index(node) + 1] == end_node:
                        return True
                    
    return False

# 导航函数
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    start_node = f"{start_classroom}@{start_level}"
    end_node = f"{end_classroom}@{end_level}"

    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "❌ 无效的教室或楼层"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 第一次计算路径
    distances, previous_nodes = dijkstra(graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node)

    # 验证路径是否符合所有要求
    valid = True
    
    # 验证起点是否先到走廊
    if len(path) >= 2 and graph.nodes[path[1]]['type'] != 'corridor':
        valid = False
    
    # 验证终点侧路径是否符合要求
    if not validate_destination_path(graph, path, end_node):
        valid = False
    
    # 如果路径无效，使用强制算法
    if not valid:
        return force_standard_path(graph, start_node, end_node)
    
    if path:
        total_distance = distances[end_node]
        return path, f"✅ 路径规划成功！总距离：{total_distance:.2f} 单位"
    else:
        return None, "❌ 无有效路径"

# 强制生成符合标准的路径
def force_standard_path(graph, start_node, end_node):
    # 创建临时图，强化终点侧路径约束
    temp_graph = Graph()
    for node_id, node_info in graph.nodes.items():
        temp_graph.add_node(
            node_id=node_id,
            node_type=node_info['type'],
            name=node_info['name'],
            level=node_info['level'],
            coordinates=node_info['coordinates']
        )
        # 复制走廊与教室的距离信息
        if node_id in graph.classroom_proximity:
            for class_id, dist in graph.classroom_proximity[node_id].items():
                temp_graph.add_classroom_proximity(node_id, class_id, dist)
    
    # 复制边，对终点侧关键路径给予极低权重
    end_level = graph.nodes[end_node]['level']
    for node1 in graph.nodes:
        for node2, weight in graph.nodes[node1]['neighbors'].items():
            # 对终点楼层的关键路径给予权重优势
            if graph.nodes[node1]['level'] == end_level:
                # 楼梯到楼梯附近走廊
                if (graph.nodes[node1]['type'] == 'stair' and 
                    graph.nodes[node2]['type'] == 'corridor' and 
                    graph.stair_proximity.get(node2, float('inf')) < 5):
                    weight *= 0.3
                    
                # 楼梯附近走廊到终点教室附近走廊
                if (graph.nodes[node1]['type'] == 'corridor' and 
                    graph.stair_proximity.get(node1, float('inf')) < 5 and
                    graph.nodes[node2]['type'] == 'corridor' and
                    graph.classroom_proximity.get(node2, {}).get(end_node, float('inf')) < 5):
                    weight *= 0.3
                    
                # 终点教室附近走廊到终点教室
                if (graph.nodes[node1]['type'] == 'corridor' and 
                    graph.classroom_proximity.get(node1, {}).get(end_node, float('inf')) < 5 and
                    node2 == end_node):
                    weight *= 0.3
            
            temp_graph.add_edge(node1, node2, weight)

    # 重新计算路径
    distances, previous_nodes = dijkstra(temp_graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node)

    if path and validate_destination_path(temp_graph, path, end_node):
        total_distance = distances[end_node]
        return path, f"✅ 已生成符合标准的路径！总距离：{total_distance:.2f} 单位"
    else:
        return None, "❌ 无法生成符合要求的路径"

# 在3D图上绘制路径（突出显示终点侧路径）
def plot_path(ax, graph, path, end_node):
    x_coords = []
    y_coords = []
    z_coords = []
    node_types = []
    node_details = []  # 存储节点详细信息

    end_level = graph.nodes[end_node]['level']
    # 找到进入终点楼层的位置
    dest_entry_idx = 0
    for i in range(1, len(path)):
        if graph.nodes[path[i]]['level'] == end_level and graph.nodes[path[i-1]]['level'] != end_level:
            dest_entry_idx = i
            break

    for idx, node_id in enumerate(path):
        node = graph.nodes[node_id]
        coords = node['coordinates']
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        z_coords.append(coords[2])
        node_types.append(node['type'])
        
        # 标记特殊节点
        detail = ""
        if node['type'] == 'corridor':
            if idx == 1:  # 起点后的第一个走廊
                detail = "start_near_classroom"
            # 终点侧路径标记
            elif idx >= dest_entry_idx:
                if graph.stair_proximity.get(node_id, float('inf')) < 5:
                    detail = "dest_stair_corridor"
                elif graph.classroom_proximity.get(node_id, {}).get(end_node, float('inf')) < 5:
                    detail = "dest_class_corridor"
        elif node['type'] == 'stair' and idx >= dest_entry_idx:
            detail = "dest_stair"
            
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
        elif detail == "dest_stair":  # 终点楼层的楼梯
            ax.scatter(x, y, z, color='red', s=250, marker='^', label='Destination Stair')
        elif detail == "dest_stair_corridor":  # 终点楼梯临近走廊
            ax.scatter(x, y, z, color='orange', s=180, marker='o', label='Near Stair (Dest)')
        elif detail == "dest_class_corridor":  # 终点教室临近走廊
            ax.scatter(x, y, z, color='magenta', s=180, marker='o', label='Near Classroom (Dest)')
        elif detail == "start_near_classroom":  # 起点临近走廊
            ax.scatter(x, y, z, color='cyan', s=150, marker='o', label='Near Classroom (Start)')
        elif node_type == 'stair':  # 其他楼梯
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stair')
        elif node_type == 'corridor':  # 其他走廊
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
    st.subheader("3D地图与精细化路径规划（优化终点侧路径）")

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
            end_node = f"{end_classroom}@{end_level}"
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            if path:
                st.success(message)
                st.markdown("#### 🛤️ 路径详情")
                
                # 确定终点楼层
                end_level = nav_graph.nodes[end_node]['level']
                dest_entry_idx = 0
                for i in range(1, len(path)):
                    if nav_graph.nodes[path[i]]['level'] == end_level and nav_graph.nodes[path[i-1]]['level'] != end_level:
                        dest_entry_idx = i
                        break
                
                for i, node in enumerate(path, 1):
                    if node.startswith("corridor_"):
                        # 识别特殊走廊节点
                        if i == 2:  # 起点后的第一个走廊
                            st.write(f"{i}. 起点临近走廊")
                        elif i > dest_entry_idx:
                            if nav_graph.stair_proximity.get(node, float('inf')) < 5:
                                st.write(f"{i}. 终点楼梯临近走廊")
                            elif nav_graph.classroom_proximity.get(node, {}).get(end_node, float('inf')) < 5:
                                st.write(f"{i}. 终点教室临近走廊")
                            else:
                                st.write(f"{i}. 走廊")
                        else:
                            st.write(f"{i}. 走廊")
                    elif 'stair' in node:
                        if i > dest_entry_idx:
                            st.write(f"{i}. 终点楼层楼梯")
                        else:
                            st.write(f"{i}. 楼梯")
                    else:
                        room, floor = node.split('@')
                        st.write(f"{i}. {room}（楼层：{floor}）")
                
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, nav_graph, path, end_node)
                st.session_state['fig'] = fig
            else:
                st.error(message)
        
        st.pyplot(st.session_state['fig'])

if __name__ == "__main__":
    main()
    
