import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
from collections import defaultdict

# 基础配置
plt.switch_backend('Agg')

# 读取JSON数据
def load_school_data_detailed(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        return None

# 绘制3D地图
def plot_3d_map(school_data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同楼层使用不同颜色
    floor_colors = {0: 'blue', 2: 'green', 5: 'orange', 10: 'purple'}  

    # 处理每个楼层
    for level in school_data['buildingA']['levels']:
        z = level['z']
        color = floor_colors.get(z, 'gray')
        level_name = level['name']

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
        ax.plot(x_plane, y_plane, z_plane, color=color, linewidth=2, label=level_name)

        # 绘制走廊
        for corridor in level['corridors']:
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            ax.plot(x, y, z_coords, color=color, linewidth=5, alpha=0.8)

        # 绘制走廊交叉点（特殊标记）
        if 'corridor_intersections' in level:
            for intersection in level['corridor_intersections']:
                x, y, _ = intersection['coordinates']
                ax.scatter(x, y, z, color='yellow', s=300, marker='X', label='Corridor Intersection' if z == 0 else "")
                ax.text(x, y, z+0.1, f"X{intersection['id']}", color='black', fontweight='bold')

        # 绘制楼梯（突出显示）
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stairs' if z == 0 else "")
            ax.text(x, y, z+0.1, stair['name'], color='red', fontweight='bold')
            ax.scatter(x, y, z, color='red', s=800, alpha=0.2, marker='o')

        # 绘制教室
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']
            ax.text(x, y, z, classroom['name'], color='black', fontweight='bold')
            ax.scatter(x, y, z, color=color, s=50)
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color=color, linestyle='--')

    # 设置坐标轴
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Floor')
    ax.set_title('School 3D Map (Path: Classroom→Intersection→Stairs→Intersection→Classroom)')
    ax.legend()

    return fig, ax

# 自定义图数据结构
class Graph:
    def __init__(self):
        self.nodes = {}  # key: node_id, value: node_info
        self.stair_proximity = {}  # 记录走廊节点与最近楼梯的距离
        self.intersections = set()  # 走廊交叉点节点ID集合

    def add_node(self, node_id, node_type, name, level, coordinates, stair_distance=None, is_intersection=False):
        self.nodes[node_id] = {
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        if node_type == 'corridor' and stair_distance is not None:
            self.stair_proximity[node_id] = stair_distance
        if is_intersection:
            self.intersections.add(node_id)

    def add_edge(self, node1, node2, weight, is_intersection_edge=False):
        if node1 in self.nodes and node2 in self.nodes:
            # 为经过交叉点的边设置较低权重，优先选择
            adjusted_weight = weight * 0.5 if is_intersection_edge else weight
            self.nodes[node1]['neighbors'][node2] = adjusted_weight
            self.nodes[node2]['neighbors'][node1] = adjusted_weight

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 检测走廊交叉点（两条走廊线段的交点）
def detect_corridor_intersections(corridors):
    intersections = []
    intersection_id = 1
    
    # 提取所有走廊线段
    all_segments = []
    for corridor_idx, corridor in enumerate(corridors):
        points = corridor['points']
        for i in range(len(points) - 1):
            all_segments.append((
                corridor_idx,
                (points[i][0], points[i][1]),
                (points[i+1][0], points[i+1][1])
            ))
    
    # 检查线段间是否相交
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    def segments_intersect(a1, a2, b1, b2):
        return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)
    
    def segment_intersection(a1, a2, b1, b2):
        # 计算两条线段的交点
        x1, y1 = a1
        x2, y2 = a2
        x3, y3 = b1
        x4, y4 = b2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None  # 平行线
        
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t = t_num / denom
        u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
        u = u_num / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        return None
    
    # 检查所有线段对
    seen = set()
    for i in range(len(all_segments)):
        corridor1, a1, a2 = all_segments[i]
        for j in range(i + 1, len(all_segments)):
            corridor2, b1, b2 = all_segments[j]
            if corridor1 == corridor2:
                continue  # 同一条走廊的线段不检测
            
            key = frozenset([i, j])
            if key in seen:
                continue
            seen.add(key)
            
            if segments_intersect(a1, a2, b1, b2):
                point = segment_intersection(a1, a2, b1, b2)
                if point:
                    intersections.append({
                        'id': intersection_id,
                        'coordinates': (point[0], point[1], 0),  # z坐标后续会设置
                        'corridors': [corridor1, corridor2]
                    })
                    intersection_id += 1
    
    return intersections

# 构建导航图（包含走廊交叉点）
def build_navigation_graph(school_data):
    if not school_data:
        return None
        
    graph = Graph()

    # 步骤1：添加所有节点（教室、楼梯、走廊、交叉点）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']
        
        # 检测或获取走廊交叉点
        if 'corridor_intersections' not in level:
            level['corridor_intersections'] = detect_corridor_intersections(level['corridors'])
        else:
            # 确保z坐标正确
            for intersection in level['corridor_intersections']:
                intersection['coordinates'] = (
                    intersection['coordinates'][0],
                    intersection['coordinates'][1],
                    z
                )
        
        # 收集当前楼层楼梯坐标
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

        # 1.3 添加走廊交叉点节点（优先添加，确保路径经过）
        for intersection in level['corridor_intersections']:
            coords = intersection['coordinates']
            node_id = f"intersection_{intersection['id']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='corridor',
                name=f"Intersection {intersection['id']}",
                level=level_name,
                coordinates=coords,
                stair_distance=min(euclidean_distance(coords, sc) for sc in stair_coords) if stair_coords else 0,
                is_intersection=True
            )

        # 1.4 添加走廊节点
        for corridor_idx, corridor in enumerate(level['corridors']):
            for point_idx, point in enumerate(corridor['points']):
                node_id = f"corridor_{corridor_idx}_{point_idx}@{level_name}"
                if node_id not in graph.nodes:
                    min_stair_dist = min(euclidean_distance(point, sc) for sc in stair_coords) if stair_coords else 0
                    graph.add_node(
                        node_id=node_id,
                        node_type='corridor',
                        name=f"Corridor_{corridor_idx+1}_Point_{point_idx+1}",
                        level=level_name,
                        coordinates=point,
                        stair_distance=min_stair_dist
                    )

    # 步骤2：添加边（优先连接交叉点）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 2.1 教室 ↔ 走廊/交叉点（只连接到走廊，不直接连楼梯）
        for classroom in level['classrooms']:
            classroom_node_id = f"{classroom['name']}@{level_name}"
            classroom_coords = classroom['coordinates']
            
            # 优先连接到交叉点
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            
            # 分离交叉点和普通走廊节点
            intersection_nodes = [n for n in corridor_nodes if n in graph.intersections]
            regular_corridor_nodes = [n for n in corridor_nodes if n not in graph.intersections]
            
            # 先连接到交叉点，再连接到普通走廊
            all_corridor_nodes = intersection_nodes + regular_corridor_nodes
            
            corridor_distances = [
                (node_id, euclidean_distance(classroom_coords, graph.nodes[node_id]['coordinates']))
                for node_id in all_corridor_nodes
            ]
            corridor_distances.sort(key=lambda x: x[1])
            
            # 连接最近的1个交叉点（如果有）和1个普通走廊
            connected = 0
            for node_id, distance in corridor_distances:
                if connected >= 2:
                    break
                # 交叉点权重更低，优先选择
                weight_factor = 0.3 if node_id in graph.intersections else 0.5
                graph.add_edge(classroom_node_id, node_id, distance * weight_factor)
                connected += 1

        # 2.2 楼梯 ↔ 走廊/交叉点（楼梯只连接走廊）
        for stair in level['stairs']:
            stair_node_id = f"{stair['name']}@{level_name}"
            stair_coords = stair['coordinates']
            
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            
            # 优先连接到交叉点
            intersection_nodes = [n for n in corridor_nodes if n in graph.intersections]
            regular_corridor_nodes = [n for n in corridor_nodes if n not in graph.intersections]
            all_corridor_nodes = intersection_nodes + regular_corridor_nodes
            
            corridor_distances = [
                (node_id, euclidean_distance(stair_coords, graph.nodes[node_id]['coordinates']))
                for node_id in all_corridor_nodes
            ]
            
            for node_id, distance in corridor_distances:
                # 交叉点权重更低
                weight_factor = 0.2 if node_id in graph.intersections else (0.3 if distance < 5 else 1.0)
                graph.add_edge(stair_node_id, node_id, distance * weight_factor, 
                              is_intersection_edge=node_id in graph.intersections)

        # 2.3 走廊节点之间的连接（强调交叉点连接）
        corridor_nodes = [
            node_id for node_id in graph.nodes 
            if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
        ]
        
        # 优先连接交叉点之间的边
        intersection_nodes = [n for n in corridor_nodes if n in graph.intersections]
        for i in range(len(intersection_nodes)):
            for j in range(i + 1, len(intersection_nodes)):
                node1 = intersection_nodes[i]
                node2 = intersection_nodes[j]
                
                distance = euclidean_distance(
                    graph.nodes[node1]['coordinates'], 
                    graph.nodes[node2]['coordinates']
                )
                
                # 交叉点之间的边权重最低，确保优先选择
                graph.add_edge(node1, node2, distance * 0.1, is_intersection_edge=True)
        
        # 连接所有走廊节点（包括交叉点和普通节点）
        for i in range(len(corridor_nodes)):
            for j in range(i + 1, len(corridor_nodes)):
                node1 = corridor_nodes[i]
                node2 = corridor_nodes[j]
                
                # 已经连接过的交叉点对跳过
                if node1 in graph.intersections and node2 in graph.intersections:
                    continue
                
                distance = euclidean_distance(
                    graph.nodes[node1]['coordinates'], 
                    graph.nodes[node2]['coordinates']
                )
                
                # 包含交叉点的边权重较低
                is_intersection = node1 in graph.intersections or node2 in graph.intersections
                stair_factor = 0.7 if (graph.stair_proximity.get(node1, float('inf')) < 5 or 
                                      graph.stair_proximity.get(node2, float('inf')) < 5) else 1.0
                weight = distance * stair_factor * (0.3 if is_intersection else 1.0)
                graph.add_edge(node1, node2, weight, is_intersection_edge=is_intersection)

    # 2.4 楼梯 ↔ 楼梯：跨楼层连接
    for connection in school_data['buildingA']['connections']:
        from_stair_name, from_level = connection['from']
        to_stair_name, to_level = connection['to']
        
        from_stair_node = f"{from_stair_name}@{from_level}"
        to_stair_node = f"{to_stair_name}@{to_level}"
        
        if from_stair_node in graph.nodes and to_stair_node in graph.nodes:
            graph.add_edge(from_stair_node, to_stair_node, 1.0)

    return graph

# 带路径阶段约束的Dijkstra算法 - 确保经过交叉点
def constrained_dijkstra(graph, start_node, end_node):
    # 验证输入节点是否有效
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, None
        
    # 确定起点和终点楼层
    start_level = graph.nodes[start_node]['level']
    end_level = graph.nodes[end_node]['level']
    need_stairs = start_level != end_level  # 是否需要跨楼层（经过楼梯）
    
    # 获取起点和终点所在楼层的交叉点
    start_level_intersections = [n for n in graph.intersections 
                               if graph.nodes[n]['level'] == start_level]
    end_level_intersections = [n for n in graph.intersections 
                             if graph.nodes[n]['level'] == end_level]
    
    # 初始化距离和路径阶段跟踪
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    
    # 路径阶段跟踪：0=起点教室, 1=已到走廊, 2=已到交叉点, 3=已到楼梯, 4=目标楼层交叉点, 5=目标楼层走廊, 6=已到终点
    path_phase = {node: 0 for node in graph.nodes}
    path_phase[start_node] = 0  # 起点是教室（阶段0）
    
    unvisited_nodes = set(graph.nodes.keys())

    while unvisited_nodes:
        # 找到距离最小的未访问节点
        current_node = None
        min_distance = float('inf')
        for node in unvisited_nodes:
            if distances[node] < min_distance:
                min_distance = distances[node]
                current_node = node
        
        if current_node is None:
            break  # 没有可达节点了
        if current_node == end_node:
            break  # 到达终点
        if min_distance == float('inf'):
            break  # 无法继续前进
        
        unvisited_nodes.remove(current_node)

        # 获取当前节点属性
        current_level = graph.nodes[current_node]['level']
        current_type = graph.nodes[current_node]['type']
        current_phase = path_phase[current_node]
        is_current_intersection = current_node in graph.intersections

        # 遍历邻居节点
        for neighbor, weight in graph.nodes[current_node]['neighbors'].items():
            if neighbor not in unvisited_nodes:
                continue  # 已访问节点跳过
                
            neighbor_type = graph.nodes[neighbor]['type']
            neighbor_level = graph.nodes[neighbor]['level']
            is_neighbor_intersection = neighbor in graph.intersections
            new_phase = current_phase
            valid_transition = False
            
            # 阶段转换规则（强制经过交叉点）
            if current_phase == 0:  # 起点教室
                if neighbor_type == 'corridor':
                    new_phase = 1  # 先到走廊
                    valid_transition = True
            
            elif current_phase == 1:  # 走廊
                if neighbor_type == 'corridor':
                    if is_neighbor_intersection:
                        new_phase = 2  # 到达交叉点（必须经过）
                    else:
                        new_phase = 1  # 继续在走廊
                    valid_transition = True
                elif neighbor_type == 'stair' and need_stairs and is_current_intersection:
                    # 只有经过交叉点后才能去楼梯
                    new_phase = 3  # 到达楼梯
                    valid_transition = True
            
            elif current_phase == 2:  # 交叉点
                if neighbor_type == 'corridor':
                    new_phase = 1  # 返回走廊
                    valid_transition = True
                elif neighbor_type == 'stair' and need_stairs:
                    new_phase = 3  # 到达楼梯
                    valid_transition = True
            
            elif current_phase == 3:  # 楼梯
                if neighbor_type == 'stair':
                    new_phase = 3  # 跨楼层楼梯
                    valid_transition = True
                elif neighbor_type == 'corridor' and neighbor_level == end_level:
                    if is_neighbor_intersection:
                        new_phase = 4  # 到达目标楼层交叉点
                    else:
                        new_phase = 5  # 到达目标楼层走廊
                    valid_transition = True
            
            elif current_phase == 4:  # 目标楼层交叉点
                if neighbor_type == 'corridor':
                    new_phase = 5  # 到目标楼层走廊
                    valid_transition = True
            
            elif current_phase == 5:  # 目标楼层走廊
                if neighbor_type == 'corridor':
                    if is_neighbor_intersection:
                        new_phase = 4  # 到达目标楼层交叉点
                    else:
                        new_phase = 5  # 继续在目标楼层走廊
                    valid_transition = True
                elif neighbor == end_node:
                    new_phase = 6  # 到达终点
                    valid_transition = True

            # 更新距离
            if valid_transition:
                new_distance = distances[current_node] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_nodes[neighbor] = current_node
                    path_phase[neighbor] = new_phase

    return distances, previous_nodes

# 生成最短路径
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes.get(current_node)  # 使用get避免KeyError
    return path if len(path) > 0 and path[0] != end_node else None

# 验证路径是否符合规定顺序（必须经过交叉点）
def validate_path_order(graph, path):
    if not path or len(path) < 2:
        return False, "路径太短"
    
    # 提取路径类型序列和交叉点信息
    path_types = [graph.nodes[node]['type'] for node in path]
    path_intersections = [node in graph.intersections for node in path]
    start_level = graph.nodes[path[0]]['level']
    end_level = graph.nodes[path[-1]]['level']
    need_stairs = start_level != end_level
    
    # 检查起点和终点是否为教室
    if path_types[0] != 'classroom' or path_types[-1] != 'classroom':
        return False, "起点和终点必须是教室"
    
    # 检查是否经过交叉点
    if not any(path_intersections):
        return False, "路径必须经过走廊交叉点"
    
    # 检查起点楼层是否经过交叉点
    start_level_nodes = [i for i, node in enumerate(path) 
                        if graph.nodes[node]['level'] == start_level]
    start_intersections = any(path_intersections[i] for i in start_level_nodes)
    if not start_intersections:
        return False, f"起点楼层({start_level})必须经过走廊交叉点"
    
    # 检查终点楼层是否经过交叉点（如果与起点不同楼层）
    if need_stairs:
        end_level_nodes = [i for i, node in enumerate(path) 
                         if graph.nodes[node]['level'] == end_level]
        end_intersections = any(path_intersections[i] for i in end_level_nodes)
        if not end_intersections:
            return False, f"终点楼层({end_level})必须经过走廊交叉点"
    
    # 检查是否先到走廊
    if path_types[1] != 'corridor':
        return False, "必须先从教室到走廊"
    
    # 检查跨楼层时是否经过楼梯
    if need_stairs and 'stair' not in path_types:
        return False, "跨楼层路径必须经过楼梯"
    
    # 检查最后一步是否从走廊到教室
    if path_types[-2] != 'corridor':
        return False, "最后必须从走廊到教室"
    
    return True, "路径顺序有效"

# 导航函数
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    if not graph:
        return None, "❌ 导航图未初始化"
        
    start_node = f"{start_classroom}@{start_level}"
    end_node = f"{end_classroom}@{end_level}"

    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "❌ 无效的教室或楼层"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 使用带约束的Dijkstra算法
    distances, previous_nodes = constrained_dijkstra(graph, start_node, end_node)
    if not distances or not previous_nodes:
        return None, "❌ 路径计算失败"
        
    path = construct_path(previous_nodes, end_node)
    if not path:
        return None, "❌ 无法构建路径"

    # 验证路径顺序，如果不符合则强制修正
    is_valid, message = validate_path_order(graph, path)
    if not is_valid:
        st.warning(f"路径顺序调整: {message}")
        return force_valid_path(graph, start_node, end_node)
    
    total_distance = distances[end_node]
    return path, f"✅ 路径规划成功！总距离：{total_distance:.2f} 单位"

# 强制生成符合顺序的路径（确保经过交叉点）
def force_valid_path(graph, start_node, end_node):
    start_level = graph.nodes[start_node]['level']
    end_level = graph.nodes[end_node]['level']
    need_stairs = start_level != end_level
    
    # 获取起点和终点楼层的交叉点
    start_intersections = [n for n in graph.intersections 
                          if graph.nodes[n]['level'] == start_level]
    end_intersections = [n for n in graph.intersections 
                        if graph.nodes[n]['level'] == end_level]
    
    if not start_intersections:
        return None, "❌ 起点楼层没有走廊交叉点"
    if need_stairs and not end_intersections:
        return None, "❌ 终点楼层没有走廊交叉点"
    
    # 1. 找到起点教室到最近交叉点的路径
    start_corridors = [n for n in graph.nodes[start_node]['neighbors'] 
                      if graph.nodes[n]['type'] == 'corridor']
    start_intersection_distances = [(n, graph.nodes[start_node]['neighbors'][n]) 
                                   for n in start_corridors if n in start_intersections]
    
    # 如果没有直接连接到交叉点，找最近的交叉点
    if not start_intersection_distances:
        start_intersection_distances = [
            (n, euclidean_distance(graph.nodes[start_node]['coordinates'], graph.nodes[n]['coordinates']))
            for n in start_intersections
        ]
    nearest_start_intersection = min(start_intersection_distances, key=lambda x: x[1])[0]
    
    # 2. 找到终点教室到最近交叉点的路径
    end_corridors = [n for n in graph.nodes[end_node]['neighbors'] 
                    if graph.nodes[n]['type'] == 'corridor']
    end_intersection_distances = [(n, graph.nodes[end_node]['neighbors'][n]) 
                                 for n in end_corridors if n in end_intersections]
    
    if not end_intersection_distances:
        end_intersection_distances = [
            (n, euclidean_distance(graph.nodes[end_node]['coordinates'], graph.nodes[n]['coordinates']))
            for n in end_intersections
        ]
    nearest_end_intersection = min(end_intersection_distances, key=lambda x: x[1])[0]
    
    # 3. 如果需要跨楼层，找到连接的楼梯（经过交叉点）
    stair_path = []
    if need_stairs:
        # 找到起点楼层交叉点到楼梯的路径
        start_stairs = [n for n in graph.nodes if graph.nodes[n]['type'] == 'stair' 
                       and graph.nodes[n]['level'] == start_level]
        stair_from_intersection = []
        for s in start_stairs:
            for n in start_intersections:
                if s in graph.nodes[n]['neighbors']:
                    stair_from_intersection.append((n, s, graph.nodes[n]['neighbors'][s]))
        
        if not stair_from_intersection:
            return None, "❌ 起点楼层交叉点与楼梯无连接"
        best_start = min(stair_from_intersection, key=lambda x: x[2])
        start_intersection, start_stair = best_start[0], best_start[1]
        
        # 找到终点楼层楼梯到交叉点的路径
        end_stairs = [n for n in graph.nodes if graph.nodes[n]['type'] == 'stair' 
                     and graph.nodes[n]['level'] == end_level]
        stair_to_intersection = []
        for s in end_stairs:
            for n in end_intersections:
                if s in graph.nodes[n]['neighbors']:
                    stair_to_intersection.append((s, n, graph.nodes[s]['neighbors'][n]))
        
        if not stair_to_intersection:
            return None, "❌ 终点楼层楼梯与交叉点无连接"
        best_end = min(stair_to_intersection, key=lambda x: x[2])
        end_stair, end_intersection = best_end[0], best_end[1]
        
        # 找到连接的楼梯对
        connected_stairs = []
        for s1 in start_stairs:
            for s2 in end_stairs:
                if s2 in graph.nodes[s1]['neighbors']:
                    connected_stairs.append((s1, s2))
        
        if not connected_stairs:
            return None, "❌ 楼层之间没有连接的楼梯"
        
        # 找到起点交叉点到起点楼梯的路径
        dist1, prev1 = constrained_dijkstra(graph, nearest_start_intersection, start_stair)
        path1 = construct_path(prev1, start_stair) if prev1 else []
        
        # 找到终点楼梯到终点交叉点的路径
        dist2, prev2 = constrained_dijkstra(graph, end_stair, nearest_end_intersection)
        path2 = construct_path(prev2, nearest_end_intersection) if prev2 else []
        
        # 找到连接楼梯
        stair_connection = next((s for s in connected_stairs if s[0] == start_stair and s[1] == end_stair), None)
        if not stair_connection:
            stair_connection = connected_stairs[0]  # 退而求其次
        
        stair_path = path1[1:] + [stair_connection[1]] + path2[1:]
    
    # 4. 如果不需要跨楼层，直接连接交叉点
    else:
        dist, prev = constrained_dijkstra(graph, nearest_start_intersection, nearest_end_intersection)
        stair_path = construct_path(prev, nearest_end_intersection)[1:] if prev else []
    
    # 组合完整路径
    full_path = [start_node, nearest_start_intersection] + stair_path + [end_node]
    
    # 去重
    seen = set()
    full_path = [node for node in full_path if not (node in seen or seen.add(node))]
    
    return full_path, "✅ 已生成经过走廊交叉点的路径"

# 在3D图上绘制路径
def plot_path(ax, graph, path):
    if not path:
        return
        
    x_coords = []
    y_coords = []
    z_coords = []
    node_types = []
    is_intersection = []

    for node_id in path:
        if node_id not in graph.nodes:
            continue
        node = graph.nodes[node_id]
        coords = node['coordinates']
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        z_coords.append(coords[2])
        node_types.append(node['type'])
        is_intersection.append(node_id in graph.intersections)

    # 绘制路径主线
    ax.plot(
        x_coords, y_coords, z_coords,
        color='red', linewidth=3, linestyle='-', marker='o', markersize=6
    )

    # 标记特殊节点
    for i, (x, y, z, node_type, intersection) in enumerate(zip(
            x_coords, y_coords, z_coords, node_types, is_intersection)):
        if i == 0:  # 起点教室
            ax.scatter(x, y, z, color='green', s=300, marker='*', label='Start Classroom')
        elif i == len(path) - 1:  # 终点教室
            ax.scatter(x, y, z, color='purple', s=300, marker='*', label='End Classroom')
        elif node_type == 'stair':  # 楼梯
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Staircase')
        elif intersection:  # 走廊交叉点
            ax.scatter(x, y, z, color='yellow', s=250, marker='X', label='Corridor Intersection')
        elif node_type == 'corridor':  # 普通走廊
            ax.scatter(x, y, z, color='blue', s=100, marker='o', label='Corridor')

    ax.legend()

# 获取所有楼层和教室信息
def get_classroom_info(school_data):
    if not school_data:
        return [], {}
        
    levels = []
    classrooms_by_level = {}
    
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        levels.append(level_name)
        classrooms = [classroom['name'] for classroom in level['classrooms']]
        classrooms_by_level[level_name] = classrooms
        
    return levels, classrooms_by_level

# Streamlit界面逻辑
def main():
    st.title("🏫 校园导航系统")
    st.subheader("强制路径顺序：教室→走廊→交叉点→楼梯→交叉点→走廊→教室")

    try:
        # 尝试加载数据
        school_data = load_school_data_detailed('school_data_detailed.json')
        if not school_data:
            st.error("❌ 无法继续，缺少校园数据")
            return
            
        nav_graph = build_navigation_graph(school_data)
        if not nav_graph:
            st.error("❌ 无法构建导航图")
            return
            
        levels, classrooms_by_level = get_classroom_info(school_data)
        if not levels:
            st.error("❌ 未找到任何楼层信息")
            return
            
        st.success("✅ 校园数据加载成功！")
    except Exception as e:
        st.error(f"❌ 初始化错误: {str(e)}")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📍 选择位置")
        
        st.markdown("#### 起点")
        start_level = st.selectbox("楼层", levels, key="start_level")
        start_classrooms = classrooms_by_level.get(start_level, [])
        if not start_classrooms:
            st.warning("该楼层没有教室信息")
            return
        start_classroom = st.selectbox("教室", start_classrooms, key="start_classroom")

        st.markdown("#### 终点")
        end_level = st.selectbox("楼层", levels, key="end_level")
        end_classrooms = classrooms_by_level.get(end_level, [])
        if not end_classrooms:
            st.warning("该楼层没有教室信息")
            return
        end_classroom = st.selectbox("教室", end_classrooms, key="end_classroom")

        nav_button = st.button("🔍 查找最优路径", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ 3D校园地图")
        
        # 初始化地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        if nav_button:
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            if path:
                st.success(message)
                st.markdown("#### 🛤️ 路径详情（按顺序）")
                
                # 解析路径阶段并显示
                path_phases = []
                for i, node in enumerate(path):
                    if node not in nav_graph.nodes:
                        continue
                    node_type = nav_graph.nodes[node]['type']
                    is_intersection = node in nav_graph.intersections
                    
                    if i == 0:
                        path_phases.append(f"{i+1}. 起点教室: {node.split('@')[0]}")
                    elif i == len(path)-1:
                        path_phases.append(f"{i+1}. 终点教室: {node.split('@')[0]}")
                    elif node_type == 'stair':
                        path_phases.append(f"{i+1}. 楼梯: {node.split('@')[0]}")
                    elif is_intersection:
                        path_phases.append(f"{i+1}. 走廊交叉点: {nav_graph.nodes[node]['name']}")
                    else:  # corridor
                        path_phases.append(f"{i+1}. 走廊")
                
                for phase in path_phases:
                    st.write(phase)
                
                # 重新绘制地图和路径
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, nav_graph, path)
                st.session_state['fig'] = fig
            else:
                st.error(message)
        
        # 显示地图
        st.pyplot(st.session_state['fig'])

if __name__ == "__main__":
    main()
    
