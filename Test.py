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

# 绘制3D地图（增强走廊-楼梯-教室衔接可视化）
def plot_3d_map(school_data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同楼层使用不同颜色（走廊/楼层边框同色，增强辨识度）
    floor_colors = {0: '#1E88E5', 2: '#43A047', 5: '#FB8C00'}  
    corridor_linewidth = 6  # 加粗走廊线条，突出导航主通道

    # 处理每个楼层
    for level in school_data['buildingA']['levels']:
        z = level['z']
        color = floor_colors.get(z, 'gray')
        level_name = level['name']

        # 收集当前楼层所有走廊的坐标点（用于计算楼层范围和衔接关系）
        all_corridor_points = []
        for corridor in level['corridors']:
            all_corridor_points.extend(corridor['points'])
        if not all_corridor_points:
            continue  

        # 1. 绘制楼层平面边框（次要元素，用细线条）
        xs = [p[0] for p in all_corridor_points]
        ys = [p[1] for p in all_corridor_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        plane_vertices = [
            [min_x, min_y, z], [max_x, min_y, z], 
            [max_x, max_y, z], [min_x, max_y, z], [min_x, min_y, z]
        ]
        x_plane = [p[0] for p in plane_vertices]
        y_plane = [p[1] for p in plane_vertices]
        z_plane = [p[2] for p in plane_vertices]
        ax.plot(x_plane, y_plane, z_plane, color=color, linewidth=1.5, 
                label=f'Level {level_name} (Floor {z})')

        # 2. 绘制走廊（核心导航通道，加粗高亮）
        for corridor_idx, corridor in enumerate(level['corridors']):
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            # 走廊主线
            ax.plot(x, y, z_coords, color=color, linewidth=corridor_linewidth, 
                    alpha=0.8, label=f'Corridor (Level {z})' if corridor_idx == 0 else "")
            # 走廊节点（增强连续性视觉）
            ax.scatter(x, y, z_coords, color=color, s=80, alpha=0.6)

        # 3. 绘制楼梯（突出与走廊的衔接，红色三角标记）
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            # 楼梯位置标记
            ax.scatter(x, y, z, color='#E53935', s=300, marker='^', 
                      label='Stair' if z == 0 else "")
            ax.text(x, y, z+0.15, stair['name'], color='#E53935', 
                    fontweight='bold', fontsize=10)
            # 楼梯-走廊衔接提示（半透明红色圆圈，标记楼梯影响范围）
            ax.scatter(x, y, z, color='#E53935', s=1200, alpha=0.15, marker='o')

        # 4. 绘制教室（突出与走廊的衔接，深蓝色方块标记）
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']

            # 教室标签
            ax.text(x, y, z, classroom['name'], color='black', 
                    fontweight='bold', fontsize=9)
            # 教室位置点
            ax.scatter(x, y, z, color='#3949AB', s=120, marker='s')
            # 教室边界（虚线框）
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color='#3949AB', linestyle='--', linewidth=2)

    # 设置坐标轴（明确单位，提升实用性）
    ax.set_xlabel('X Position (Meters)', fontsize=11)
    ax.set_ylabel('Y Position (Meters)', fontsize=11)
    ax.set_zlabel('Floor Level', fontsize=11)
    ax.set_title('School 3D Map (Classroom→Corridor→Stair→Corridor→Classroom)', 
                 fontsize=14, fontweight='bold')
    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    return fig, ax

# 自定义图数据结构（强化走廊-楼梯关联）
class Graph:
    def __init__(self):
        self.nodes = {}  # key: node_id, value: node_info
        self.stair_corridor_link = {}  # 记录楼梯与关联走廊的映射
        self.STAIR_PROXIMITY_THRESHOLD = 5  # 楼梯-走廊衔接阈值（单位：米）

    def add_node(self, node_id, node_type, name, level, coordinates):
        """添加节点（教室/楼梯/走廊）"""
        self.nodes[node_id] = {
            'type': node_type,       # 节点类型：classroom/stair/corridor
            'name': name,            # 节点名称（如Class101/Stair1/Corridor1）
            'level': level,          # 节点所在楼层（如Level0）
            'coordinates': coordinates,  # 3D坐标 (x,y,z)
            'neighbors': {}          # 邻居节点：{neighbor_id: weight}
        }

    def add_edge(self, node1, node2, weight):
        """添加双向边（权重为欧氏距离）"""
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1]['neighbors'][node2] = weight
            self.nodes[node2]['neighbors'][node1] = weight

    def bind_stair_corridor(self, stair_node_id, corridor_node_ids):
        """绑定楼梯与关联走廊（明确哪些走廊衔接该楼梯）"""
        self.stair_corridor_link[stair_node_id] = corridor_node_ids

    def is_stair_corridor(self, corridor_node_id):
        """判断走廊节点是否为“楼梯衔接走廊”（即与楼梯直接关联）"""
        for stair_node, corridor_nodes in self.stair_corridor_link.items():
            if corridor_node_id in corridor_nodes:
                return True, stair_node  # 返回：是否衔接、关联的楼梯节点
        return False, None

    def get_corridor_stair(self, corridor_node_id):
        """获取走廊节点关联的楼梯节点（若存在）"""
        is_linked, stair_node = self.is_stair_corridor(corridor_node_id)
        return stair_node if is_linked else None

# 计算欧氏距离（3D坐标）
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（严格遵循“教室-走廊-楼梯-走廊-教室”衔接逻辑）
def build_navigation_graph(school_data):
    graph = Graph()

    # 步骤1：添加所有节点（先添加走廊，再添加教室和楼梯，便于后续关联）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 1.1 添加走廊节点（按走廊线段拆分，确保路径连续）
        for corridor_idx, corridor in enumerate(level['corridors']):
            corridor_name = f"Corridor_{corridor_idx+1}_{level_name}"
            for point_idx, point in enumerate(corridor['points']):
                # 走廊节点ID：走廊名_坐标_楼层（确保唯一性）
                node_id = f"{corridor_name}_({point[0]},{point[1]})"
                if node_id not in graph.nodes:
                    graph.add_node(
                        node_id=node_id,
                        node_type='corridor',
                        name=f"{corridor_name}_Point_{point_idx+1}",
                        level=level_name,
                        coordinates=point
                    )

        # 1.2 添加教室节点（仅与走廊关联，禁止直接连楼梯）
        for classroom in level['classrooms']:
            node_id = f"Class_{classroom['name']}@{level_name}"
            if node_id not in graph.nodes:
                graph.add_node(
                    node_id=node_id,
                    node_type='classroom',
                    name=classroom['name'],
                    level=level_name,
                    coordinates=classroom['coordinates']
                )

        # 1.3 添加楼梯节点（仅与走廊关联，禁止直接连教室）
        for stair in level['stairs']:
            node_id = f"Stair_{stair['name']}@{level_name}"
            if node_id not in graph.nodes:
                graph.add_node(
                    node_id=node_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

    # 步骤2：添加边（严格控制衔接关系，禁止跨类型直接连接）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 筛选当前楼层的节点（按类型分类）
        current_level_nodes = [n for n in graph.nodes if graph.nodes[n]['level'] == level_name]
        classrooms = [n for n in current_level_nodes if graph.nodes[n]['type'] == 'classroom']
        stairs = [n for n in current_level_nodes if graph.nodes[n]['type'] == 'stair']
        corridors = [n for n in current_level_nodes if graph.nodes[n]['type'] == 'corridor']

        # 2.1 教室 ↔ 走廊：仅连接最近的2个走廊节点（确保教室只能通过走廊进出）
        for classroom_node in classrooms:
            # 计算教室到所有走廊节点的距离
            dists = [
                (corr_node, euclidean_distance(
                    graph.nodes[classroom_node]['coordinates'],
                    graph.nodes[corr_node]['coordinates']
                )) for corr_node in corridors
            ]
            # 按距离排序，取最近的2个走廊（避免单一连接点故障）
            dists_sorted = sorted(dists, key=lambda x: x[1])[:2]
            for corr_node, dist in dists_sorted:
                # 教室-走廊权重：距离×1.2（略高于走廊内部，优先走走廊）
                graph.add_edge(classroom_node, corr_node, dist * 1.2)

        # 2.2 楼梯 ↔ 走廊：仅连接距离≤阈值的走廊节点（绑定楼梯与衔接走廊）
        stair_corridor_map = {}  # 临时存储楼梯与关联走廊的映射
        for stair_node in stairs:
            dists = [
                (corr_node, euclidean_distance(
                    graph.nodes[stair_node]['coordinates'],
                    graph.nodes[corr_node]['coordinates']
                )) for corr_node in corridors
            ]
            # 筛选距离≤阈值的走廊节点（衔接楼梯的走廊）
            linked_corridors = [
                (corr_node, dist) for corr_node, dist in dists 
                if dist <= graph.STAIR_PROXIMITY_THRESHOLD
            ]
            # 记录楼梯与关联走廊的映射
            stair_corridor_map[stair_node] = [c for c, d in linked_corridors]
            # 添加楼梯-走廊边（权重：距离×0.8，优先选择楼梯附近走廊）
            for corr_node, dist in linked_corridors:
                graph.add_edge(stair_node, corr_node, dist * 0.8)

        # 2.3 走廊 ↔ 走廊：同一线段内连续节点+不同线段邻近节点（确保路径连续）
        # 先按走廊名分组（同一走廊线段的节点）
        corridor_groups = {}
        for corr_node in corridors:
            # 提取走廊名（如Corridor_1_Level0）
            corr_name = '_'.join(corr_node.split('_')[:3])
            if corr_name not in corridor_groups:
                corridor_groups[corr_name] = []
            corridor_groups[corr_name].append(corr_node)
        
        # 2.3.1 同一走廊线段内：连接连续节点（权重最低，优先走直线走廊）
        for corr_name, nodes in corridor_groups.items():
            # 按X坐标排序（确保节点顺序与走廊走向一致）
            nodes_sorted = sorted(nodes, key=lambda x: graph.nodes[x]['coordinates'][0])
            for i in range(len(nodes_sorted) - 1):
                node1 = nodes_sorted[i]
                node2 = nodes_sorted[i+1]
                dist = euclidean_distance(
                    graph.nodes[node1]['coordinates'],
                    graph.nodes[node2]['coordinates']
                )
                # 走廊内部权重：距离×0.5（最低权重，优先选择）
                graph.add_edge(node1, node2, dist * 0.5)
        
        # 2.3.2 不同走廊线段间：连接邻近节点（距离≤3米，处理走廊交叉口）
        for i in range(len(corridors)):
            for j in range(i + 1, len(corridors)):
                node1 = corridors[i]
                node2 = corridors[j]
                # 跳过同一走廊线段的节点（已处理）
                corr1_name = '_'.join(node1.split('_')[:3])
                corr2_name = '_'.join(node2.split('_')[:3])
                if corr1_name == corr2_name:
                    continue
                # 距离≤3米的不同走廊线段，添加连接
                dist = euclidean_distance(
                    graph.nodes[node1]['coordinates'],
                    graph.nodes[node2]['coordinates']
                )
                if dist <= 3:
                    graph.add_edge(node1, node2, dist * 0.7)

        # 2.4 绑定楼梯与关联走廊（更新graph的stair_corridor_link）
        for stair_node, linked_corrs in stair_corridor_map.items():
            graph.bind_stair_corridor(stair_node, linked_corrs)

    # 步骤3：添加跨楼层边（仅允许楼梯↔楼梯连接，模拟上下楼）
    for connection in school_data['buildingA']['connections']:
        from_stair_name, from_level = connection['from']
        to_stair_name, to_level = connection['to']
        
        # 构建跨楼层楼梯节点ID
        from_stair_node = f"Stair_{from_stair_name}@{from_level}"
        to_stair_node = f"Stair_{to_stair_name}@{to_level}"
        
        if from_stair_node in graph.nodes and to_stair_node in graph.nodes:
            # 跨楼层权重：固定1.0（模拟上下楼成本，不随距离变化）
            graph.add_edge(from_stair_node, to_stair_node, 1.0)

    return graph

# 改进的Dijkstra算法（强制全路径遵循“教室→走廊→楼梯→走廊→教室”）
def dijkstra(graph, start_node, end_node):
    # 初始化：距离（起点为0，其他为无穷大）、前驱节点
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    unvisited_nodes = set(graph.nodes.keys())

    # 起点/终点类型（用于路径约束）
    start_type = graph.nodes[start_node]['type']
    end_type = graph.nodes[end_node]['type']
    # 终点所在楼层（用于跨楼层路径优化）
    end_level = graph.nodes[end_node]['level']

    while unvisited_nodes:
        # 选择当前距离最小的节点
        current_node = min(unvisited_nodes, key=lambda x: distances[x])
        unvisited_nodes.remove(current_node)

        if distances[current_node] == float('inf'):
            break  # 无法到达的节点

        current_type = graph.nodes[current_node]['type']
        current_level = graph.nodes[current_node]['level']

        # 遍历邻居节点，更新距离
        for neighbor, weight in graph.nodes[current_node]['neighbors'].items():
            # 核心约束1：禁止非走廊节点之间的直接连接
            neighbor_type = graph.nodes[neighbor]['type']
            if current_type != 'corridor' and neighbor_type != 'corridor':
                continue  # 禁止教室↔楼梯、教室↔教室、楼梯↔楼梯（同楼层）
            
            # 核心约束2：跨楼层移动必须通过楼梯
            neighbor_level = graph.nodes[neighbor]['level']
            if current_level != neighbor_level:
                if current_type != 'stair' or neighbor_type != 'stair':
                    continue  # 跨楼层只能通过楼梯节点
            
            # 权重优化：根据节点类型和目标调整权重
            weight_factor = 1.0
            
            # 1. 若需要跨楼层，优先引导到楼梯
            if current_level != end_level and current_type == 'corridor':
                # 走廊越靠近楼梯，权重越低（优先选择）
                is_stair_corr, _ = graph.is_stair_corridor(current_node)
                weight_factor = 0.7 if is_stair_corr else 1.2
            
            # 2. 若已在目标楼层，优先引导到终点附近走廊
            if current_level == end_level and neighbor_type == 'corridor':
                # 计算走廊到终点的距离（距离越近权重越低）
                end_dist = euclidean_distance(
                    graph.nodes[neighbor]['coordinates'],
                    graph.nodes[end_node]['coordinates']
                )
                weight_factor = 0.5 + (min(end_dist, 10) / 10) * 0.5  # 0.5-1.0
            
            # 计算新距离
            new_distance = distances[current_node] + weight * weight_factor
            
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node

    return distances, previous_nodes

# 生成路径（从后往前回溯）
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path

# 完整路径验证：确保严格遵循“教室→走廊→楼梯→走廊→教室”
def validate_full_path(graph, path):
    if len(path) < 2:
        return True  # 单点路径无需验证
    
    # 1. 起点规则：教室必须先到走廊
    start_node = path[0]
    if graph.nodes[start_node]['type'] == 'classroom' and len(path) >= 2:
        first_step = path[1]
        if graph.nodes[first_step]['type'] != 'corridor':
            return False, "起点教室必须先进入走廊"
    
    # 2. 终点规则：走廊必须最后到教室
    end_node = path[-1]
    if graph.nodes[end_node]['type'] == 'classroom' and len(path) >= 2:
        last_step = path[-2]
        if graph.nodes[last_step]['type'] != 'corridor':
            return False, "终点教室必须从走廊进入"
    
    # 3. 楼梯规则：楼梯必须前后都是走廊
    for i in range(1, len(path)-1):
        current_node = path[i]
        if graph.nodes[current_node]['type'] == 'stair':
            prev_node = path[i-1]
            next_node = path[i+1]
            if (graph.nodes[prev_node]['type'] != 'corridor' or 
                graph.nodes[next_node]['type'] != 'corridor'):
                return False, "楼梯必须前后都连接走廊"
    
    # 4. 跨楼层规则：跨楼层必须经过楼梯
    for i in range(len(path)-1):
        current_level = graph.nodes[path[i]]['level']
        next_level = graph.nodes[path[i+1]]['level']
        if current_level != next_level:
            current_type = graph.nodes[path[i]]['type']
            next_type = graph.nodes[path[i+1]]['type']
            if current_type != 'stair' or next_type != 'stair':
                return False, "跨楼层移动必须通过楼梯"
    
    return True, "路径验证通过"

# 强制生成符合规则的路径
def force_valid_path(graph, start_node, end_node):
    # 第一次尝试
    dists, prevs = dijkstra(graph, start_node, end_node)
    path = construct_path(prevs, end_node)
    if path:
        is_valid, msg = validate_full_path(graph, path)
        if is_valid:
            return path, dists[end_node], msg
    
    # 若第一次失败，分步构建路径
    start_level = graph.nodes[start_node]['level']
    end_level = graph.nodes[end_node]['level']
    
    # 步骤1：起点教室 → 起点楼层走廊
    start_corridors = [n for n in graph.nodes[start_node]['neighbors'] 
                      if graph.nodes[n]['type'] == 'corridor']
    if not start_corridors:
        return None, 0, "起点教室无法连接到任何走廊"
    
    # 步骤2：起点楼层走廊 → 起点楼层楼梯
    start_stairs = [n for n in graph.nodes if 
                   graph.nodes[n]['type'] == 'stair' and 
                   graph.nodes[n]['level'] == start_level]
    if not start_stairs:
        return None, 0, "起点楼层没有楼梯"
    
    # 步骤3：跨楼层楼梯（若需要）
    mid_stairs = [start_stairs[0]]
    if start_level != end_level:
        # 找到连接两个楼层的楼梯对
        connected_stairs = []
        for s1 in start_stairs:
            for neighbor in graph.nodes[s1]['neighbors']:
                if (graph.nodes[neighbor]['type'] == 'stair' and 
                    graph.nodes[neighbor]['level'] == end_level):
                    connected_stairs.append((s1, neighbor))
        if not connected_stairs:
            return None, 0, "无法找到连接两个楼层的楼梯"
        mid_stairs = [s1 for s1, s2 in connected_stairs] + [s2 for s1, s2 in connected_stairs]
    
    # 步骤4：终点楼层楼梯 → 终点楼层走廊
    end_stairs = [n for n in graph.nodes if 
                 graph.nodes[n]['type'] == 'stair' and 
                 graph.nodes[n]['level'] == end_level]
    end_corridors = []
    for stair in end_stairs:
        end_corridors.extend(graph.stair_corridor_link.get(stair, []))
    if not end_corridors:
        return None, 0, "终点楼层楼梯无法连接到走廊"
    
    # 步骤5：终点楼层走廊 → 终点教室
    valid_end_corridors = [c for c in end_corridors if end_node in graph.nodes[c]['neighbors']]
    if not valid_end_corridors:
        return None, 0, "终点教室无法连接到走廊"
    
    # 计算最优组合路径
    min_total_dist = float('inf')
    best_path = None
    
    for start_corr in start_corridors:
        for start_stair in start_stairs:
            # 起点→起点走廊→起点楼梯
            d1, p1 = dijkstra(graph, start_node, start_corr)
            p1_path = construct_path(p1, start_corr)
            if not p1_path: continue
            
            d2, p2 = dijkstra(graph, start_corr, start_stair)
            p2_path = construct_path(p2, start_stair)[1:]  # 去重
            if not p2_path: continue
            
            # 处理跨楼层
            if start_level != end_level:
                for _, end_stair in connected_stairs:
                    d3, p3 = dijkstra(graph, start_stair, end_stair)
                    p3_path = construct_path(p3, end_stair)[1:]  # 去重
                    if not p3_path: continue
                    
                    # 终点楼梯→终点走廊→终点
                    for end_corr in valid_end_corridors:
                        d4, p4 = dijkstra(graph, end_stair, end_corr)
                        p4_path = construct_path(p4, end_corr)[1:]  # 去重
                        if not p4_path: continue
                        
                        d5 = graph.nodes[end_corr]['neighbors'][end_node]
                        full_path = p1_path + p2_path + p3_path + p4_path + [end_node]
                        total_dist = d1[start_corr] + d2[start_stair] + d3[end_stair] + d4[end_corr] + d5
                        
                        if total_dist < min_total_dist:
                            min_total_dist = total_dist
                            best_path = full_path
            else:
                # 同楼层：起点楼梯→终点走廊→终点
                for end_corr in valid_end_corridors:
                    d4, p4 = dijkstra(graph, start_stair, end_corr)
                    p4_path = construct_path(p4, end_corr)[1:]  # 去重
                    if not p4_path: continue
                    
                    d5 = graph.nodes[end_corr]['neighbors'][end_node]
                    full_path = p1_path + p2_path + p4_path + [end_node]
                    total_dist = d1[start_corr] + d2[start_stair] + d4[end_corr] + d5
                    
                    if total_dist < min_total_dist:
                        min_total_dist = total_dist
                        best_path = full_path
    
    if best_path:
        return best_path, min_total_dist, "强制生成符合规则的路径"
    else:
        return None, 0, "无法生成符合规则的路径"

# 导航函数（集成路径验证和强制调整）
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    # 构建起点和终点节点ID
    start_node = f"Class_{start_classroom}@{start_level}"
    end_node = f"Class_{end_classroom}@{end_level}"

    # 基础校验
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "❌ 无效的教室或楼层"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 尝试生成并验证路径
    path, total_dist, msg = force_valid_path(graph, start_node, end_node)
    
    if path:
        is_valid, valid_msg = validate_full_path(graph, path)
        if is_valid:
            return path, f"✅ 路径规划成功！总距离：{total_dist:.2f} 米\n{valid_msg}"
        else:
            return None, f"❌ 路径验证失败：{valid_msg}"
    else:
        return None, f"❌ 路径生成失败：{msg}"

# 在3D图上绘制路径（突出显示各阶段节点）
def plot_path(ax, graph, path):
    x_coords = []
    y_coords = []
    z_coords = []
    node_types = []
    path_segments = []  # 记录路径阶段：start→corridor→stair→corridor→end

    for i, node_id in enumerate(path):
        node = graph.nodes[node_id]
        coords = node['coordinates']
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        z_coords.append(coords[2])
        node_types.append(node['type'])
        
        # 标记路径阶段
        if i == 0:
            path_segments.append("start")
        elif i == len(path) - 1:
            path_segments.append("end")
        elif node['type'] == 'stair':
            path_segments.append("stair")
        else:  # corridor
            # 判断是楼梯前还是楼梯后的走廊
            has_stair_before = any(graph.nodes[path[j]]['type'] == 'stair' for j in range(i))
            path_segments.append("corridor_after_stair" if has_stair_before else "corridor_before_stair")

    # 绘制路径主线（分阶段使用不同样式）
    for i in range(len(x_coords) - 1):
        x1, y1, z1 = x_coords[i], y_coords[i], z_coords[i]
        x2, y2, z2 = x_coords[i+1], y_coords[i+1], z_coords[i+1]
        seg_type = path_segments[i]
        
        # 根据阶段设置线条样式
        if seg_type == "start":
            # 起点→走廊：虚线
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='#4CAF50', linewidth=2, linestyle='--')
        elif seg_type == "corridor_before_stair":
            # 楼梯前走廊：实线
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='#2196F3', linewidth=3, linestyle='-')
        elif seg_type == "stair":
            # 楼梯附近：加粗红线
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='#F44336', linewidth=4, linestyle='-')
        elif seg_type == "corridor_after_stair":
            # 楼梯后走廊：实线
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='#FF9800', linewidth=3, linestyle='-')
        elif seg_type == "end":
            # 走廊→终点：虚线
            ax.plot([x1, x2], [y1, y2], [z1, z2], color='#9C27B0', linewidth=2, linestyle='--')

    # 标记关键节点
    for i, (x, y, z, node_type, seg) in enumerate(zip(x_coords, y_coords, z_coords, node_types, path_segments)):
        if seg == "start":
            ax.scatter(x, y, z, color='#4CAF50', s=300, marker='*', label='Start Classroom')
        elif seg == "end":
            ax.scatter(x, y, z, color='#9C27B0', s=300, marker='*', label='End Classroom')
        elif node_type == 'stair':
            ax.scatter(x, y, z, color='#F44336', s=250, marker='^', label='Stair')
        elif seg == "corridor_before_stair":
            ax.scatter(x, y, z, color='#2196F3', s=120, marker='o', label='Corridor (to Stair)')
        elif seg == "corridor_after_stair":
            ax.scatter(x, y, z, color='#FF9800', s=120, marker='o', label='Corridor (from Stair)')

    # 更新图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

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
    st.subheader("严格遵循：起点教室→走廊→楼梯→走廊→终点教室")

    try:
        # 加载校园数据
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_navigation_graph(school_data)
        levels, classrooms_by_level = get_classroom_info(school_data)
        st.success("✅ 校园数据加载成功！")
    except FileNotFoundError:
        st.error("❌ 错误：未找到'school_data_detailed.json'文件，请检查文件路径。")
        return
    except Exception as e:
        st.error(f"❌ 数据加载失败：{str(e)}")
        return

    # 界面布局：左侧选择面板，右侧地图展示
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

        nav_button = st.button("🔍 查找合规路径", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ 3D校园地图（路径阶段高亮）")
        
        # 初始化地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        # 路径规划逻辑
        if nav_button:
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            if path:
                st.success(message)
                st.markdown("#### 🛤️ 路径详情（阶段划分）")
                for i, node in enumerate(path, 1):
                    node_info = nav_graph.nodes[node]
                    if node_info['type'] == 'classroom':
                        st.write(f"{i}. {node_info['name']}（{node_info['type']}，楼层：{node_info['level']}）")
                    elif node_info['type'] == 'stair':
                        st.write(f"{i}. {node_info['name']}（{node_info['type']}，楼层：{node_info['level']}）→ 上下楼")
                    else:  # corridor
                        # 判断是楼梯前还是楼梯后的走廊
                        has_stair_before = any(nav_graph.nodes[path[j]]['type'] == 'stair' for j in range(i))
                        stage = "楼梯后走廊（前往终点）" if has_stair_before else "楼梯前走廊（前往楼梯）"
                        st.write(f"{i}. {node_info['name']}（{stage}）")
                
                # 绘制带路径的地图
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, nav_graph, path)
                st.session_state['fig'] = fig
            else:
                st.error(message)
        
        # 显示地图
        st.pyplot(st.session_state['fig'])

if __name__ == "__main__":
    main()
