import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. 基础配置 --------------------------
plt.switch_backend('Agg')

# 定义标准颜色（使用RGB值确保兼容性）
COLORS = {
    'start_classroom': (0.0, 0.5, 0.0),    # 绿色
    'end_classroom': (0.5, 0.0, 0.5),      # 紫色
    'stair': (0.8, 0.0, 0.0),              # 红色
    'corridor_after_start': (0.0, 1.0, 1.0),# 青色
    'corridor_before_end': (1.0, 0.5, 0.0),# 橙色
    'corridor_middle': (0.0, 0.0, 1.0),    # 蓝色
    'stair_area': (0.8, 0.0, 0.0)          # 红色
}

# -------------------------- 2. 核心功能实现 --------------------------
# 读取JSON数据
def load_school_data_detailed(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载数据文件出错: {str(e)}")
        return None

# 绘制3D地图
def plot_3d_map(school_data):
    if not school_data:
        return None, None
        
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同楼层使用不同颜色
    floor_colors = {0: 'blue', 2: 'green', 5: 'orange'}  

    # 处理每个楼层
    for level in school_data.get('buildingA', {}).get('levels', []):
        z = level.get('z', 0)
        color = floor_colors.get(z, 'gray')

        # 收集当前楼层所有走廊的坐标点
        all_corridor_points = []
        for corridor in level.get('corridors', []):
            all_corridor_points.extend(corridor.get('points', []))
        if not all_corridor_points:
            continue  

        # 计算平面的X/Y轴范围
        xs = [p[0] for p in all_corridor_points]
        ys = [p[1] for p in all_corridor_points]
        min_x, max_x = min(xs), max(xs) if xs else 0, 10
        min_y, max_y = min(ys), max(ys) if ys else 0, 10

        # 构造平面的4个顶点
        plane_vertices = [
            [min_x, min_y, z], [max_x, min_y, z], 
            [max_x, max_y, z], [min_x, max_y, z], [min_x, min_y, z]
        ]
        x_plane = [p[0] for p in plane_vertices]
        y_plane = [p[1] for p in plane_vertices]
        z_plane = [p[2] for p in plane_vertices]

        # 绘制楼层平面边框
        ax.plot(x_plane, y_plane, z_plane, color=color, linewidth=2, label=level.get('name', f'Floor {z}'))

        # 绘制走廊
        for corridor in level.get('corridors', []):
            points = corridor.get('points', [])
            if not points:
                continue
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            ax.plot(x, y, z_coords, color=color, linewidth=5, alpha=0.8)

        # 绘制楼梯（突出显示，方便识别楼梯附近走廊）
        for stair in level.get('stairs', []):
            coords = stair.get('coordinates', [0, 0, 0])
            x, y = coords[0], coords[1]
            ax.scatter([x], [y], [z], c=[COLORS['stair']], s=200, marker='^', 
                      label='Stairs' if z == 0 else "")
            ax.text(x, y, z+0.1, stair.get('name', 'Stair'), color='red', fontweight='bold')
            
            # 标记楼梯附近的走廊区域（增加半透明圆圈）
            ax.scatter([x], [y], [z], c=[COLORS['stair_area']], s=800, alpha=0.2, marker='o')

        # 绘制教室
        for classroom in level.get('classrooms', []):
            coords = classroom.get('coordinates', [0, 0, 0])
            x, y = coords[0], coords[1]
            width, depth = classroom.get('size', [1, 1])

            # 教室标签
            ax.text(x, y, z, classroom.get('name', 'Classroom'), color='black', fontweight='bold')
            # 教室位置点
            ax.scatter([x], [y], [z], c=[color], s=50)
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

    def has_node(self, node_id):
        """检查节点是否存在"""
        return node_id in self.nodes

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    try:
        return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))
    except:
        return float('inf')

# 构建导航图（优化走廊和楼梯路径）
def build_navigation_graph(school_data):
    if not school_data:
        return None
        
    graph = Graph()

    # 步骤1：添加所有节点（教室、楼梯、走廊）
    for level in school_data.get('buildingA', {}).get('levels', []):
        level_name = level.get('name', '')
        z = level.get('z', 0)
        
        # 收集当前楼层楼梯坐标（用于计算走廊与楼梯的距离）
        stair_coords = [stair.get('coordinates', [0,0,0]) for stair in level.get('stairs', [])]

        # 1.1 添加教室节点
        for classroom in level.get('classrooms', []):
            node_id = f"{classroom.get('name', '')}@{level_name}"
            if not graph.has_node(node_id):
                graph.add_node(
                    node_id=node_id,
                    node_type='classroom',
                    name=classroom.get('name', ''),
                    level=level_name,
                    coordinates=classroom.get('coordinates', [0,0,0])
                )

        # 1.2 添加楼梯节点
        for stair in level.get('stairs', []):
            node_id = f"{stair.get('name', '')}@{level_name}"
            if not graph.has_node(node_id):
                graph.add_node(
                    node_id=node_id,
                    node_type='stair',
                    name=stair.get('name', ''),
                    level=level_name,
                    coordinates=stair.get('coordinates', [0,0,0])
                )

        # 1.3 添加走廊节点（记录与最近楼梯的距离）
        for corridor_idx, corridor in enumerate(level.get('corridors', [])):
            for point_idx, point in enumerate(corridor.get('points', [])):
                node_id = f"corridor_{point[0]}_{point[1]}_{z}"
                if not graph.has_node(node_id):
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
    for level in school_data.get('buildingA', {}).get('levels', []):
        level_name = level.get('name', '')
        z = level.get('z', 0)

        # 2.1 教室 ↔ 走廊：仅连接走廊，删除与楼梯的任何直接连接
        for classroom in level.get('classrooms', []):
            classroom_node_id = f"{classroom.get('name', '')}@{level_name}"
            if not graph.has_node(classroom_node_id):
                continue
                
            classroom_coords = classroom.get('coordinates', [0,0,0])
            
            # 仅筛选当前楼层的【走廊节点】（排除楼梯节点）
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor'  # 只选走廊
                and graph.nodes[node_id]['level'] == level_name
            ]
            
            # 按距离排序，连接最近的2个走廊（确保教室只能通过走廊出行）
            corridor_distances = [
                (node_id, euclidean_distance(classroom_coords, graph.nodes[node_id]['coordinates']))
                for node_id in corridor_nodes
            ]
            corridor_distances.sort(key=lambda x: x[1])
            
            # 仅连接前2个最近的走廊（权重折扣，优先走最近走廊）
            for i, (node_id, distance) in enumerate(corridor_distances[:2]):  # 只连前2个走廊
                weight = distance * 0.5  # 走廊连接权重降低，优先选择
                graph.add_edge(classroom_node_id, node_id, weight)

        # 2.2 楼梯 ↔ 走廊：仅连接走廊，切断楼梯与教室的直接连接
        for stair in level.get('stairs', []):
            stair_node_id = f"{stair.get('name', '')}@{level_name}"
            if not graph.has_node(stair_node_id):
                continue
                
            stair_coords = stair.get('coordinates', [0,0,0])
            
            # 仅筛选当前楼层的【走廊节点】（排除教室节点）
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor'  # 只选走廊
                and graph.nodes[node_id]['level'] == level_name
            ]
            
            # 楼梯优先连接5单位内的走廊（权重更低，引导路径走楼梯附近走廊）
            for node_id in corridor_nodes:
                distance = euclidean_distance(stair_coords, graph.nodes[node_id]['coordinates'])
                weight = distance * 0.3 if distance < 5 else distance  # 楼梯附近走廊权重低
                graph.add_edge(stair_node_id, node_id, weight)

        # 2.3 走廊 ↔ 走廊：优化权重，使路径更倾向于通向楼梯附近的走廊
        corridor_nodes = [
            node_id for node_id in graph.nodes 
            if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
        ]
        
        for i in range(len(corridor_nodes)):
            for j in range(i + 1, len(corridor_nodes)):
                node1 = corridor_nodes[i]
                node2 = corridor_nodes[j]
                
                # 基础距离权重
                distance = euclidean_distance(
                    graph.nodes[node1]['coordinates'], 
                    graph.nodes[node2]['coordinates']
                )
                
                # 楼梯 proximity 因子：如果走廊靠近楼梯，给予权重优势
                # 两个走廊中至少有一个靠近楼梯，则降低权重
                stair_factor = 0.7 if (graph.stair_proximity.get(node1, float('inf')) < 5 or 
                                      graph.stair_proximity.get(node2, float('inf')) < 5) else 1.0
                
                # 最终权重 = 距离 × 楼梯因子
                weight = distance * stair_factor
                graph.add_edge(node1, node2, weight)

    # 2.4 楼梯 ↔ 楼梯：跨楼层连接
    for connection in school_data.get('buildingA', {}).get('connections', []):
        from_info = connection.get('from', ['', ''])
        to_info = connection.get('to', ['', ''])
        from_stair_name, from_level = from_info[0], from_info[1]
        to_stair_name, to_level = to_info[0], to_info[1]
        
        from_stair_node = f"{from_stair_name}@{from_level}"
        to_stair_node = f"{to_stair_name}@{to_level}"
        
        if graph.has_node(from_stair_node) and graph.has_node(to_stair_node):
            graph.add_edge(from_stair_node, to_stair_node, 1.0)

    return graph

# 改进的Dijkstra算法，强制路径遵循教室→走廊→楼梯→走廊→教室的流程
def dijkstra(graph, start_node, end_node):
    if not graph or not graph.has_node(start_node) or not graph.has_node(end_node):
        return {}, {}
        
    # 初始化距离：起点为0，其他为无穷大
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    unvisited_nodes = set(graph.nodes.keys())

    # 关键变量：记录路径是否已经过楼梯（跨楼层必须经过楼梯）
    has_passed_stair = {node: False for node in graph.nodes}
    # 起点和终点楼层（判断是否需要跨楼层）
    start_level = graph.nodes[start_node]['level']
    end_level = graph.nodes[end_node]['level']
    need_cross_floor = (start_level != end_level)

    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=lambda x: distances[x])
        unvisited_nodes.remove(current_node)

        if distances[current_node] == float('inf'):
            break

        # 更新当前节点是否经过楼梯（若当前节点是楼梯，标记为已经过）
        current_has_stair = has_passed_stair[current_node] or (graph.nodes[current_node]['type'] == 'stair')

        for neighbor, weight in graph.nodes[current_node]['neighbors'].items():
            # 核心约束：跨楼层时，未经过楼梯的路径必须惩罚
            extra_factor = 1.0
            neighbor_level = graph.nodes[neighbor]['level']
            
            # 情况1：跨楼层且未经过楼梯 → 大幅增加权重（禁止跳步）
            if need_cross_floor and not current_has_stair:
                # 若当前节点和邻居不在同一楼层（且未走楼梯），权重×100（几乎不可选）
                if neighbor_level != start_level:
                    extra_factor = 100.0  # 惩罚跳层路径
                # 若在同一楼层但未靠近楼梯，权重×2（引导走向楼梯）
                elif graph.nodes[current_node]['type'] == 'corridor':
                    stair_dist = graph.stair_proximity.get(current_node, float('inf'))
                    extra_factor = 2.0 if stair_dist >= 5 else 1.0  # 远离楼梯的走廊惩罚
                
            # 情况2：已经过楼梯（跨楼层后）→ 引导走走廊（靠近终点教室）
            elif need_cross_floor and current_has_stair:
                if graph.nodes[neighbor]['type'] == 'corridor':
                    # 跨楼层后优先走走廊（权重×0.8），避免楼梯直接连教室（已切断）
                    extra_factor = 0.8

            # 计算新距离（应用权重约束）
            new_distance = distances[current_node] + weight * extra_factor

            # 更新距离和前置节点
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_nodes[neighbor] = current_node
                has_passed_stair[neighbor] = current_has_stair  # 传递楼梯经过状态

    return distances, previous_nodes

# 生成最短路径
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None and current_node in previous_nodes:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path if path else None

# 导航函数
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    if not graph:
        return None, "❌ 导航图未初始化"
        
    start_node = f"{start_classroom}@{start_level}"
    end_node = f"{end_classroom}@{end_level}"

    if not graph.has_node(start_node) or not graph.has_node(end_node):
        return None, "❌ 无效的教室或楼层"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 使用改进的Dijkstra算法
    distances, previous_nodes = dijkstra(graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node)
    
    if not path:
        return None, "❌ 无法找到路径"

    # 校验路径是否符合标准流程：教室→走廊→楼梯→走廊→教室
    def is_valid_path(path):
        if len(path) < 2:
            return False
        
        # 提取路径中各节点的类型
        path_types = []
        for node in path:
            if graph.has_node(node):
                path_types.append(graph.nodes[node]['type'])
            else:
                return False
                
        start_type = path_types[0]
        end_type = path_types[-1]
        has_corridor_before_stair = False  # 楼梯前是否有走廊
        has_corridor_after_stair = False   # 楼梯后是否有走廊
        stair_count = 0  # 跨楼层至少1个楼梯（同楼层无需楼梯）

        # 遍历路径检查节点顺序
        for i in range(len(path_types)):
            if path_types[i] == 'stair':
                stair_count += 1
                # 楼梯前必须有走廊（i≥1且前一个是走廊）
                if i >= 1 and path_types[i-1] == 'corridor':
                    has_corridor_before_stair = True
                # 楼梯后必须有走廊（i<len-1且后一个是走廊）
                if i < len(path_types)-1 and path_types[i+1] == 'corridor':
                    has_corridor_after_stair = True
        
        # 同楼层场景（无需楼梯）：流程为“教室→走廊→教室”
        if start_level == end_level:
            return (start_type == 'classroom' and end_type == 'classroom' 
                    and 'corridor' in path_types 
                    and stair_count == 0)
        
        # 跨楼层场景：流程为“教室→走廊→楼梯→走廊→教室”
        else:
            return (start_type == 'classroom' and end_type == 'classroom'
                    and has_corridor_before_stair and has_corridor_after_stair
                    and stair_count >= 1)  # 至少1个楼梯（跨楼层）

    # 若路径不合法，尝试重新规划
    if not is_valid_path(path):
        # 创建临时图，简化连接以强制标准路径
        temp_graph = Graph()
        # 复制所有节点
        for node_id, node_info in graph.nodes.items():
            temp_graph.add_node(
                node_id=node_id,
                node_type=node_info['type'],
                name=node_info['name'],
                level=node_info['level'],
                coordinates=node_info['coordinates'],
                stair_distance=graph.stair_proximity.get(node_id)
            )
        
        # 重新添加边，严格限制连接规则
        for node1 in graph.nodes:
            for node2, weight in graph.nodes[node1]['neighbors'].items():
                type1 = graph.nodes[node1]['type']
                type2 = graph.nodes[node2]['type']
                
                # 只允许：教室-走廊、走廊-走廊、走廊-楼梯、楼梯-楼梯 之间的连接
                valid_connection = (
                    (type1 == 'classroom' and type2 == 'corridor') or
                    (type1 == 'corridor' and type2 == 'classroom') or
                    (type1 == 'corridor' and type2 == 'corridor') or
                    (type1 == 'corridor' and type2 == 'stair') or
                    (type1 == 'stair' and type2 == 'corridor') or
                    (type1 == 'stair' and type2 == 'stair')
                )
                
                if valid_connection:
                    temp_graph.add_edge(node1, node2, weight)
        
        # 重新规划路径
        distances, previous_nodes = dijkstra(temp_graph, start_node, end_node)
        path = construct_path(previous_nodes, end_node)
        
        # 二次校验
        if not path or not is_valid_path(path):
            return None, "❌ 路径不符合标准流程（教室→走廊→楼梯→走廊→教室），请检查地图数据"

    total_distance = distances.get(end_node, float('inf'))
    return path, f"✅ 最优路径规划成功！总距离：{total_distance:.2f} 单位"

# 在3D图上绘制路径（突出显示标准流程节点）
def plot_path(ax, graph, path):
    # 确保路径不为空且图形对象有效
    if not path or not ax or not graph:
        return
    
    x_coords = []
    y_coords = []
    z_coords = []
    path_types = []

    for node_id in path:
        if not graph.has_node(node_id):
            continue  # 跳过无效节点
            
        node = graph.nodes[node_id]
        coords = node['coordinates']
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        z_coords.append(coords[2])
        path_types.append(node['type'])

    # 绘制路径主线（红色实线，突出显示）
    ax.plot(x_coords, y_coords, z_coords, color='red', linewidth=4, linestyle='-', marker='o', markersize=8)

    # 标记标准流程节点（按顺序高亮）
    for i, (x, y, z_val, node_id, node_type) in enumerate(zip(x_coords, y_coords, z_coords, path, path_types)):
        try:
            # 确保节点存在
            if not graph.has_node(node_id):
                continue
                
            if i == 0:  # 起点教室
                ax.scatter([x], [y], [z_val], c=[COLORS['start_classroom']], 
                          s=500, marker='*', label='Start (Classroom)')
                ax.text(x, y, z_val+0.2, '起点教室', color='green', fontsize=10, fontweight='bold')
            elif i == len(path) - 1:  # 终点教室
                ax.scatter([x], [y], [z_val], c=[COLORS['end_classroom']], 
                          s=500, marker='*', label='End (Classroom)')
                ax.text(x, y, z_val+0.2, '终点教室', color='purple', fontsize=10, fontweight='bold')
            elif node_type == 'stair':  # 楼梯（跨楼层关键节点）
                ax.scatter([x], [y], [z_val], c=[COLORS['stair']], 
                          s=400, marker='^', label='Stair (Cross Floor)')
                ax.text(x, y, z_val+0.2, '楼梯', color='red', fontsize=10, fontweight='bold')
            elif node_type == 'corridor':  # 走廊（区分“起点后”和“终点前”）
                if i == 1:  # 起点教室后的第一个走廊（第一步）
                    ax.scatter([x], [y], [z_val], c=[COLORS['corridor_after_start']], 
                              s=300, marker='s', label='Corridor (After Start)')
                    ax.text(x, y, z_val+0.2, '起点后走廊', color='cyan', fontsize=9, fontweight='bold')
                elif i == len(path) - 2:  # 终点教室前的最后一个走廊（倒数第二步）
                    ax.scatter([x], [y], [z_val], c=[COLORS['corridor_before_end']], 
                              s=300, marker='s', label='Corridor (Before End)')
                    ax.text(x, y, z_val+0.2, '终点前走廊', color='orange', fontsize=9, fontweight='bold')
                else:  # 中间走廊
                    ax.scatter([x], [y], [z_val], c=[COLORS['corridor_middle']], 
                              s=200, marker='o', label='Corridor (Middle)')
        except Exception as e:
            st.warning(f"绘制节点 {node_id} 时出错: {str(e)}")
            continue

    # 调整图例（避免重复）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)

# 获取所有楼层和教室信息
def get_classroom_info(school_data):
    levels = []
    classrooms_by_level = {}
    
    if not school_data:
        return levels, classrooms_by_level
        
    for level in school_data.get('buildingA', {}).get('levels', []):
        level_name = level.get('name', '')
        levels.append(level_name)
        classrooms = [classroom.get('name', '') for classroom in level.get('classrooms', [])]
        classrooms_by_level[level_name] = classrooms
        
    return levels, classrooms_by_level

# -------------------------- 3. Streamlit界面逻辑 --------------------------
def main():
    st.title("🏫 校园导航系统")
    st.subheader("3D地图与标准化路径规划（教室→走廊→楼梯→走廊→教室）")

    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_navigation_graph(school_data) if school_data else None
        levels, classrooms_by_level = get_classroom_info(school_data)
        
        if school_data and nav_graph:
            st.success("✅ 校园数据加载成功！")
        else:
            st.warning("⚠️ 无法加载完整的校园数据")
    except Exception as e:
        st.error(f"❌ 初始化错误：{str(e)}")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📍 选择位置")
        
        st.markdown("#### 起点")
        start_level = st.selectbox("楼层", levels, key="start_level") if levels else None
        start_classrooms = classrooms_by_level.get(start_level, []) if start_level else []
        start_classroom = st.selectbox("教室", start_classrooms, key="start_classroom") if start_classrooms else None

        st.markdown("#### 终点")
        end_level = st.selectbox("楼层", levels, key="end_level") if levels else None
        end_classrooms = classrooms_by_level.get(end_level, []) if end_level else []
        end_classroom = st.selectbox("教室", end_classrooms, key="end_classroom") if end_classrooms else None

        nav_button = st.button("🔍 查找最优路径", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ 3D校园地图")
        
        # 初始化图形
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
            st.session_state['ax'] = ax
        
        if nav_button and start_classroom and start_level and end_classroom and end_level and nav_graph:
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            if path:
                st.success(message)
                st.markdown("#### 🛤️ 路径详情")
                for i, node in enumerate(path, 1):
                    if node.startswith("corridor_"):
                        # 识别特殊走廊节点
                        if i == 2:  # 起点后的第一个走廊
                            st.write(f"{i}. 起点后走廊")
                        elif i == len(path) - 2:  # 终点前的最后一个走廊
                            st.write(f"{i}. 终点前走廊")
                        else:
                            st.write(f"{i}. 中间走廊")
                    else:
                        try:
                            room, floor = node.split('@')
                            if "stair" in room.lower():
                                st.write(f"{i}. {room}（楼层：{floor}）")
                            else:
                                st.write(f"{i}. {room}（楼层：{floor}）")
                        except:
                            st.write(f"{i}. {node}")
                
                # 重新绘制地图和路径
                fig, ax = plot_3d_map(school_data)
                if fig and ax:
                    plot_path(ax, nav_graph, path)
                    st.session_state['fig'] = fig
                    st.session_state['ax'] = ax
            else:
                st.error(message)
        
        # 显示图形
        try:
            if 'fig' in st.session_state and st.session_state['fig']:
                st.pyplot(st.session_state['fig'])
            else:
                st.info("🗺️ 请等待地图加载...")
        except Exception as e:
            st.error(f"绘制地图时出错：{str(e)}")

if __name__ == "__main__":
    main()
    
