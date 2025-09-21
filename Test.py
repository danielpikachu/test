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

        # 绘制走廊（突出显示，明确走廊区域）
        for corridor in level['corridors']:
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            ax.plot(x, y, z_coords, color=color, linewidth=6, alpha=0.9, label='Corridor' if z == 0 else "")

        # 绘制楼梯（突出显示，方便识别楼梯与走廊的连接）
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=250, marker='^', label='Stairs' if z == 0 else "")
            ax.text(x, y, z+0.1, stair['name'], color='red', fontweight='bold', fontsize=10)
            
            # 标记楼梯附近的走廊区域（半透明圆圈，明确“楼梯临近走廊”范围）
            ax.scatter(x, y, z, color='red', s=1000, alpha=0.2, marker='o', label='Stair Proximity' if z == 0 else "")

        # 绘制教室（明确教室与走廊的位置关系）
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']

            # 教室标签（标注教室名称）
            ax.text(x, y, z, classroom['name'], color='black', fontweight='bold', fontsize=9)
            # 教室位置点
            ax.scatter(x, y, z, color='darkblue', s=80, marker='s')
            # 教室边界（虚线框，明确教室范围）
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color='darkblue', linestyle='--', linewidth=2)

    # 设置坐标轴（明确单位，提升可读性）
    ax.set_xlabel('X Position (Meters)', fontsize=11)
    ax.set_ylabel('Y Position (Meters)', fontsize=11)
    ax.set_zlabel('Floor (Level)', fontsize=11)
    ax.set_title('School 3D Map (Stair → Corridor → Classroom Path)', fontsize=14, fontweight='bold')
    # 调整图例，避免重复
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    return fig, ax

# 自定义图数据结构（强化“楼梯临近走廊”的属性）
class Graph:
    def __init__(self):
        self.nodes = {}  # key: node_id, value: node_info
        self.stair_proximity = {}  # 记录走廊节点与最近楼梯的距离
        self.STAIR_PROXIMITY_THRESHOLD = 5  # 楼梯临近走廊的距离阈值（可调整，单位：米）

    def add_node(self, node_id, node_type, name, level, coordinates, stair_distance=None):
        """添加节点，新增stair_distance参数记录与最近楼梯的距离"""
        self.nodes[node_id] = {
            'type': node_type,       # 节点类型：classroom/stair/corridor
            'name': name,            # 节点名称（如“Class101”“Stair1”）
            'level': level,          # 节点所在楼层（如“Level0”）
            'coordinates': coordinates,  # 节点坐标 (x,y,z)
            'neighbors': {}          # 邻居节点：{neighbor_id: weight}
        }
        # 仅走廊节点记录与楼梯的距离
        if node_type == 'corridor' and stair_distance is not None:
            self.stair_proximity[node_id] = stair_distance

    def add_edge(self, node1, node2, weight):
        """添加边（双向，权重为距离）"""
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1]['neighbors'][node2] = weight
            self.nodes[node2]['neighbors'][node1] = weight

    def is_near_stair_corridor(self, node_id):
        """判断节点是否为“楼梯临近走廊”（距离≤阈值）"""
        if self.nodes[node_id]['type'] != 'corridor':
            return False
        return self.stair_proximity.get(node_id, float('inf')) <= self.STAIR_PROXIMITY_THRESHOLD

    def get_stair_before_corridor(self, corridor_node):
        """获取走廊节点之前最近的楼梯节点（用于路径验证）"""
        corridor_level = self.nodes[corridor_node]['level']
        # 查找同楼层所有楼梯节点
        stair_nodes = [
            node_id for node_id in self.nodes
            if self.nodes[node_id]['type'] == 'stair' and self.nodes[node_id]['level'] == corridor_level
        ]
        if not stair_nodes:
            return None
        # 计算走廊到各楼梯的距离，返回最近的楼梯
        min_dist = float('inf')
        nearest_stair = None
        for stair in stair_nodes:
            dist = euclidean_distance(self.nodes[corridor_node]['coordinates'], self.nodes[stair]['coordinates'])
            if dist < min_dist:
                min_dist = dist
                nearest_stair = stair
        return nearest_stair

# 计算欧氏距离（3D坐标）
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（强化楼梯→走廊→教室的连接逻辑）
def build_navigation_graph(school_data):
    graph = Graph()

    # 步骤1：添加所有节点（教室、楼梯、走廊）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']
        
        # 收集当前楼层楼梯坐标（用于计算走廊与楼梯的距离）
        stair_coords = [stair['coordinates'] for stair in level['stairs']]

        # 1.1 添加教室节点（仅连接到“临近教室的走廊”，避免直接连楼梯）
        for classroom in level['classrooms']:
            node_id = f"{classroom['name']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='classroom',
                name=classroom['name'],
                level=level_name,
                coordinates=classroom['coordinates']
            )

        # 1.2 添加楼梯节点（仅连接到“楼梯临近走廊”，不直接连教室）
        for stair in level['stairs']:
            node_id = f"{stair['name']}@{level_name}"
            graph.add_node(
                node_id=node_id,
                node_type='stair',
                name=stair['name'],
                level=level_name,
                coordinates=stair['coordinates']
            )

        # 1.3 添加走廊节点（记录与最近楼梯的距离，区分“临近教室”和“临近楼梯”）
        for corridor_idx, corridor in enumerate(level['corridors']):
            for point_idx, point in enumerate(corridor['points']):
                node_id = f"corridor_{point[0]}_{point[1]}_{z}"
                if node_id not in graph.nodes:
                    # 计算该走廊点与最近楼梯的距离
                    min_stair_dist = min(euclidean_distance(point, sc) for sc in stair_coords) if stair_coords else float('inf')
                    
                    graph.add_node(
                        node_id=node_id,
                        node_type='corridor',
                        name=f"Corridor_{corridor_idx+1}_Point_{point_idx+1}",
                        level=level_name,
                        coordinates=point,
                        stair_distance=min_stair_dist
                    )

    # 步骤2：添加边（严格控制连接关系：楼梯→走廊→教室，禁止跨级连接）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 2.1 教室 ↔ 走廊：仅连接“临近教室的走廊”（距离教室最近的2个走廊节点）
        for classroom in level['classrooms']:
            classroom_node_id = f"{classroom['name']}@{level_name}"
            classroom_coords = classroom['coordinates']
            
            # 筛选当前楼层的走廊节点
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            if not corridor_nodes:
                continue
            
            # 按距离排序，仅连接最近的2个走廊节点（确保教室只能通过走廊抵达）
            corridor_distances = sorted(
                [(node_id, euclidean_distance(classroom_coords, graph.nodes[node_id]['coordinates'])) 
                 for node_id in corridor_nodes],
                key=lambda x: x[1]
            )[:2]  # 取前2个最近的走廊
            
            for node_id, distance in corridor_distances:
                graph.add_edge(classroom_node_id, node_id, distance)

        # 2.2 楼梯 ↔ 走廊：仅连接“楼梯临近走廊”（距离楼梯≤阈值的走廊节点）
        for stair in level['stairs']:
            stair_node_id = f"{stair['name']}@{level_name}"
            stair_coords = stair['coordinates']
            
            # 筛选当前楼层的“楼梯临近走廊”
            near_stair_corridors = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' 
                and graph.nodes[node_id]['level'] == level_name
                and graph.is_near_stair_corridor(node_id)
            ]
            if not near_stair_corridors:
                continue
            
            # 连接楼梯与所有“临近走廊”，距离越近权重越低（优先选择）
            for node_id in near_stair_corridors:
                distance = euclidean_distance(stair_coords, graph.nodes[node_id]['coordinates'])
                weight = distance * 0.3  # 降低楼梯→走廊的权重，优先选择该路径
                graph.add_edge(stair_node_id, node_id, weight)

        # 2.3 走廊 ↔ 走廊：全连接（确保走廊间可通行，权重为距离）
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
                graph.add_edge(node1, node2, distance)

    # 2.4 楼梯 ↔ 楼梯：跨楼层连接（仅允许楼梯间跨楼层，权重为1.0表示“楼层切换成本”）
    for connection in school_data['buildingA']['connections']:
        from_stair_name, from_level = connection['from']
        to_stair_name, to_level = connection['to']
        
        from_stair_node = f"{from_stair_name}@{from_level}"
        to_stair_node = f"{to_stair_name}@{to_level}"
        
        if from_stair_node in graph.nodes and to_stair_node in graph.nodes:
            graph.add_edge(from_stair_node, to_stair_node, 1.0)

    return graph

# 改进的Dijkstra算法：强制“楼梯→走廊→终点教室”的路径逻辑
def dijkstra(graph, start_node, end_node):
    # 初始化距离：起点为0，其他为无穷大
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    unvisited_nodes = set(graph.nodes.keys())

    # 终点所在楼层（用于优化跨楼层路径）
    end_level = graph.nodes[end_node]['level'] if end_node in graph.nodes else None

    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=lambda x: distances[x])
        unvisited_nodes.remove(current_node)

        if distances[current_node] == float('inf'):
            break  # 无可达路径

        # 核心约束：终点教室的邻居仅保留“走廊节点”（禁止直接从楼梯/其他节点直达）
        neighbors = graph.nodes[current_node]['neighbors'].items()
        if current_node == end_node:
            neighbors = [
                (neighbor, weight) for neighbor, weight in neighbors
                if graph.nodes[neighbor]['type'] == 'corridor'  # 终点只能连走廊
            ]

        # 遍历邻居，更新距离
        for neighbor, weight in neighbors:
            # 额外优化：跨楼层时优先选择“楼梯临近走廊”（降低权重）
            extra_factor = 1.0
            current_level = graph.nodes[current_node]['level']
            if (current_level != end_level) and graph.nodes[current_node]['type'] == 'corridor':
                # 走廊越靠近楼梯，权重越低（优先引导到楼梯）
                stair_dist = graph.stair_proximity.get(current_node, float('inf'))
                extra_factor = 0.5 + (min(stair_dist, 10) / 10) * 0.5  # 范围：0.5~1.0
            
            new_distance = distances[current_node] + weight * extra_factor
            
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

# 路径验证：确保终点教室满足“楼梯→走廊→教室”的顺序
def validate_end_path(graph, path, end_node):
    if len(path) < 3 or path[-1] != end_node:
        return False  # 路径过短或终点不匹配
    
    # 验证终点教室的前序节点是走廊（走廊→教室）
    pre_end_node = path[-2]
    if graph.nodes[pre_end_node]['type'] != 'corridor':
        return False
    
    # 验证走廊的前序节点中包含楼梯（楼梯→走廊）
    # 查找走廊节点在路径中的位置
    corridor_index = path.index(pre_end_node)
    # 检查走廊节点之前的路径是否包含楼梯
    has_stair_before = any(
        graph.nodes[path[i]]['type'] == 'stair' 
        for i in range(corridor_index)
    )
    
    return has_stair_before

# 强制路径满足“楼梯→走廊→教室”的终点规则
def force_stair_corridor_classroom_path(graph, start_node, end_node):
    # 获取终点所在楼层的所有走廊节点
    end_level = graph.nodes[end_node]['level']
    valid_corridors = [
        node_id for node_id in graph.nodes
        if graph.nodes[node_id]['level'] == end_level and 
           graph.nodes[node_id]['type'] == 'corridor' and
           end_node in graph.nodes[node_id]['neighbors']  # 走廊必须直接连接终点教室
    ]
    
    if not valid_corridors:
        return None, "❌ 终点教室附近没有可用的走廊节点"
    
    # 筛选能通过楼梯到达的走廊
    valid_stair_corridors = []
    for corridor in valid_corridors:
        # 查找走廊对应的最近楼梯
        nearest_stair = graph.get_stair_before_corridor(corridor)
        if nearest_stair and nearest_stair in graph.nodes:
            valid_stair_corridors.append((corridor, nearest_stair))
    
    if not valid_stair_corridors:
        return None, "❌ 终点教室附近的走廊无法通过楼梯到达"
    
    # 计算最优路径：起点→楼梯→走廊→终点
    min_total_dist = float('inf')
    best_path = None
    
    for corridor, stair in valid_stair_corridors:
        # 1. 起点 → 楼梯
        dist1, prev1 = dijkstra(graph, start_node, stair)
        path1 = construct_path(prev1, stair)
        if not path1 or path1[0] != start_node:
            continue
            
        # 2. 楼梯 → 走廊
        dist2, prev2 = dijkstra(graph, stair, corridor)
        path2 = construct_path(prev2, corridor)[1:]  # 去除重复的楼梯节点
        if not path2:
            continue
            
        # 3. 走廊 → 终点
        path3 = [end_node]
        
        # 合并路径
        full_path = path1 + path2 + path3
        total_dist = dist1[stair] + dist2[corridor] + graph.nodes[corridor]['neighbors'][end_node]
        
        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best_path = full_path
    
    if best_path:
        return best_path, f"✅ 路径规划成功！总距离：{min_total_dist:.2f} 米（遵循楼梯→走廊→教室）"
    else:
        return None, "❌ 无法找到符合楼梯→走廊→教室规则的路径"

# 导航函数（集成路径验证）
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    start_node = f"{start_classroom}@{start_level}"
    end_node = f"{end_classroom}@{end_level}"

    # 基础校验
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "❌ 无效的教室或楼层"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 1. 第一次路径规划
    distances, previous_nodes = dijkstra(graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node)

    # 2. 验证路径是否满足“楼梯→走廊→教室”规则
    if path and validate_end_path(graph, path, end_node):
        total_distance = distances[end_node]
        return path, f"✅ 路径规划成功！总距离：{total_distance:.2f} 米（已满足楼梯→走廊→教室）"
    
    # 3. 若验证失败，强制调整路径
    path, msg = force_stair_corridor_classroom_path(graph, start_node, end_node)
    if path:
        return path, msg
    else:
        return None, "❌ 无法找到符合规则的有效路径"

# 在3D图上绘制路径（突出显示楼梯→走廊→教室的关键节点）
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

    # 绘制路径主线
    ax.plot(
        x_coords, y_coords, z_coords,
        color='red', linewidth=3, linestyle='-', marker='o', markersize=6
    )

    # 标记特殊节点（明确路径顺序）
    for i, (x, y, z, node_type) in enumerate(zip(x_coords, y_coords, z_coords, node_types)):
        if i == 0:  # 起点
            ax.scatter(x, y, z, color='green', s=300, marker='*', label='Start (Classroom)')
        elif i == len(path) - 1:  # 终点
            ax.scatter(x, y, z, color='purple', s=300, marker='*', label='End (Classroom)')
        elif node_type == 'stair':  # 楼梯
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stair')
        elif node_type == 'corridor':  # 走廊
            # 识别终点前的最后一个走廊（走廊→教室的关键节点）
            if i == len(path) - 2:
                ax.scatter(x, y, z, color='orange', s=180, marker='o', label='Final Corridor (to Classroom)')
            else:
                ax.scatter(x, y, z, color='blue', s=100, marker='o', label='Corridor')

    # 调整图例，避免重复
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
    st.subheader("3D地图与路径规划（严格遵循：楼梯→走廊→教室）")

    try:
        # 加载校园数据（请确保JSON文件在同一目录）
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

        nav_button = st.button("🔍 查找最优路径", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ 3D校园地图")
        
        # 初始化地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        # 路径规划逻辑
        if nav_button:
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            if path:
                st.success(message)
                st.markdown("#### 🛤️ 路径详情（楼梯→走廊→教室）")
                for i, node in enumerate(path, 1):
                    if node.startswith("corridor_"):
                        # 标记终点前的最后一个走廊
                        if i == len(path) - 1:
                            st.write(f"{i}. 终点前走廊（连接到教室）")
                        else:
                            st.write(f"{i}. 走廊")
                    else:
                        room, floor = node.split('@')
                        node_type = "楼梯" if "Stair" in room else "教室"
                        st.write(f"{i}. {room}（{node_type}，楼层：{floor}）")
                
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
