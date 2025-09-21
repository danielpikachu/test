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
        self.STAIR_PROXIMITY_THRESHOLD = 5  # 楼梯临近走廊的距离阈值（可调整）

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

    def is_near_stair_corridor(self, node_id):
        """判断节点是否为“楼梯临近走廊”"""
        if self.nodes[node_id]['type'] != 'corridor':
            return False
        # 走廊距离最近楼梯≤阈值 → 视为临近走廊
        return self.stair_proximity.get(node_id, float('inf')) <= self.STAIR_PROXIMITY_THRESHOLD

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

        # 2.1 教室 ↔ 走廊：优先连接最近的走廊（临近教室的走廊）
        for classroom in level['classrooms']:
            classroom_node_id = f"{classroom['name']}@{level_name}"
            classroom_coords = classroom['coordinates']
            
            # 找出当前楼层所有走廊节点并按距离排序
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            
            # 按距离排序，最近的走廊优先连接（权重更小）
            corridor_distances = [
                (node_id, euclidean_distance(classroom_coords, graph.nodes[node_id]['coordinates']))
                for node_id in corridor_nodes
            ]
            corridor_distances.sort(key=lambda x: x[1])
            
            # 添加连接，对最近的几个走廊给予权重优势
            for i, (node_id, distance) in enumerate(corridor_distances):
                # 对最近的走廊给予权重折扣（更优先选择）
                weight = distance * (0.5 if i < 2 else 1.0)  # 前2个最近的走廊权重减半
                graph.add_edge(classroom_node_id, node_id, weight)

        # 2.2 楼梯 ↔ 走廊：优先连接楼梯附近的走廊
        for stair in level['stairs']:
            stair_node_id = f"{stair['name']}@{level_name}"
            stair_coords = stair['coordinates']
            
            corridor_nodes = [
                node_id for node_id in graph.nodes 
                if graph.nodes[node_id]['type'] == 'corridor' and graph.nodes[node_id]['level'] == level_name
            ]
            
            # 计算楼梯到各走廊的距离
            corridor_distances = [
                (node_id, euclidean_distance(stair_coords, graph.nodes[node_id]['coordinates']))
                for node_id in corridor_nodes
            ]
            
            # 添加连接，楼梯附近的走廊权重更低
            for node_id, distance in corridor_distances:
                # 距离楼梯越近的走廊，权重越低（更优先选择）
                weight = distance * (0.3 if distance < 5 else 1.0)  # 楼梯5单位内的走廊权重大幅降低
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
                stair_factor = 0.7 if (graph.is_near_stair_corridor(node1) or graph.is_near_stair_corridor(node2)) else 1.0
                
                # 最终权重 = 距离 × 楼梯因子
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

# 改进的Dijkstra算法：强制终点教室需先经过楼梯临近走廊
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
            break

        # 核心修改：处理终点教室的邻居时，仅保留“楼梯临近走廊”
        neighbors = graph.nodes[current_node]['neighbors'].items()
        if current_node == end_node:
            # 终点教室的邻居必须是“楼梯临近走廊”，否则过滤（强制路径经过临近走廊）
            neighbors = [
                (neighbor, weight) for neighbor, weight in neighbors
                if graph.is_near_stair_corridor(neighbor)
            ]

        for neighbor, weight in neighbors:
            # 额外的权重调整：如果需要跨楼层，优先靠近楼梯的走廊
            extra_factor = 1.0
            
            # 当前节点是走廊且需要跨楼层时，靠近楼梯的走廊权重更低
            current_level = graph.nodes[current_node]['level']
            if (current_level != end_level) and (graph.nodes[current_node]['type'] == 'corridor'):
                # 走廊越靠近楼梯，额外因子越小（权重越低）
                stair_dist = graph.stair_proximity.get(current_node, float('inf'))
                extra_factor = 0.5 + (min(stair_dist, 10) / 10) * 0.5  # 范围0.5-1.0
            
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

# 路径验证：确保终点教室的前序节点是“楼梯临近走廊”
def validate_end_corridor(graph, path, end_node):
    if len(path) < 2 or path[-1] != end_node:
        return False  # 路径无效或终点不匹配
    # 终点前一个节点必须是“楼梯临近走廊”
    pre_end_node = path[-2]
    return graph.is_near_stair_corridor(pre_end_node)

# 强制终点经过楼梯临近走廊的路径调整
def force_end_near_stair_corridor(graph, start_node, end_node):
    # 获取终点所在楼层的所有楼梯临近走廊
    end_level = graph.nodes[end_node]['level']
    near_stair_corridors = [
        node_id for node_id in graph.nodes
        if graph.nodes[node_id]['level'] == end_level and 
           graph.is_near_stair_corridor(node_id)
    ]
    
    if not near_stair_corridors:
        return None, "❌ 终点楼层没有可用的楼梯临近走廊"
    
    # 计算从起点到每个临近走廊的路径，再连接到终点
    min_total_dist = float('inf')
    best_path = None
    
    for corridor in near_stair_corridors:
        # 1. 起点 → 临近走廊
        dist1, prev1 = dijkstra(graph, start_node, corridor)
        path1 = construct_path(prev1, corridor)
        if not path1 or path1[0] != start_node:
            continue
            
        # 2. 临近走廊 → 终点（必须直接连接）
        if end_node not in graph.nodes[corridor]['neighbors']:
            continue
            
        # 3. 合并路径
        full_path = path1 + [end_node]
        total_dist = dist1[corridor] + graph.nodes[corridor]['neighbors'][end_node]
        
        if total_dist < min_total_dist:
            min_total_dist = total_dist
            best_path = full_path
    
    if best_path:
        return best_path, f"✅ 路径规划成功！总距离：{min_total_dist:.2f} 单位（强制经过楼梯临近走廊）"
    else:
        return None, "❌ 无法找到经过终点楼梯临近走廊的有效路径"

# 导航函数（集成路径验证）
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    start_node = f"{start_classroom}@{start_level}"
    end_node = f"{end_classroom}@{end_level}"

    # 基础校验
    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "❌ 无效的教室或楼层"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 1. 第一次路径规划（强制终点邻居为临近走廊）
    distances, previous_nodes = dijkstra(graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node)

    # 2. 验证路径是否满足“终点前序是临近走廊”
    if path and validate_end_corridor(graph, path, end_node):
        total_distance = distances[end_node]
        return path, f"✅ 路径规划成功！总距离：{total_distance:.2f} 单位（已经过楼梯临近走廊）"
    
    # 3. 若验证失败，尝试强制调整路径（优先选择临近走廊）
    path, msg = force_end_near_stair_corridor(graph, start_node, end_node)
    if path:
        return path, msg
    else:
        return None, "❌ 无法找到经过终点楼梯临近走廊的有效路径"

# 强制"先到走廊"的路径计算（保证起点逻辑）
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
    node_details = []  # 存储节点详细信息（是否是临近走廊/楼梯附近走廊）

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
            elif graph.is_near_stair_corridor(node_id):  # 楼梯附近的走廊
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
    st.subheader("3D地图与优化路径规划（终点需经过楼梯临近走廊）")

    try:
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
                        elif graph.is_near_stair_corridor(node):
                            st.write(f"{i}. 楼梯附近的走廊（终点前必经）")
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
    # 全局graph变量用于界面显示
    graph = None
    main()
