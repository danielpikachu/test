import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
from itertools import combinations

plt.switch_backend('Agg')

# -------------------------- 数据读取与交叉点检测 --------------------------
def load_school_data_detailed(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def detect_corridor_crossings(level):
    """检测同一楼层内走廊之间的交叉点"""
    crossings = []
    corridors = level['corridors']
    
    for i, j in combinations(range(len(corridors)), 2):
        corridor1 = corridors[i]
        corridor2 = corridors[j]
        
        for p1_idx in range(len(corridor1['points']) - 1):
            p1_start = np.array(corridor1['points'][p1_idx])
            p1_end = np.array(corridor1['points'][p1_idx + 1])
            
            for p2_idx in range(len(corridor2['points']) - 1):
                p2_start = np.array(corridor2['points'][p2_idx])
                p2_end = np.array(corridor2['points'][p2_idx + 1])
                
                if are_lines_intersecting(p1_start, p1_end, p2_start, p2_end):
                    intersection = find_line_intersection(p1_start, p1_end, p2_start, p2_end)
                    if intersection is not None:
                        if not is_point_on_segment_end(intersection, p1_start, p1_end) and \
                           not is_point_on_segment_end(intersection, p2_start, p2_end):
                            crossing_id = f"crossing_{level['name']}_{len(crossings)+1}"
                            crossings.append({
                                'id': crossing_id,
                                'coordinates': tuple(intersection),
                                'corridors': [i, j]
                            })
    return crossings

def are_lines_intersecting(p1s, p1e, p2s, p2e):
    if not np.isclose(p1s[2], p2s[2]):
        return False
        
    def ccw(A, B, C):
        return (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    
    A, B, C, D = p1s[:2], p1e[:2], p2s[:2], p2e[:2]
    return (ccw(A,B,C)*ccw(A,B,D) < 0) and (ccw(C,D,A)*ccw(C,D,B) < 0)

def find_line_intersection(p1s, p1e, p2s, p2e):
    x1, y1, z = p1s
    x2, y2, _ = p1e
    x3, y3, _ = p2s
    x4, y4, _ = p2e
    
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None
    
    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_num / den
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    u = u_num / den
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (x, y, z)
    return None

def is_point_on_segment_end(point, start, end):
    return np.isclose(point, start).all() or np.isclose(point, end).all()

# -------------------------- 3D绘图功能 --------------------------
def plot_3d_map(school_data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    floor_colors = {0: 'blue', 2: 'green', 5: 'orange', -2: 'purple', 10: 'cyan'}  

    for level in school_data['buildingA']['levels']:
        z = level['z']
        color = floor_colors.get(z, 'gray')
        level_name = level['name']
        
        # 绘制交叉点
        crossings = detect_corridor_crossings(level)
        for crossing in crossings:
            x, y, _ = crossing['coordinates']
            ax.scatter(x, y, z, color='yellow', s=150, marker='X', label='Crossing' if z == -2 else "")

        # 收集走廊点
        all_corridor_points = []
        for corridor in level['corridors']:
            all_corridor_points.extend(corridor['points'])
        if not all_corridor_points:
            continue  

        # 绘制楼层平面
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
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stairs' if z == -2 else "")

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

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Floor')
    ax.set_title('School 3D Map with Crossings & Navigation')
    ax.legend()

    return fig, ax

# -------------------------- 导航图与路径计算 --------------------------
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

def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

def build_navigation_graph(school_data):
    graph = Graph()

    for level in school_data['buildingA']['levels']:
        z = level['z']
        level_name = level['name']
        
        # 添加交叉点节点
        crossings = detect_corridor_crossings(level)
        for crossing in crossings:
            graph.add_node(
                node_id=crossing['id'],
                node_type='crossing',
                name=crossing['id'],
                level=level_name,
                coordinates=crossing['coordinates']
            )

        # 添加教室
        for classroom in level['classrooms']:
            node_id = f"{classroom['name']}@{level_name}"
            graph.add_node(node_id,
                           'classroom',
                           classroom['name'],
                           level_name,
                           classroom['coordinates'])

        # 添加楼梯
        for stair in level['stairs']:
            node_id = f"{stair['name']}@{level_name}"
            graph.add_node(node_id,
                          'stair',
                           stair['name'],
                           level_name,
                           stair['coordinates'])

    # 添加连接关系
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        crossings = detect_corridor_crossings(level)
        
        # 获取该楼层所有节点
        level_nodes = [n for n in graph.nodes.keys() if graph.nodes[n]['level'] == level_name]
        
        # 为节点建立连接（优化：优先连接同走廊节点）
        for i in range(len(level_nodes)):
            for j in range(i + 1, len(level_nodes)):
                node1 = level_nodes[i]
                node2 = level_nodes[j]
                coords1 = graph.nodes[node1]['coordinates']
                coords2 = graph.nodes[node2]['coordinates']
                
                # 检查是否在同一走廊
                on_same_corridor = False
                for corridor in level['corridors']:
                    if is_point_on_corridor(coords1, corridor) and is_point_on_corridor(coords2, corridor):
                        on_same_corridor = True
                        break
                
                # 同一走廊的节点赋予较小权重，优先选择
                distance = euclidean_distance(coords1, coords2)
                weight = distance if on_same_corridor else distance * 1.2  # 不同走廊增加20%权重
                graph.add_edge(node1, node2, weight)

    # 跨楼层连接
    for connection in school_data['buildingA']['connections']:
        from_stair, from_level = connection['from']
        to_stair, to_level = connection['to']

        from_node = f"{from_stair}@{from_level}"
        to_node = f"{to_stair}@{to_level}"

        if from_node in graph.nodes and to_node in graph.nodes:
            graph.add_edge(from_node, to_node, 1.0)

    return graph

# 新增：检查点是否在走廊上
def is_point_on_corridor(point, corridor):
    point = np.array(point)
    for i in range(len(corridor['points']) - 1):
        p1 = np.array(corridor['points'][i])
        p2 = np.array(corridor['points'][i+1])
        
        # 检查点是否在线段上
        if np.linalg.norm(np.cross(p2-p1, point-p1)) < 1e-6 and \
           min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]) and \
           min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1]):
            return True
    return False

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

def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path

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

# 修复路径绘制函数：确保所有节点（包括交叉点）都被正确连接
def plot_path(ax, graph, path):
    # 提取所有节点的坐标
    coords = [graph.nodes[node]['coordinates'] for node in path]
    
    # 分解坐标为X, Y, Z列表
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    z = [p[2] for p in coords]
    
    # 绘制完整路径线（重点修复：确保所有节点都被连接）
    ax.plot(x, y, z, color='red', linewidth=3, marker='o', markersize=8)
    
    # 为不同类型的节点添加特殊标记
    for i, node in enumerate(path):
        node_data = graph.nodes[node]
        node_type = node_data['type']
        x, y, z = node_data['coordinates']
        
        if node_type == 'crossing':
            # 交叉点用黄色X标记
            ax.scatter(x, y, z, color='yellow', s=200, marker='X', label='Crossing' if i == 0 else "")
        elif i == 0:
            # 起点用绿色星形标记
            ax.scatter(x, y, z, color='green', s=300, marker='*', label='Start')
        elif i == len(path) - 1:
            # 终点用紫色星形标记
            ax.scatter(x, y, z, color='purple', s=300, marker='*', label='End')
        elif node_type == 'stair':
            # 楼梯用红色三角形标记
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stair' if i == 1 else "")

    # 添加图例
    ax.legend()

def get_classroom_info(school_data):
    levels = []
    classrooms_by_level = {}
    
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        levels.append(level_name)
        classrooms = [classroom['name'] for classroom in level['classrooms']]
        classrooms_by_level[level_name] = classrooms
        
    return levels, classrooms_by_level

# -------------------------- Streamlit界面 --------------------------
def main():
    st.title("🏫 School Campus Navigation System")
    st.subheader("3D Map with Visible Crossings in Path")

    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_navigation_graph(school_data)
        levels, classrooms_by_level = get_classroom_info(school_data)
        st.success("✅ School data loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Error: 'school_data_detailed.json' not found.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📍 Select Locations")
        
        # 起点选择
        st.markdown("#### Start Point")
        start_level = st.selectbox("Floor", levels, key="start_level")
        start_classrooms = classrooms_by_level[start_level]
        start_classroom = st.selectbox("Classroom", start_classrooms, key="start_classroom")

        # 终点选择
        st.markdown("#### End Point")
        end_level = st.selectbox("Floor", levels, key="end_level")
        end_classrooms = classrooms_by_level[end_level]
        end_classroom = st.selectbox("Classroom", end_classrooms, key="end_classroom")

        # 导航按钮
        nav_button = st.button("🔍 Find Shortest Path", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ 3D Campus Map")
        
        # 初始显示地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        # 计算并显示路径
        if nav_button:
            path, message = navigate(nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            if path:
                st.success(f"📊 Navigation Result: {message}")
                st.markdown("#### 🛤️ Path Details (including crossings)")
                for i, node in enumerate(path, 1):
                    node_info = nav_graph.nodes[node]
                    room, floor = node.split('@') if '@' in node else (node, node_info['level'])
                    st.write(f"{i}. {room} (Floor: {floor}, Type: {node_info['type']})")
                
                # 绘制带路径和交叉点的地图
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, nav_graph, path)
                st.session_state['fig'] = fig
            else:
                st.error(f"❌ {message}")
        
        # 显示3D图
        st.pyplot(st.session_state['fig'])

if __name__ == "__main__":
    main()
    
