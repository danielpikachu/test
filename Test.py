import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
from itertools import combinations

plt.switch_backend('Agg')

# -------------------------- 数据处理与几何计算 --------------------------
def load_school_data_detailed(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# 检测走廊交叉点
def detect_corridor_crossings(level):
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

# 计算点到线段的最近点
def nearest_point_on_segment(point, seg_start, seg_end):
    p = np.array(point)
    a = np.array(seg_start)
    b = np.array(seg_end)
    
    # 计算投影参数
    ab = b - a
    ap = p - a
    t = max(0, min(1, np.dot(ap, ab) / (np.dot(ab, ab) + 1e-10)))  # 避免除零
    projection = a + t * ab
    return tuple(projection)

# 找到教室到最近的走廊点
def find_nearest_corridor_point(classroom_coords, level):
    min_dist = float('inf')
    nearest_point = None
    
    for corridor in level['corridors']:
        for i in range(len(corridor['points']) - 1):
            seg_start = corridor['points'][i]
            seg_end = corridor['points'][i + 1]
            
            # 计算教室到当前走廊线段的最近点
            point = nearest_point_on_segment(classroom_coords, seg_start, seg_end)
            dist = euclidean_distance(classroom_coords, point)
            
            if dist < min_dist:
                min_dist = dist
                nearest_point = point
    
    return nearest_point, min_dist

# 几何辅助函数
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

def is_point_on_corridor(point, corridor):
    point = np.array(point)
    for i in range(len(corridor['points']) - 1):
        p1 = np.array(corridor['points'][i])
        p2 = np.array(corridor['points'][i+1])
        
        if np.linalg.norm(np.cross(p2-p1, point-p1)) < 1e-6 and \
           min(p1[0], p2[0]) <= point[0] <= max(p1[0], p2[0]) and \
           min(p1[1], p2[1]) <= point[1] <= max(p1[1], p2[1]):
            return True
    return False

def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

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

        # 绘制走廊
        for corridor in level['corridors']:
            points = corridor['points']
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z_coords = [p[2] for p in points]
            ax.plot(x, y, z_coords, color=color, linewidth=5, alpha=0.7)

        # 绘制楼梯
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stairs' if z == -2 else "")

        # 绘制教室
        for classroom in level['classrooms']:
            x, y, _ = classroom['coordinates']
            width, depth = classroom['size']
            ax.text(x, y, z, classroom['name'], color='black', fontweight='bold')
            ax.scatter(x, y, z, color=color, s=80)
            ax.plot([x, x + width, x + width, x, x],
                    [y, y, y + depth, y + depth, y],
                    [z, z, z, z, z],
                    color=color, linestyle='--')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Floor')
    ax.set_title('School 3D Map with Complete Path Nodes')
    ax.legend()

    return fig, ax

# -------------------------- 导航图与路径计算 --------------------------
class Graph:
    def __init__(self):
        self.nodes = {}  # 包含所有类型节点：教室、走廊点、交叉点、楼梯

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

def build_enhanced_graph(school_data):
    graph = Graph()

    for level in school_data['buildingA']['levels']:
        z = level['z']
        level_name = level['name']
        crossings = detect_corridor_crossings(level)
        
        # 1. 添加交叉点
        for crossing in crossings:
            graph.add_node(
                node_id=crossing['id'],
                node_type='crossing',
                name=crossing['id'],
                level=level_name,
                coordinates=crossing['coordinates']
            )

        # 2. 添加楼梯
        for stair in level['stairs']:
            node_id = f"{stair['name']}@{level_name}"
            graph.add_node(node_id,
                          'stair',
                           stair['name'],
                           level_name,
                           stair['coordinates'])

        # 3. 添加教室的"最近走廊点"
        for classroom in level['classrooms']:
            class_id = classroom['name']
            class_coords = classroom['coordinates']
            
            # 计算最近走廊点
            nearest_corridor_pt, _ = find_nearest_corridor_point(class_coords, level)
            corridor_node_id = f"corridor_{class_id}@{level_name}"
            
            # 添加走廊点节点
            graph.add_node(
                node_id=corridor_node_id,
                node_type='corridor_point',
                name=f"Corridor near {class_id}",
                level=level_name,
                coordinates=nearest_corridor_pt
            )

            # 添加教室到最近走廊点的边（仅单向，实际导航是教室→走廊）
            dist = euclidean_distance(class_coords, nearest_corridor_pt)
            graph.add_edge(f"{class_id}@{level_name}", corridor_node_id, dist)

        # 4. 连接同一楼层的所有节点（走廊点、交叉点、楼梯）
        level_nodes = [n for n in graph.nodes.keys() if graph.nodes[n]['level'] == level_name]
        for i in range(len(level_nodes)):
            for j in range(i + 1, len(level_nodes)):
                node1 = level_nodes[i]
                node2 = level_nodes[j]
                coords1 = graph.nodes[node1]['coordinates']
                coords2 = graph.nodes[node2]['coordinates']
                
                # 计算距离并添加边
                distance = euclidean_distance(coords1, coords2)
                graph.add_edge(node1, node2, distance)

    # 5. 跨楼层楼梯连接
    for connection in school_data['buildingA']['connections']:
        from_stair, from_level = connection['from']
        to_stair, to_level = connection['to']

        from_node = f"{from_stair}@{from_level}"
        to_node = f"{to_stair}@{to_level}"

        if from_node in graph.nodes and to_node in graph.nodes:
            graph.add_edge(from_node, to_node, 1.0)  # 楼梯权重

    return graph

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

def construct_detailed_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path

def navigate_detailed(graph, start_classroom, start_level, end_classroom, end_level, school_data):
    # 1. 起点处理：教室 → 最近走廊点
    start_class_node = f"{start_classroom}@{start_level}"
    start_corridor_node = f"corridor_{start_classroom}@{start_level}"
    
    # 2. 终点处理：最近走廊点 → 教室
    end_class_node = f"{end_classroom}@{end_level}"
    end_corridor_node = f"corridor_{end_classroom}@{end_level}"
    
    # 3. 计算核心路径：起点走廊点 → 终点走廊点
    if start_corridor_node not in graph.nodes or end_corridor_node not in graph.nodes:
        return None, "Missing corridor nodes for start or end classroom"

    distances, previous_nodes = dijkstra(graph, start_corridor_node)
    core_path = construct_detailed_path(previous_nodes, end_corridor_node)
    
    if not core_path:
        return None, "No path exists between corridor points"

    # 4. 拼接完整路径：教室 → 走廊点 → ... → 走廊点 → 教室
    full_path = [start_class_node] + core_path + [end_class_node]
    total_distance = distances[end_corridor_node]
    
    # 添加教室到走廊点的距离
    start_dist = euclidean_distance(
        graph.nodes[start_class_node]['coordinates'],
        graph.nodes[start_corridor_node]['coordinates']
    )
    end_dist = euclidean_distance(
        graph.nodes[end_corridor_node]['coordinates'],
        graph.nodes[end_class_node]['coordinates']
    )
    total_distance += start_dist + end_dist

    return full_path, f"Total distance: {total_distance:.2f} units"

# 绘制完整路径（包含所有节点）
def plot_detailed_path(ax, graph, path):
    # 提取所有节点坐标
    coords = [graph.nodes[node]['coordinates'] for node in path]
    x = [p[0] for p in coords]
    y = [p[1] for p in coords]
    z = [p[2] for p in coords]
    
    # 绘制完整路径线
    ax.plot(x, y, z, color='red', linewidth=3, linestyle='-')
    
    # 为不同类型节点添加特殊标记
    for i, node in enumerate(path):
        node_data = graph.nodes[node]
        node_type = node_data['type']
        px, py, pz = node_data['coordinates']
        
        if node_type == 'classroom':
            # 教室节点（起点/终点）
            marker = '*' if i in [0, len(path)-1] else 'o'
            color = 'green' if i == 0 else 'purple' if i == len(path)-1 else 'blue'
            size = 300 if i in [0, len(path)-1] else 100
            ax.scatter(px, py, pz, color=color, s=size, marker=marker, 
                      label='Start' if i == 0 else 'End' if i == len(path)-1 else "")
            
        elif node_type == 'corridor_point':
            # 教室附近的走廊点
            ax.scatter(px, py, pz, color='orange', s=150, marker='s', label='Corridor Point' if i == 1 else "")
            
        elif node_type == 'crossing':
            # 交叉点
            ax.scatter(px, py, pz, color='yellow', s=200, marker='X', label='Crossing' if 'crossing' not in [n.split('_')[0] for n in path[:i]] else "")
            
        elif node_type == 'stair':
            # 楼梯节点
            ax.scatter(px, py, pz, color='red', s=200, marker='^', label='Stair' if 'stair' not in [n.split('@')[0].lower() for n in path[:i]] else "")

    ax.legend()

# -------------------------- Streamlit界面 --------------------------
def get_classroom_info(school_data):
    levels = []
    classrooms_by_level = {}
    
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        levels.append(level_name)
        classrooms = [classroom['name'] for classroom in level['classrooms']]
        classrooms_by_level[level_name] = classrooms
        
    return levels, classrooms_by_level

def main():
    st.title("🏫 School Navigation System")
    st.subheader("Complete Path Visualization (Classroom → Corridor → Crossing → Stair)")

    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_enhanced_graph(school_data)
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
        nav_button = st.button("🔍 Find Detailed Path", use_container_width=True)

    with col2:
        st.markdown("### 🗺️ 3D Map with All Path Nodes")
        
        # 初始显示地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        # 计算并显示详细路径
        if nav_button:
            path, message = navigate_detailed(nav_graph, start_classroom, start_level, 
                                             end_classroom, end_level, school_data)
            
            if path:
                st.success(f"📊 Navigation Result: {message}")
                st.markdown("#### 🛤️ Complete Path Details")
                for i, node in enumerate(path, 1):
                    node_info = nav_graph.nodes[node]
                    node_type = node_info['type']
                    st.write(f"{i}. {node_info['name']} (Type: {node_type}, Floor: {node_info['level']})")
                
                # 绘制带所有节点的路径
                fig, ax = plot_3d_map(school_data)
                plot_detailed_path(ax, nav_graph, path)
                st.session_state['fig'] = fig
            else:
                st.error(f"❌ {message}")
        
        # 显示3D图
        st.pyplot(st.session_state['fig'])

if __name__ == "__main__":
    main()
    
