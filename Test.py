import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
from scipy.spatial import KDTree  # 用于快速查找最近点

# -------------------------- 1. 基础配置：解决Streamlit matplotlib渲染问题 --------------------------
plt.switch_backend('Agg')

# -------------------------- 2. 核心功能：数据处理与路径规划 --------------------------
# 读取JSON数据
def load_school_data_detailed(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# 提取单个楼层的所有走廊点、线段和交叉点
def get_floor_corridor_data(level_data):
    """
    提取楼层的走廊点、交叉点
    返回：
        all_corridor_points: 该楼层所有走廊的点（去重）
        corridor_segments: 该楼层所有走廊的线段（用于计算交叉点）
        intersection_points: 该楼层走廊的交叉点
    """
    all_corridor_points = []
    corridor_segments = []  # 存储所有走廊的线段（(p1, p2)）
    
    # 1. 提取所有走廊点和线段
    for corridor in level_data['corridors']:
        points = corridor['points']
        # 去重添加走廊点（避免重复计算）
        for p in points:
            if p not in all_corridor_points:
                all_corridor_points.append(p)
        # 提取走廊的线段（连续两点组成一段）
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            corridor_segments.append((p1, p2))
    
    # 2. 计算走廊交叉点（两线段不共线且相交时）
    intersection_points = []
    def ccw(A, B, C):
        """判断三点是否逆时针排列（用于线段相交判断）"""
        return (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])
    
    def segments_intersect(p1, p2, p3, p4):
        """判断两线段是否相交，返回交点（无交点返回None）"""
        # 线段1: p1-p2，线段2: p3-p4
        A, B, C, D = p1, p2, p3, p4
        # 快速排斥实验
        if (max(A[0], B[0]) < min(C[0], D[0]) or
            max(C[0], D[0]) < min(A[0], B[0]) or
            max(A[1], B[1]) < min(C[1], D[1]) or
            max(C[1], D[1]) < min(A[1], B[1])):
            return None
        # 跨立实验
        ccw1 = ccw(A, B, C)
        ccw2 = ccw(A, B, D)
        ccw3 = ccw(C, D, A)
        ccw4 = ccw(C, D, B)
        # 两线段不共线且相交
        if (ccw1 * ccw2 < 0) and (ccw3 * ccw4 < 0):
            # 计算交点（参数方程法）
            t = ((A[0]-C[0])*(C[1]-D[1]) - (A[1]-C[1])*(C[0]-D[0])) / \
                ((A[0]-B[0])*(C[1]-D[1]) - (A[1]-B[1])*(C[0]-D[0]))
            intersection = A + t * (B - A)
            return intersection.tolist()
        return None
    
    # 遍历所有线段对，计算交叉点（去重）
    for i in range(len(corridor_segments)):
        p1, p2 = corridor_segments[i]
        for j in range(i + 1, len(corridor_segments)):
            p3, p4 = corridor_segments[j]
            intersect = segments_intersect(p1, p2, p3, p4)
            if intersect and intersect not in intersection_points:
                # 确保交叉点Z坐标与楼层一致
                intersect[2] = level_data['z']
                intersection_points.append(intersect)
    
    return all_corridor_points, corridor_segments, intersection_points

# 绘制3D地图（包含走廊点和交叉点）
def plot_3d_map(school_data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为不同楼层使用不同颜色
    floor_colors = {-2: 'blue', 2: 'green', 5: 'orange', 10: 'red'}  

    # 处理每个楼层
    for level in school_data['buildingA']['levels']:
        z = level['z']
        color = floor_colors.get(z, 'gray')
        level_name = level['name']
        
        # 获取当前楼层的走廊点、线段、交叉点
        all_corridor_points, _, intersection_points = get_floor_corridor_data(level)
        
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
            ax.plot(x, y, z_coords, color=color, linewidth=5)
        
        # 绘制走廊点（灰色小圆点）
        corridor_x = [p[0] for p in all_corridor_points]
        corridor_y = [p[1] for p in all_corridor_points]
        corridor_z = [p[2] for p in all_corridor_points]
        ax.scatter(corridor_x, corridor_y, corridor_z, color='gray', s=30, label='Corridor Point' if z == -2 else "")
        
        # 绘制走廊交叉点（黄色大圆点，标注"Cross"）
        if intersection_points:
            cross_x = [p[0] for p in intersection_points]
            cross_y = [p[1] for p in intersection_points]
            cross_z = [p[2] for p in intersection_points]
            ax.scatter(cross_x, cross_y, cross_z, color='yellow', s=200, marker='D', label='Cross Point' if z == -2 else "")
            # 标注交叉点
            for p in intersection_points:
                ax.text(p[0], p[1], p[2], 'Cross', color='black', fontweight='bold')

        # 绘制楼梯
        for stair in level['stairs']:
            x, y, _ = stair['coordinates']
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Stairs' if z == -2 else "")
            ax.text(x, y, z, stair['name'], color='white', fontweight='bold')

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
    ax.set_title('School 3D Map with Navigation (Full Path)')
    ax.legend()

    return fig, ax

# 获取指定楼层的走廊信息（用于路径补全）
def get_level_corridor_info(school_data, level_name):
    """根据楼层名，返回该楼层的：教室→最近走廊点映射、走廊点、交叉点、楼梯坐标"""
    for level in school_data['buildingA']['levels']:
        if level['name'] == level_name:
            z = level['z']
            # 获取走廊点和交叉点
            all_corridor_points, _, intersection_points = get_floor_corridor_data(level)
            # 构建教室→最近走廊点的映射（用KDTree加速最近邻查找）
            corridor_kdtree = KDTree(all_corridor_points)
            classroom_nearest_corridor = {}
            for classroom in level['classrooms']:
                cls_coords = np.array(classroom['coordinates'])
                # 查找最近的走廊点
                dist, idx = corridor_kdtree.query(cls_coords)
                nearest_corridor = all_corridor_points[idx]
                classroom_nearest_corridor[classroom['name']] = nearest_corridor
            # 获取楼梯坐标
            stair_coords = level['stairs'][0]['coordinates'] if level['stairs'] else None
            return {
                'classroom_nearest_corridor': classroom_nearest_corridor,
                'all_corridor_points': all_corridor_points,
                'intersection_points': intersection_points,
                'stair_coords': stair_coords,
                'z': z
            }
    return None  # 未找到楼层

# 计算两点之间的"走廊路径"（补全走廊点和交叉点）
def compute_corridor_path(points_list, corridor_points, intersection_points):
    """
    计算两点之间经过走廊点和交叉点的路径
    points_list: 原始路径节点（如[教室坐标, 楼梯坐标]）
    corridor_points: 该楼层所有走廊点
    intersection_points: 该楼层所有交叉点
    返回：补全后的走廊路径点列表
    """
    if len(points_list) < 2:
        return points_list
    
    full_corridor_path = []
    corridor_kdtree = KDTree(corridor_points)
    cross_kdtree = KDTree(intersection_points) if intersection_points else None
    
    # 遍历原始路径的相邻节点对，补全走廊路径
    for i in range(len(points_list) - 1):
        start = np.array(points_list[i])
        end = np.array(points_list[i + 1])
        
        # 步骤1：找到起点到终点之间的所有走廊点（在两点连线上或附近）
        # 计算两点连线的参数方程：start + t*(end - start)，t∈[0,1]
        t_list = []
        candidate_points = []
        for p in corridor_points:
            p_np = np.array(p)
            # 判断点是否在两点连线的附近（距离<0.5单位）
            dist_to_line = np.linalg.norm(np.cross(end - start, start - p_np)) / np.linalg.norm(end - start)
            if dist_to_line < 0.5:
                # 计算t值（判断点是否在两点之间）
                t = np.dot(p_np - start, end - start) / (np.linalg.norm(end - start) ** 2)
                if 0 <= t <= 1:
                    t_list.append(t)
                    candidate_points.append(p)
        
        # 步骤2：按t值排序（从起点到终点的顺序）
        if candidate_points:
            sorted_indices = np.argsort(t_list)
            sorted_corridor_points = [candidate_points[idx] for idx in sorted_indices]
        else:
            # 无走廊点时，用起点→最近走廊点→终点（避免路径断连）
            dist_start, idx_start = corridor_kdtree.query(start)
            dist_end, idx_end = corridor_kdtree.query(end)
            sorted_corridor_points = [corridor_points[idx_start], corridor_points[idx_end]]
        
        # 步骤3：插入交叉点（如果交叉点在当前路径段上）
        if cross_kdtree and intersection_points:
            for cross in intersection_points:
                cross_np = np.array(cross)
                # 判断交叉点是否在当前路径段的走廊点之间
                dist_to_line = np.linalg.norm(np.cross(end - start, start - cross_np)) / np.linalg.norm(end - start)
                if dist_to_line < 0.5:
                    t = np.dot(cross_np - start, end - start) / (np.linalg.norm(end - start) ** 2)
                    if 0 <= t <= 1 and cross not in sorted_corridor_points:
                        # 插入到正确位置
                        sorted_corridor_points.append(cross)
                        # 重新按t值排序
                        t_cross = [np.dot(np.array(p) - start, end - start) / (np.linalg.norm(end - start) ** 2) 
                                   for p in sorted_corridor_points]
                        sorted_indices = np.argsort(t_cross)
                        sorted_corridor_points = [sorted_corridor_points[idx] for idx in sorted_indices]
        
        # 步骤4：添加到完整路径（避免重复节点）
        for p in sorted_corridor_points:
            if not full_corridor_path or (np.array(p) != np.array(full_corridor_path[-1])).any():
                full_corridor_path.append(p)
    
    return full_corridor_path

# 自定义图数据结构
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

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图
def build_navigation_graph(school_data):
    graph = Graph()

    # 添加所有位置节点（教室、楼梯）
    for level in school_data['buildingA']['levels']:
        z = level['z']
        level_name = level['name']

        # 添加教室
        for classroom in level['classrooms']:
            node_id = f"classroom_{classroom['name']}@{level_name}"
            graph.add_node(node_id,
                           'classroom',
                           classroom['name'],
                           level_name,
                           classroom['coordinates'])

        # 添加楼梯
        for stair in level['stairs']:
            node_id = f"stair_{stair['name']}@{level_name}"
            graph.add_node(node_id,
                          'stair',
                           stair['name'],
                           level_name,
                           stair['coordinates'])

    # 添加连接关系
    # 1. 同一楼层内的连接（教室-楼梯，基于欧氏距离）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        # 获取该楼层所有节点
        level_nodes = [n for n in graph.nodes.keys() if graph.nodes[n]['level'] == level_name]
        # 所有节点两两连接（确保连通性）
        for i in range(len(level_nodes)):
            for j in range(i + 1, len(level_nodes)):
                coords1 = graph.nodes[level_nodes[i]]['coordinates']
                coords2 = graph.nodes[level_nodes[j]]['coordinates']
                distance = euclidean_distance(coords1, coords2)
                graph.add_edge(level_nodes[i], level_nodes[j], distance)

    # 2. 跨楼层连接（楼梯）
    for connection in school_data['buildingA']['connections']:
        from_stair, from_level = connection['from']
        to_stair, to_level = connection['to']

        from_node = f"stair_{from_stair}@{from_level}"
        to_node = f"stair_{to_stair}@{to_level}"

        if from_node in graph.nodes and to_node in graph.nodes:
            graph.add_edge(from_node, to_node, 1.0)  # 楼梯连接权重设为1

    return graph

# 自定义Dijkstra算法
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

# 生成最短路径（原始路径：教室→楼梯→...→教室）
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path

# 导航函数（核心修改：补全走廊路径和交叉点）
def navigate(school_data, graph, start_classroom, start_level, end_classroom, end_level):
    start_node = f"classroom_{start_classroom}@{start_level}"
    end_node = f"classroom_{end_classroom}@{end_level}"

    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "Invalid classroom or level"

    # 1. 计算原始路径（教室→楼梯→...→教室）
    distances, previous_nodes = dijkstra(graph, start_node)
    raw_path = construct_path(previous_nodes, end_node)
    if not raw_path:
        return None, "No path exists between these classrooms"

    # 2. 解析原始路径，提取关键节点（教室和楼梯的坐标）
    key_points = []  # 存储关键节点坐标
    key_levels = []  # 存储对应楼层
    for node in raw_path:
        node_type, rest = node.split('_', 1)
        name, level = rest.split('@')
        key_points.append(graph.nodes[node]['coordinates'])
        key_levels.append(level)

    # 3. 按楼层补全路径（添加走廊点和交叉点）
    full_path = []
    i = 0
    while i < len(key_levels):
        current_level = key_levels[i]
        # 找到当前楼层的所有连续节点
        j = i
        while j < len(key_levels) and key_levels[j] == current_level:
            j += 1
        
        # 获取当前楼层的走廊信息
        level_info = get_level_corridor_info(school_data, current_level)
        if not level_info:
            i = j
            continue
        
        # 提取当前楼层的关键节点
        current_level_key_points = key_points[i:j]
        
        # 补全当前楼层的走廊路径（包含走廊点和交叉点）
        corridor_path = compute_corridor_path(
            current_level_key_points,
            level_info['all_corridor_points'],
            level_info['intersection_points']
        )
        
        # 添加到完整路径
        full_path.extend(corridor_path)
        
        i = j

    # 4. 计算总距离
    total_distance = 0
    for i in range(len(full_path) - 1):
        total_distance += euclidean_distance(full_path[i], full_path[i+1])

    return full_path, f"Total distance: {total_distance:.2f} units"

# 在3D图上绘制完整路径
def plot_path(ax, path):
    if not path:
        return
        
    x = [p[0] for p in path]
    y = [p[1] for p in path]
    z = [p[2] for p in path]

    # 绘制完整路径
    ax.plot(x, y, z, color='red', linewidth=3, linestyle='-', marker='o')

    # 标记起点和终点
    ax.scatter(x[0], y[0], z[0], color='green', s=300, marker='*', label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='purple', s=300, marker='*', label='End')
    
    # 标记路径中的关键节点（交叉点）
    for i, point in enumerate(path):
        if i > 0 and i < len(path) - 1:  # 跳过起点和终点
            ax.text(point[0], point[1], point[2], f'P{i}', color='darkred', fontsize=8)

# 获取所有楼层和教室信息（适配Streamlit下拉框）
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
    # 1. 页面标题和数据加载
    st.title("🏫 School Campus Navigation System")
    st.subheader("3D Map & Full Path Finder")

    # 加载JSON数据
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_navigation_graph(school_data)
        levels, classrooms_by_level = get_classroom_info(school_data)
        st.success("✅ School data loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Error: 'school_data_detailed.json' not found. Please check the file path.")
        return  # 数据加载失败，终止程序

    # 2. 布局：左右分栏
    col1, col2 = st.columns([1, 2])

    with col1:
        # 左侧：起点和终点选择
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
        nav_button = st.button("🔍 Find Full Path", use_container_width=True)

    with col2:
        # 右侧：显示3D地图和导航结果
        st.markdown("### 🗺️ 3D Campus Map")
        
        # 初始显示空的3D地图
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        # 点击导航按钮后，计算路径并更新地图
        if nav_button:
            # 调用导航函数
            path, message = navigate(school_data, nav_graph, start_classroom, start_level, end_classroom, end_level)
            
            # 显示导航结果
            if path:
                st.success(f"📊 Navigation Result: {message}")
                # 显示路径详情
                st.markdown("#### 🛤️ Path Details")
                for i, point in enumerate(path[:10] + (["..."] if len(path) > 10 else []) + path[-10:]):
                    if isinstance(point, list):
                        st.write(f"{i+1}. Coordinates: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
                
                # 重新绘制带路径的3D图
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, path)
                st.session_state['fig'] = fig
            else:
                st.error(f"❌ {message}")
        
        # 显示3D图
        st.pyplot(st.session_state['fig'])

# -------------------------- 4. 运行主函数 --------------------------
if __name__ == "__main__":
    main()
