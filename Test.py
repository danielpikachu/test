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
    ax.set_title('School 3D Map (Constrained Path: Classroom→Corridor→Stairs→Corridor→Classroom)')
    ax.legend()

    return fig, ax

# 自定义图数据结构
class Graph:
    def __init__(self):
        self.nodes = {}  # key: node_id, value: node_info
        self.stair_proximity = {}  # 记录走廊节点与最近楼梯的距离

    def add_node(self, node_id, node_type, name, level, coordinates, stair_distance=None):
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

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图
def build_navigation_graph(school_data):
    graph = Graph()

    # 步骤1：添加所有节点（教室、楼梯、走廊）
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']
        
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

        # 1.3 添加走廊节点
        for corridor_idx, corridor in enumerate(level['corridors']):
            for point_idx, point in enumerate(corridor['points']):
                node_id = f"corridor_{point[0]}_{point[1]}_{z}"
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

    # 步骤2：添加边
    for level in school_data['buildingA']['levels']:
        level_name = level['name']
        z = level['z']

        # 2.1 教室 ↔ 走廊（只连接到走廊，不直接连楼梯）
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
            
            # 只连接最近的2个走廊，确保教室只能先到走廊
            for i, (node_id, distance) in enumerate(corridor_distances[:2]):
                weight = distance * 0.5  # 优先连接走廊
                graph.add_edge(classroom_node_id, node_id, weight)

        # 2.2 楼梯 ↔ 走廊（楼梯只连接走廊）
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
            
            for node_id, distance in corridor_distances:
                weight = distance * (0.3 if distance < 5 else 1.0)
                graph.add_edge(stair_node_id, node_id, weight)

        # 2.3 走廊 ↔ 走廊
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

# 带路径阶段约束的Dijkstra算法
def constrained_dijkstra(graph, start_node, end_node):
    # 确定起点和终点楼层
    start_level = graph.nodes[start_node]['level']
    end_level = graph.nodes[end_node]['level']
    need_stairs = start_level != end_level  # 是否需要跨楼层（经过楼梯）
    
    # 初始化距离和路径阶段跟踪
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_node] = 0
    previous_nodes = {node: None for node in graph.nodes}
    
    # 路径阶段跟踪：0=起点教室, 1=已到走廊, 2=已到楼梯, 3=目标楼层走廊, 4=已到终点
    path_phase = {node: 0 for node in graph.nodes}
    path_phase[start_node] = 0  # 起点是教室（阶段0）
    
    unvisited_nodes = set(graph.nodes.keys())

    while unvisited_nodes:
        current_node = min(unvisited_nodes, key=lambda x: distances[x])
        unvisited_nodes.remove(current_node)

        # 修复：先获取当前节点的属性再使用
        current_level = graph.nodes[current_node]['level']
        current_type = graph.nodes[current_node]['type']
        current_phase = path_phase[current_node]

        # 现在current_level已定义，可以安全使用
        if current_level == end_level and current_node == end_node:
            break
        if distances[current_node] == float('inf'):
            break

        for neighbor, weight in graph.nodes[current_node]['neighbors'].items():
            neighbor_type = graph.nodes[neighbor]['type']
            neighbor_level = graph.nodes[neighbor]['level']
            new_phase = current_phase
            
            # 阶段转换规则（强制路径顺序）
            valid_transition = False
            
            # 阶段0: 起点教室 -> 只能去走廊（阶段1）
            if current_phase == 0:
                if neighbor_type == 'corridor':
                    new_phase = 1
                    valid_transition = True
            
            # 阶段1: 走廊 -> 可以去其他走廊或楼梯（如果需要跨楼层）
            elif current_phase == 1:
                if neighbor_type == 'corridor':
                    new_phase = 1  # 继续在走廊
                    valid_transition = True
                elif neighbor_type == 'stair' and need_stairs:
                    new_phase = 2  # 到达楼梯
                    valid_transition = True
            
            # 阶段2: 楼梯 -> 可以去其他楼梯（跨楼层）或目标楼层走廊
            elif current_phase == 2:
                if neighbor_type == 'stair':
                    new_phase = 2  # 跨楼层楼梯
                    valid_transition = True
                elif neighbor_type == 'corridor' and neighbor_level == end_level:
                    new_phase = 3  # 到达目标楼层走廊
                    valid_transition = True
            
            # 阶段3: 目标楼层走廊 -> 可以去其他走廊或终点教室
            elif current_phase == 3:
                if neighbor_type == 'corridor':
                    new_phase = 3  # 继续在目标楼层走廊
                    valid_transition = True
                elif neighbor == end_node:  # 只能去终点教室
                    new_phase = 4  # 到达终点
                    valid_transition = True

            # 只有有效转换才允许更新路径
            if valid_transition:
                new_distance = distances[current_node] + weight
                
                if new_distance < distances[neighbor] or (new_distance == distances[neighbor] and new_phase > path_phase[neighbor]):
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
        current_node = previous_nodes[current_node]
    return path

# 验证路径是否符合规定顺序
def validate_path_order(graph, path):
    if len(path) < 2:
        return False, "路径太短"
    
    # 提取路径类型序列
    path_types = [graph.nodes[node]['type'] for node in path]
    start_level = graph.nodes[path[0]]['level']
    end_level = graph.nodes[path[-1]]['level']
    need_stairs = start_level != end_level
    
    # 检查起点和终点是否为教室
    if path_types[0] != 'classroom' or path_types[-1] != 'classroom':
        return False, "起点和终点必须是教室"
    
    # 检查是否先到走廊
    if path_types[1] != 'corridor':
        return False, "必须先从教室到走廊"
    
    # 检查跨楼层时是否经过楼梯
    if need_stairs:
        if 'stair' not in path_types:
            return False, "跨楼层路径必须经过楼梯"
        
        # 检查楼梯位置是否合理（在走廊之后，目标走廊之前）
        stair_indices = [i for i, t in enumerate(path_types) if t == 'stair']
        last_stair_index = stair_indices[-1]
        
        # 确保楼梯之后是目标楼层的走廊
        if last_stair_index >= len(path_types) - 2:
            return False, "楼梯之后必须连接目标楼层的走廊"
            
        if path_types[last_stair_index + 1] != 'corridor':
            return False, "楼梯之后必须是走廊"
    
    # 检查最后一步是否从走廊到教室
    if path_types[-2] != 'corridor':
        return False, "最后必须从走廊到教室"
    
    return True, "路径顺序有效"

# 导航函数
def navigate(graph, start_classroom, start_level, end_classroom, end_level):
    start_node = f"{start_classroom}@{start_level}"
    end_node = f"{end_classroom}@{end_level}"

    if start_node not in graph.nodes or end_node not in graph.nodes:
        return None, "❌ 无效的教室或楼层"
    if start_node == end_node:
        return [start_node], "✅ 起点和终点相同，无需移动"

    # 使用带约束的Dijkstra算法
    distances, previous_nodes = constrained_dijkstra(graph, start_node, end_node)
    path = construct_path(previous_nodes, end_node)

    # 验证路径顺序，如果不符合则强制修正
    is_valid, message = validate_path_order(graph, path)
    if not is_valid:
        st.warning(f"路径顺序调整: {message}")
        return force_valid_path(graph, start_node, end_node)
    
    if path:
        total_distance = distances[end_node]
        return path, f"✅ 路径规划成功！总距离：{total_distance:.2f} 单位"
    else:
        return None, "❌ 无有效路径"

# 强制生成符合顺序的路径
def force_valid_path(graph, start_node, end_node):
    start_level = graph.nodes[start_node]['level']
    end_level = graph.nodes[end_node]['level']
    need_stairs = start_level != end_level
    
    # 1. 找到起点教室到最近走廊的路径
    start_corridors = [n for n in graph.nodes[start_node]['neighbors'] 
                      if graph.nodes[n]['type'] == 'corridor']
    if not start_corridors:
        return None, "❌ 起点教室没有连接到任何走廊"
    nearest_start_corridor = min(start_corridors, 
                               key=lambda x: graph.nodes[start_node]['neighbors'][x])
    
    # 2. 找到终点教室到最近走廊的路径
    end_corridors = [n for n in graph.nodes[end_node]['neighbors'] 
                    if graph.nodes[n]['type'] == 'corridor']
    if not end_corridors:
        return None, "❌ 终点教室没有连接到任何走廊"
    nearest_end_corridor = min(end_corridors,
                             key=lambda x: graph.nodes[end_node]['neighbors'][x])
    
    # 3. 如果需要跨楼层，找到连接的楼梯
    stair_path = []
    if need_stairs:
        start_stairs = [n for n in graph.nodes if graph.nodes[n]['type'] == 'stair' 
                       and graph.nodes[n]['level'] == start_level]
        end_stairs = [n for n in graph.nodes if graph.nodes[n]['type'] == 'stair' 
                     and graph.nodes[n]['level'] == end_level]
        
        if not start_stairs or not end_stairs:
            return None, "❌ 缺少连接的楼梯"
        
        # 找到连接的楼梯对
        connected_stairs = []
        for s1 in start_stairs:
            for s2 in end_stairs:
                if s2 in graph.nodes[s1]['neighbors']:
                    connected_stairs.append((s1, s2))
        
        if not connected_stairs:
            return None, "❌ 楼层之间没有连接的楼梯"
        
        # 选择距离最近的楼梯对
        s1, s2 = min(connected_stairs, 
                    key=lambda x: euclidean_distance(
                        graph.nodes[x[0]]['coordinates'],
                        graph.nodes[nearest_start_corridor]['coordinates']
                    ) + euclidean_distance(
                        graph.nodes[x[1]]['coordinates'],
                        graph.nodes[nearest_end_corridor]['coordinates']
                    ))
        
        # 找到起点走廊到起点楼梯的路径
        dist1, prev1 = constrained_dijkstra(graph, nearest_start_corridor, s1)
        path1 = construct_path(prev1, s1)
        
        # 找到终点楼梯到终点走廊的路径
        dist2, prev2 = constrained_dijkstra(graph, s2, nearest_end_corridor)
        path2 = construct_path(prev2, nearest_end_corridor)
        
        stair_path = path1[1:] + [s2] + path2[1:]
    
    # 4. 如果不需要跨楼层，直接连接走廊
    else:
        dist, prev = constrained_dijkstra(graph, nearest_start_corridor, nearest_end_corridor)
        stair_path = construct_path(prev, nearest_end_corridor)[1:]
    
    # 组合完整路径
    full_path = [start_node, nearest_start_corridor] + stair_path + [end_node]
    
    # 去重
    seen = set()
    full_path = [node for node in full_path if not (node in seen or seen.add(node))]
    
    return full_path, "✅ 已生成符合顺序的路径"

# 在3D图上绘制路径
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

    # 标记特殊节点
    for i, (x, y, z, node_type) in enumerate(zip(x_coords, y_coords, z_coords, node_types)):
        if i == 0:  # 起点教室
            ax.scatter(x, y, z, color='green', s=300, marker='*', label='Start Classroom')
        elif i == len(path) - 1:  # 终点教室
            ax.scatter(x, y, z, color='purple', s=300, marker='*', label='End Classroom')
        elif node_type == 'stair':  # 楼梯
            ax.scatter(x, y, z, color='red', s=200, marker='^', label='Staircase')
        elif node_type == 'corridor':  # 走廊
            # 区分起点附近走廊和终点附近走廊
            if i == 1:
                ax.scatter(x, y, z, color='cyan', s=150, marker='o', label='Start Corridor')
            elif i == len(path) - 2:
                ax.scatter(x, y, z, color='orange', s=150, marker='o', label='End Corridor')
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
    st.subheader("强制路径顺序：教室→走廊→楼梯→走廊→教室")

    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        nav_graph = build_navigation_graph(school_data)
        levels, classrooms_by_level = get_classroom_info(school_data)
        st.success("✅ 校园数据加载成功！")
    except FileNotFoundError:
        st.error("❌ 错误：未找到'school_data_detailed.json'文件，请检查文件路径。")
        return
    except Exception as e:
        st.error(f"❌ 数据加载错误: {str(e)}")
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
                st.markdown("#### 🛤️ 路径详情（按顺序）")
                
                # 解析路径阶段并显示
                path_phases = []
                for i, node in enumerate(path):
                    node_type = nav_graph.nodes[node]['type']
                    if i == 0:
                        path_phases.append(f"{i+1}. 起点教室: {node.split('@')[0]}")
                    elif i == len(path)-1:
                        path_phases.append(f"{i+1}. 终点教室: {node.split('@')[0]}")
                    elif node_type == 'stair':
                        path_phases.append(f"{i+1}. 楼梯: {node.split('@')[0]}")
                    else:  # corridor
                        path_phases.append(f"{i+1}. 走廊")
                
                for phase in path_phases:
                    st.write(phase)
                
                fig, ax = plot_3d_map(school_data)
                plot_path(ax, nav_graph, path)
                st.session_state['fig'] = fig
            else:
                st.error(message)
        
        st.pyplot(st.session_state['fig'])

if __name__ == "__main__":
    main()
