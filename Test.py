import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st

# -------------------------- 1. 基础配置 --------------------------
plt.switch_backend('Agg')  # 解决Streamlit matplotlib渲染问题

# 定义颜色常量
COLORS = {
    'building': {'A': 'lightblue', 'C': 'lightcoral'},
    'floor_z': {-6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},
    'corridor_line': {'A': 'cyan', 'C': 'salmon'},
    'corridor_node': 'navy',
    'corridor_label': 'darkblue',
    'stair': 'red',
    'stair_label': 'darkred',
    'classroom_label': 'black',
    'path': 'darkred',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold'
}

# -------------------------- 2. 核心功能实现 --------------------------
# 读取JSON数据
def load_school_data_detailed(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载数据文件失败: {str(e)}")
        return None

# 绘制3D地图（3D图放大两倍）
def plot_3d_map(school_data):
    # 图尺寸从(14,12)放大到(28,24)，整体占比扩大两倍
    fig = plt.figure(figsize=(35, 30))
    ax = fig.add_subplot(111, projection='3d')

    # 放大坐标轴刻度标签
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)

    # 遍历所有建筑
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        building_data = school_data[building_id]
        
        # 处理建筑内每个楼层
        for level in building_data['levels']:
            z = level['z']
            level_name = level['name']
            floor_border_color = COLORS['floor_z'].get(z, 'gray')
            building_fill_color = COLORS['building'][building_name]

            # 绘制楼层平面（加粗边框）
            fp = level['floorPlane']
            plane_vertices = [
                [fp['minX'], fp['minY'], z],
                [fp['maxX'], fp['minY'], z],
                [fp['maxX'], fp['maxY'], z],
                [fp['minX'], fp['maxY'], z],
                [fp['minX'], fp['minY'], z]
            ]
            x_plane = [p[0] for p in plane_vertices]
            y_plane = [p[1] for p in plane_vertices]
            z_plane = [p[2] for p in plane_vertices]
            
            # 边框线宽从2调整为4
            ax.plot(x_plane, y_plane, z_plane, color=floor_border_color, linewidth=4, 
                    label=f"{building_name}楼-{level_name}" if f"{building_name}楼-{level_name}" not in ax.get_legend_handles_labels()[1] else "")
            ax.plot_trisurf(x_plane[:-1], y_plane[:-1], z_plane[:-1], 
                            color=building_fill_color, alpha=0.3)

            # 绘制走廊（加粗线条）
            for corr_idx, corridor in enumerate(level['corridors']):
                points = corridor['points']
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                z_coords = [p[2] for p in points]
                
                if 'name' in corridor and ('connectToBuilding' in corridor['name']):
                    corr_line_color = COLORS['connect_corridor']
                    corr_line_width = 12  # 跨楼走廊线宽放大
                else:
                    corr_line_color = COLORS['corridor_line'][building_name]
                    corr_line_width = 8  # 普通走廊线宽放大
                    corr_label = None
                
                ax.plot(x, y, z_coords, color=corr_line_color, linewidth=corr_line_width, 
                        alpha=0.8, label=corr_label if (corr_label and corr_label not in ax.get_legend_handles_labels()[1]) else "")
                
                # 走廊节点尺寸放大
                for p_idx, (px, py, pz) in enumerate(points):
                    ax.scatter(px, py, pz, color=COLORS['corridor_node'], s=40, marker='s', alpha=0.9)
                    ax.text(px, py, pz, f'{building_name}C{corr_idx}-P{p_idx}', 
                            color=COLORS['corridor_label'], fontsize=12)

            # 绘制楼梯（尺寸放大）
            for stair in level['stairs']:
                x, y, _ = stair['coordinates']
                stair_label = f"{building_name}楼-{stair['name']}"
                if stair_label not in ax.get_legend_handles_labels()[1]:
                    ax.scatter(x, y, z, color=COLORS['stair'], s=600, marker='^', label=stair_label)
                else:
                    ax.scatter(x, y, z, color=COLORS['stair'], s=600, marker='^')
                ax.text(x, y, z, stair['name'], color=COLORS['stair_label'], fontweight='bold', fontsize=14)

            # 绘制教室（尺寸放大）
            for classroom in level['classrooms']:
                x, y, _ = classroom['coordinates']
                width, depth = classroom['size']
                class_name = classroom['name']

                ax.text(x, y, z, class_name, color=COLORS['classroom_label'], fontweight='bold', fontsize=14)
                ax.scatter(x, y, z, color=building_fill_color, s=160, edgecolors=floor_border_color)
                ax.plot([x, x + width, x + width, x, x],
                        [y, y, y + depth, y + depth, y],
                        [z, z, z, z, z],
                        color=floor_border_color, linestyle='--', alpha=0.6, linewidth=2)

    # 放大坐标轴标签和标题
    ax.set_xlabel('X 坐标', fontsize=18, fontweight='bold')
    ax.set_ylabel('Y 坐标', fontsize=18, fontweight='bold')
    ax.set_zlabel('楼层高度 (Z值)', fontsize=18, fontweight='bold')
    ax.set_title('校园3D导航地图（支持A/C楼跨楼导航）', fontsize=24, fontweight='bold', pad=20)
    
    # 放大图例
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16, frameon=True)
    ax.grid(True, alpha=0.3, linewidth=2)

    return fig, ax

# 自定义图数据结构
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}  # 多重映射关系

    def add_node(self, building_id, node_type, name, level, coordinates):
        building_name = building_id.replace('building', '')
        
        # 简洁节点ID
        if node_type == 'corridor':
            node_id = f"{building_name}-corr-{name}@{level}"
        else:
            node_id = f"{building_name}-{node_type}-{name}@{level}"
        
        # 存储节点信息
        self.nodes[node_id] = {
            'building': building_name,
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        
        # 建立多重映射
        map_key = (building_id, node_type, name, level)
        self.node_id_map[map_key] = node_id
        # 教室额外映射
        if node_type == 'classroom':
            class_key = (building_name, name, level)
            self.node_id_map[class_key] = node_id
            
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

# 计算欧氏距离
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b)** 2 for a, b in zip(coords1, coords2)))

# 构建导航图
def build_navigation_graph(school_data):
    graph = Graph()

    # 第一步：添加所有建筑的节点
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_data = school_data[building_id]
        building_name = building_id.replace('building', '')
        
        for level in building_data['levels']:
            level_name = level['name']

            # 添加教室节点
            for classroom in level['classrooms']:
                class_name = classroom['name']
                graph.add_node(
                    building_id=building_id,
                    node_type='classroom',
                    name=class_name,
                    level=level_name,
                    coordinates=classroom['coordinates']
                )

            # 添加楼梯节点
            for stair in level['stairs']:
                graph.add_node(
                    building_id=building_id,
                    node_type='stair',
                    name=stair['name'],
                    level=level_name,
                    coordinates=stair['coordinates']
                )

            # 添加走廊节点
            for corr_idx, corridor in enumerate(level['corridors']):
                corr_name = corridor.get('name', f'corr{corr_idx}')
                for p_idx, point in enumerate(corridor['points']):
                    corridor_point_name = f"{corr_name}-p{p_idx}"
                    graph.add_node(
                        building_id=building_id,
                        node_type='corridor',
                        name=corridor_point_name,
                        level=level_name,
                        coordinates=point
                    )

    # 第二步：添加所有连接关系
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_data = school_data[building_id]
        building_name = building_id.replace('building', '')

        for level in building_data['levels']:
            level_name = level['name']
            
            # 获取当前建筑当前楼层的所有走廊节点
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]

            # 1. 同一走廊内的节点连接
            for corr_idx, corridor in enumerate(level['corridors']):
                corr_name = corridor.get('name', f'corr{corr_idx}')
                corr_points = corridor['points']
                for p_idx in range(len(corr_points) - 1):
                    current_point_name = f"{corr_name}-p{p_idx}"
                    next_point_name = f"{corr_name}-p{p_idx + 1}"
                    current_node_id = graph.node_id_map.get((building_id, 'corridor', current_point_name, level_name))
                    next_node_id = graph.node_id_map.get((building_id, 'corridor', next_point_name, level_name))
                    
                    if current_node_id and next_node_id:
                        coords1 = graph.nodes[current_node_id]['coordinates']
                        coords2 = graph.nodes[next_node_id]['coordinates']
                        distance = euclidean_distance(coords1, coords2)
                        graph.add_edge(current_node_id, next_node_id, distance)

            # 2. 不同走廊间的节点连接
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2)
                    
                    if distance < 3.0:
                        graph.add_edge(node1_id, node2_id, distance)

            # 3. 教室 → 最近的走廊节点连接
            class_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'classroom' 
                and node_info['level'] == level_name
            ]
            for class_node_id in class_nodes:
                class_coords = graph.nodes[class_node_id]['coordinates']
                min_dist = float('inf')
                nearest_corr_node_id = None
                
                for corr_node_id in corr_nodes:
                    corr_coords = graph.nodes[corr_node_id]['coordinates']
                    dist = euclidean_distance(class_coords, corr_coords)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(class_node_id, nearest_corr_node_id, min_dist)

            # 4. 楼梯 → 最近的走廊节点连接
            stair_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'stair' 
                and node_info['level'] == level_name
            ]
            for stair_node_id in stair_nodes:
                stair_coords = graph.nodes[stair_node_id]['coordinates']
                min_dist = float('inf')
                nearest_corr_node_id = None
                
                for corr_node_id in corr_nodes:
                    corr_coords = graph.nodes[corr_node_id]['coordinates']
                    dist = euclidean_distance(stair_coords, corr_coords)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_corr_node_id = corr_node_id
                
                if nearest_corr_node_id:
                    graph.add_edge(stair_node_id, nearest_corr_node_id, min_dist)

        # 5. 建筑内跨楼层连接
        for connection in building_data['connections']:
            from_obj_name, from_level = connection['from']
            to_obj_name, to_level = connection['to']
            
            from_obj_type = 'stair' if from_obj_name.startswith('Stairs') else 'corridor'
            to_obj_type = 'stair' if to_obj_name.startswith('Stairs') else 'corridor'
            
            if from_obj_type == 'corridor':
                from_obj_name = f"{from_obj_name}-p0"
            if to_obj_type == 'corridor':
                to_obj_name = f"{to_obj_name}-p0"
            
            from_node_id = graph.node_id_map.get((building_id, from_obj_type, from_obj_name, from_level))
            to_node_id = graph.node_id_map.get((building_id, to_obj_type, to_obj_name, to_level))
            
            if from_node_id and to_node_id:
                graph.add_edge(from_node_id, to_node_id, 5.0)

    # 6. 跨建筑连接
    a_building_id = 'buildingA'
    c_building_id = 'buildingC'
    connect_level = 'level3'
    
    # A楼连接C楼的走廊节点
    a_corr_name = 'connectToBuildingC-p2'
    a_connect_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr_name, connect_level))
    
    # C楼连接A楼的走廊节点
    c_corr_name = 'connectToBuildingA-p0'
    c_connect_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr_name, connect_level))
    
    if a_connect_node_id and c_connect_node_id:
        coords_a = graph.nodes[a_connect_node_id]['coordinates']
        coords_c = graph.nodes[c_connect_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_c)
        graph.add_edge(a_connect_node_id, c_connect_node_id, distance)
    else:
        st.warning("跨楼走廊连接节点未找到，可能影响跨楼导航")

    return graph

# Dijkstra算法
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

# 生成最短路径
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path if len(path) > 1 else None

# 导航函数
def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    try:
        # 使用多重映射查找节点
        start_key = (start_building, start_classroom, start_level)
        end_key = (end_building, end_classroom, end_level)
        
        start_node = graph.node_id_map.get(start_key)
        end_node = graph.node_id_map.get(end_key)
        
        # 备选方案：构建节点ID
        if not start_node:
            start_node = f"{start_building}-classroom-{start_classroom}@{start_level}"
        if not end_node:
            end_node = f"{end_building}-classroom-{end_classroom}@{end_level}"

        # 验证节点是否存在
        if start_node not in graph.nodes:
            return None, f"起点教室不存在: {start_building}{start_classroom}@{start_level}", None
        if end_node not in graph.nodes:
            return None, f"终点教室不存在: {end_building}{end_classroom}@{end_level}", None

        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)

        if path:
            total_distance = distances[end_node]
            # 路径详情合并为一行
            simplified_path = []
            for node_id in path:
                node_type = graph.nodes[node_id]['type']
                node_name = graph.nodes[node_id]['name']
                node_level = graph.nodes[node_id]['level']
                node_building = graph.nodes[node_id]['building']
                if node_type in ['classroom', 'stair']:
                    simplified_path.append(f"{node_building}楼{node_name}({node_level})")
            
            # 用箭头连接所有步骤，形成单行显示
            full_path_str = " → ".join(simplified_path)
            return path, f"总距离: {total_distance:.2f} 单位", full_path_str
        else:
            return None, "两个教室之间没有可用路径", None
    except Exception as e:
        return None, f"导航错误: {str(e)}", None

# 在3D图上绘制路径（放大路径显示）
def plot_path(ax, graph, path):
    try:
        x = []
        y = []
        z = []
        labels = []

        for node_id in path:
            coords = graph.nodes[node_id]['coordinates']
            x.append(coords[0])
            y.append(coords[1])
            z.append(coords[2])
            
            node_type = graph.nodes[node_id]['type']
            if node_type == 'classroom':
                labels.append(graph.nodes[node_id]['name'])
            elif node_type == 'stair':
                labels.append(graph.nodes[node_id]['name'])
            else:
                labels.append("")

        # 路径线宽从3放大到6
        ax.plot(x, y, z, color=COLORS['path'], linewidth=6, linestyle='-', marker='o', markersize=10)
        # 起点终点标记放大
        ax.scatter(x[0], y[0], z[0], color=COLORS['start_marker'], s=1000, marker='*', label='起点', edgecolors='black')
        ax.scatter(x[-1], y[-1], z[-1], color=COLORS['end_marker'], s=1000, marker='*', label='终点', edgecolors='black')
        ax.text(x[0], y[0], z[0], f"起点\n{labels[0]}", color=COLORS['start_label'], fontweight='bold', fontsize=16)
        ax.text(x[-1], y[-1], z[-1], f"终点\n{labels[-1]}", color=COLORS['end_label'], fontweight='bold', fontsize=16)

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=16)
    except Exception as e:
        st.error(f"绘制路径失败: {str(e)}")

# 获取所有建筑、楼层和教室信息
def get_classroom_info(school_data):
    try:
        buildings = [b for b in school_data.keys() if b.startswith('building')]
        building_names = [b.replace('building', '') for b in buildings]
        
        classrooms_by_building = {}
        levels_by_building = {}
        
        for building_id in buildings:
            building_name = building_id.replace('building', '')
            building_data = school_data[building_id]
            
            levels = []
            classrooms_by_level = {}
            
            for level in building_data['levels']:
                level_name = level['name']
                levels.append(level_name)
                classrooms = [classroom['name'] for classroom in level['classrooms']]
                classrooms_by_level[level_name] = classrooms
            
            levels_by_building[building_name] = levels
            classrooms_by_building[building_name] = classrooms_by_level
            
        return building_names, levels_by_building, classrooms_by_building
    except Exception as e:
        st.error(f"获取教室信息失败: {str(e)}")
        return [], {}, {}

# -------------------------- 3. Streamlit界面逻辑 --------------------------
def main():
     # 新增：调整左右边距（此行是新增的第1行）
    st.markdown("""
        <style>
            .block-container {
                padding-left: 1rem;    /* 左侧边距减小 */
                padding-right: 1rem;   /* 右侧边距减小 */
                max-width: 100%;       /* 取消最大宽度限制 */
            }
        </style>
    """, unsafe_allow_html=True)  # 新增代码结束
    st.subheader("🏫 校园导航系统")
    st.markdown("3D地图与跨楼路径规划")

    # 加载JSON数据
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        nav_graph = build_navigation_graph(school_data)
        building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
        st.success("✅ 校园数据加载成功！")
    except Exception as e:
        st.error(f"初始化错误: {str(e)}")
        return

    # 布局调整：左侧互动界面占比1/3，右侧地图占比2/3（接近1:2）
    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### 📍 选择位置")
        
        # 起点选择
        st.markdown("#### 起点")
        start_building = st.selectbox("建筑", building_names, key="start_building")
        start_levels = levels_by_building.get(start_building, [])
        start_level = st.selectbox("楼层", start_levels, key="start_level")
        start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
        start_classroom = st.selectbox("教室", start_classrooms, key="start_classroom")

        # 终点选择
        st.markdown("#### 终点")
        end_building = st.selectbox("建筑", building_names, key="end_building")
        end_levels = levels_by_building.get(end_building, [])
        end_level = st.selectbox("楼层", end_levels, key="end_level")
        end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
        end_classroom = st.selectbox("教室", end_classrooms, key="end_classroom")

        # 导航按钮
        nav_button = st.button("🔍 查找最短路径", use_container_width=True)

    with col2:
        st.markdown("#### 🗺️ 3D校园地图")
        
        if 'fig' not in st.session_state:
            fig, ax = plot_3d_map(school_data)
            st.session_state['fig'] = fig
        
        if nav_button:
            try:
                path, message, simplified_path = navigate(
                    nav_graph, 
                    start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                
                if path:
                    st.success(f"📊 导航结果: {message}")
                    # 路径详情单行显示
                    st.markdown("##### 🛤️ 路径详情")
                    st.info(simplified_path)  # 使用info框突出显示单行路径
                    
                    fig, ax = plot_3d_map(school_data)
                    plot_path(ax, nav_graph, path)
                    st.session_state['fig'] = fig
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"导航过程出错: {str(e)}")
        
        try:
            st.pyplot(st.session_state['fig'])
        except Exception as e:
            st.error(f"显示地图失败: {str(e)}")

if __name__ == "__main__":
    main()
    





