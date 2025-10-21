import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go  # 导入plotly实现交互式3D图

st.set_page_config(page_title="SCIS Navigation System")

# -------------------------- 1. 基础配置 --------------------------
# 颜色常量定义（保持原有配色）
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral'},
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},
    'corridor_line': {'A': 'cyan', 'B': 'forestgreen', 'C': 'salmon'},
    'corridor_node': 'navy',
    'corridor_label': 'darkblue',
    'stair': {
        'Stairs1': '#FF5733',
        'Stairs2': '#33FF57',
        'Stairs3': '#3357FF',
        'Stairs4': '#FF33F5',
        'Stairs5': '#F5FF33',
        'StairsB1': '#33FFF5',
        'StairsB2': '#FF9933',
    },
    'stair_label': 'darkred',
    'classroom_label': 'black',
    'path': 'red',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold',
    'building_label': {'A': 'darkblue', 'B': 'darkgreen', 'C': 'darkred'}
}

# -------------------------- 2. 核心函数实现 --------------------------
# 读取JSON数据（保持原有逻辑）
def load_school_data_detailed(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data file: {str(e)}")
        return None

# 绘制交互式3D地图（plotly版本，支持放大缩小）
def plot_3d_map(school_data, display_options=None):
    # 初始化plotly图形对象
    fig = go.Figure()

    # 默认显示配置（保持原有逻辑）
    if display_options is None:
        display_options = {
            'start_level': None,
            'end_level': None,
            'path_stairs': set(),
            'show_all': True,
            'path': [],
            'start_building': None,
            'end_building': None
        }
    
    show_all = display_options['show_all']
    start_level = display_options['start_level']
    end_level = display_options['end_level']
    path_stairs = display_options['path_stairs']
    path = display_options.get('path', [])
    start_building = display_options.get('start_building')
    end_building = display_options.get('end_building')

    # 存储建筑物标签位置
    building_label_positions = {}

    # 遍历所有建筑物（A/B/C）
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        building_data = school_data[building_id]
        
        displayed_levels = []  # 已显示的楼层
        max_displayed_z = -float('inf')  # 最高显示楼层的Z值
        max_displayed_y = -float('inf')  # 最大Y坐标
        corresponding_x = 0  # 对应X坐标
        level_count = 0  # 楼层计数
        
        # 遍历建筑物的每一层
        for level in building_data['levels']:
            level_name = level['name']
            z = level['z']
            
            # 判断当前楼层是否需要显示（保持原有逻辑）
            show_level = show_all
            if not show_all:
                if building_name == 'B':
                    show_level = any((building_name, s_name, level_name) in path_stairs for s_name in ['StairsB1', 'StairsB2'])
                    if (start_building == 'B' or end_building == 'B') or (start_building in ['A','C'] and end_building in ['A','C'] and 'B' in [start_building, end_building]):
                        show_level = show_level or (level_name == 'level1')
                else:
                    show_level = (level_name == start_level) or (level_name == end_level)
            
            # 记录显示楼层的信息
            if show_level:
                displayed_levels.append(level)
                if z > max_displayed_z:
                    max_displayed_z = z
                
                fp = level['floorPlane']
                current_max_y = fp['maxY']
                if current_max_y > max_displayed_y:
                    max_displayed_y = current_max_y
                    corresponding_x = (fp['minX'] + fp['maxX']) / 2
            
            level_count += 1
            
            # 获取当前楼层的颜色配置
            floor_border_color = COLORS['floor_z'].get(z, 'gray')
            building_fill_color = COLORS['building'].get(building_name, 'lightgray')

            # 绘制楼层平面（仅显示需要展示的楼层）
            if show_level:
                fp = level['floorPlane']
                # 楼层平面顶点坐标
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
                
                # 添加楼层边框（plotly Scatter3d）
                fig.add_trace(go.Scatter3d(
                    x=x_plane, y=y_plane, z=z_plane,
                    mode='lines',
                    line=dict(color=floor_border_color, width=4),
                    name=f"Building {building_name}-{level_name}",
                    showlegend=True
                ))
                
                # 添加楼层填充面（plotly Mesh3d）
                fig.add_trace(go.Mesh3d(
                    x=x_plane[:-1], y=y_plane[:-1], z=z_plane[:-1],
                    color=building_fill_color,
                    opacity=0.3,
                    showlegend=False
                ))

                # 绘制走廊
                for corr_idx, corridor in enumerate(level['corridors']):
                    points = corridor['points']
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    z_coords = [p[2] for p in points]
                    
                    # 判断走廊类型（外部/连接/普通）
                    is_external = corridor.get('type') == 'external'
                    if is_external:
                        ext_style = corridor.get('style', {})
                        corr_line_color = ext_style.get('color', 'gray')
                        corr_line_style = ext_style.get('lineType', 'dash')  # plotly用'dash'表示虚线
                        corr_line_width = 10
                        corr_label = f"External Corridor ({building_name}-{corridor.get('name', f'corr{corr_idx}')})"
                    
                    elif 'name' in corridor and ('connectToBuilding' in corridor['name']):
                        corr_line_color = COLORS['connect_corridor']
                        corr_line_style = 'solid'
                        corr_line_width = 12
                        corr_label = f"Connecting Corridor ({building_name}-{level_name})"
                    
                    else:
                        corr_line_color = COLORS['corridor_line'].get(building_name, 'gray')
                        corr_line_style = 'solid'
                        corr_line_width = 8
                        corr_label = None
                    
                    # 添加走廊线
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z_coords,
                        mode='lines',
                        line=dict(
                            color=corr_line_color,
                            width=corr_line_width,
                            dash=corr_line_style
                        ),
                        name=corr_label if corr_label else None,
                        showlegend=bool(corr_label)
                    ))
                    
                    # 绘制走廊节点
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z_coords,
                        mode='markers',
                        marker=dict(
                            color=COLORS['corridor_node'],
                            size=3,
                            symbol='square'
                        ),
                        showlegend=False
                    ))

                # 绘制教室
                for classroom in level['classrooms']:
                    x, y, _ = classroom['coordinates']
                    width, depth = classroom['size']
                    class_name = classroom['name']

                    # 教室标签
                    fig.add_annotation(
                        x=x, y=y, z=z,
                        text=class_name,
                        showarrow=False,
                        font=dict(
                            size=14,
                            color=COLORS['classroom_label'],
                            weight='bold'
                        )
                    )
                    
                    # 教室位置标记
                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z],
                        mode='markers',
                        marker=dict(
                            color=building_fill_color,
                            size=5,
                            symbol='circle',
                            line=dict(width=2, color=floor_border_color)
                        ),
                        showlegend=False
                    ))
                    
                    # 教室边框
                    class_border_x = [x, x + width, x + width, x, x]
                    class_border_y = [y, y, y + depth, y + depth, y]
                    class_border_z = [z, z, z, z, z]
                    fig.add_trace(go.Scatter3d(
                        x=class_border_x, y=class_border_y, z=class_border_z,
                        mode='lines',
                        line=dict(color=floor_border_color, width=2, dash='dash'),
                        showlegend=False
                    ))

            # 绘制楼梯
            for stair in level['stairs']:
                stair_name = stair['name']
                is_path_stair = (building_name, stair_name, level_name) in path_stairs
                
                if show_all or show_level or is_path_stair:
                    x, y, _ = stair['coordinates']
                    stair_label = f"Building {building_name}-{stair_name}"
                    stair_color = COLORS['stair'].get(stair_name, 'red')
                    marker_size = 8 if is_path_stair else 6  # plotly标记尺寸
                    marker_edge_width = 2 if is_path_stair else 1
                    
                    # 楼梯标记
                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z],
                        mode='markers',
                        marker=dict(
                            color=stair_color,
                            size=marker_size,
                            symbol='triangle-up',  # 三角形标记（表示楼梯）
                            line=dict(width=marker_edge_width, color='black')
                        ),
                        name=stair_label,
                        showlegend=True
                    ))
                    
                    # 楼梯标签
                    fig.add_annotation(
                        x=x, y=y, z=z,
                        text=stair_name,
                        showarrow=False,
                        font=dict(
                            size=14,
                            color=COLORS['stair_label'],
                            weight='bold'
                        )
                    )
        
        # 记录建筑物标签位置
        if level_count > 0 and len(displayed_levels) > 0:
            if building_name == 'B':
                label_y = max_displayed_y - 2.0
            else:
                label_y = max_displayed_y + 2.0
            label_z = max_displayed_z + 1.0
            center_x = corresponding_x
            building_label_positions[building_name] = (center_x, label_y, label_z)

    # 添加建筑物标签
    for building_name, (x, y, z) in building_label_positions.items():
        fig.add_annotation(
            x=x, y=y, z=z,
            text=f"Building {building_name}",
            showarrow=False,
            font=dict(
                size=30,
                color=COLORS['building_label'].get(building_name, 'black'),
                weight='bold'
            ),
            bgcolor=COLORS['building'].get(building_name, 'lightgray'),
            opacity=0.7,
            bordercolor='black',
            borderwidth=2,
            pad=10
        )

    # 绘制导航路径（仅当有路径且不显示全部时）
    if path and not show_all:
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

            # 路径线
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                line=dict(color=COLORS['path'], width=10),
                marker=dict(size=5, color=COLORS['path']),
                name='Navigation Path',
                showlegend=True
            ))
            
            # 起点标记
            fig.add_trace(go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode='markers',
                marker=dict(
                    color=COLORS['start_marker'],
                    size=12,
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name='Start',
                showlegend=True
            ))
            
            # 终点标记
            fig.add_trace(go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]],
                mode='markers',
                marker=dict(
                    color=COLORS['end_marker'],
                    size=12,
                    symbol='star',
                    line=dict(width=2, color='black')
                ),
                name='End',
                showlegend=True
            ))
            
            # 起点标签
            fig.add_annotation(
                x=x[0], y=y[0], z=z[0],
                text=f"Start\n{labels[0]}",
                showarrow=False,
                font=dict(
                    size=16,
                    color=COLORS['start_label'],
                    weight='bold'
                )
            )
            
            # 终点标签
            fig.add_annotation(
                x=x[-1], y=y[-1], z=z[-1],
                text=f"End\n{labels[-1]}",
                showarrow=False,
                font=dict(
                    size=16,
                    color=COLORS['end_label'],
                    weight='bold'
                )
            )
        except Exception as e:
            st.warning(f"Path drawing warning: {str(e)}")

    # 设置3D图布局（关键：支持交互）
    fig.update_layout(
        # 坐标轴配置
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Floor Height (Z Value)',
            # 坐标轴标签样式
            xaxis=dict(
                title_font=dict(size=18, weight='bold'),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title_font=dict(size=18, weight='bold'),
                tickfont=dict(size=14)
            ),
            zaxis=dict(
                title_font=dict(size=18, weight='bold'),
                tickfont=dict(size=14)
            ),
            # 初始视角（可调整）
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            ),
            # 网格线显示
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            zaxis_showgrid=True,
            gridwidth=2,
            gridcolor='rgba(0,0,0,0.3)'
        ),
        # 标题配置
        title=dict(
            text='Campus 3D Navigation Map (A/B/C Building Navigation)',
            font=dict(size=24, weight='bold'),
            y=0.95,
            x=0.5
        ),
        # 图例配置
        legend=dict(
            x=1.05,
            y=1,
            font=dict(size=16),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        # 图尺寸（可根据需求调整）
        width=1400,
        height=900,
        # 边距配置
        margin=dict(l=0, r=0, b=0, t=50),
        # 交互模式（默认支持放大缩小）
        modebar=dict(
            orientation='vertical',
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )

    return fig

# 自定义图数据结构（保持原有逻辑）
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}

    def add_node(self, building_id, node_type, name, level, coordinates):
        building_name = building_id.replace('building', '')
        
        if node_type == 'corridor':
            node_id = f"{building_name}-corr-{name}@{level}"
        else:
            node_id = f"{building_name}-{node_type}-{name}@{level}"
        
        self.nodes[node_id] = {
            'building': building_name,
            'type': node_type,
            'name': name,
            'level': level,
            'coordinates': coordinates,
            'neighbors': {}
        }
        
        map_key = (building_id, node_type, name, level)
        self.node_id_map[map_key] = node_id
        if node_type == 'classroom':
            class_key = (building_name, name, level)
            self.node_id_map[class_key] = node_id
            
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

# 计算欧几里得距离（保持原有逻辑）
def euclidean_distance(coords1, coords2):
    return np.sqrt(sum((a - b) **2 for a, b in zip(coords1, coords2)))

# 构建导航图（保持原有逻辑）
def build_navigation_graph(school_data):
    graph = Graph()

    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        building_data = school_data[building_id]
        
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

    # 添加节点连接关系
    for building_id in school_data.keys():
        if not building_id.startswith('building'):
            continue
        building_name = building_id.replace('building', '')
        
        building_data = school_data[building_id]

        for level in building_data['levels']:
            level_name = level['name']
            
            # 获取当前楼层的所有走廊节点
            corr_nodes = [
                node_id for node_id, node_info in graph.nodes.items()
                if node_info['building'] == building_name 
                and node_info['type'] == 'corridor' 
                and node_info['level'] == level_name
            ]

            # 1. 同一走廊内节点连接
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

            # 2. 不同走廊间节点连接（近距离）
            for i in range(len(corr_nodes)):
                node1_id = corr_nodes[i]
                coords1 = graph.nodes[node1_id]['coordinates']
                for j in range(i + 1, len(corr_nodes)):
                    node2_id = corr_nodes[j]
                    coords2 = graph.nodes[node2_id]['coordinates']
                    distance = euclidean_distance(coords1, coords2)
                    
                    if distance < 3.0:
                        graph.add_edge(node1_id, node2_id, distance)

            # 3. 教室与最近走廊节点连接
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
                else:
                    st.warning(f"Warning: Classroom {graph.nodes[class_node_id]['name']} in Building {building_name}{level_name} has no corridor connection")

            # 4. 楼梯与最近走廊节点连接
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

        # 5. 同一建筑物不同楼层节点连接（楼梯）
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

    # 6. 建筑物间节点连接（A-B、B-C、A-C）
    a_building_id = 'buildingA'
    b_building_id = 'buildingB'
    c_building_id = 'buildingC'
    
    # A-B连接（level1）
    ab_connect_level = 'level1'
    a_b_corr_name = 'connectToBuildingB-p1'
    a_b_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_b_corr_name, ab_connect_level))
    b_a_corr_name = 'connectToBuildingAAndC-p1'
    b_a_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_a_corr_name, ab_connect_level))
    
    if a_b_node_id and b_a_node_id:
        coords_a = graph.nodes[a_b_node_id]['coordinates']
        coords_b = graph.nodes[b_a_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_b)
        graph.add_edge(a_b_node_id, b_a_node_id, distance)
    else:
        st.warning("Could not find A-B level1 inter-building corridor connection nodes")
    
    # B-C连接（level1）
    bc_connect_level = 'level1'
    b_c_corr_name = 'connectToBuildingAAndC-p0'
    b_c_node_id = graph.node_id_map.get((b_building_id, 'corridor', b_c_corr_name, bc_connect_level))
    c_b_corr_name = 'connectToBuildingB-p1'
    c_b_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_b_corr_name, bc_connect_level))
    
    if b_c_node_id and c_b_node_id:
        coords_b = graph.nodes[b_c_node_id]['coordinates']
        coords_c = graph.nodes[c_b_node_id]['coordinates']
        distance = euclidean_distance(coords_b, coords_c)
        graph.add_edge(b_c_node_id, c_b_node_id, distance)
    else:
        st.warning("Could not find B-C level1 inter-building corridor connection nodes")
    
    # A-C连接（level1）
    connect_level1 = 'level1'
    a_corr1_name = 'connectToBuildingC-p3'
    a_connect1_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr1_name, connect_level1))
    c_corr1_name = 'connectToBuildingA-p0'
    c_connect1_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr1_name, connect_level1))
    
    if a_connect1_node_id and c_connect1_node_id:
        coords_a = graph.nodes[a_connect1_node_id]['coordinates']
        coords_c = graph.nodes[c_connect1_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_c)
        graph.add_edge(a_connect1_node_id, c_connect1_node_id, distance)
    else:
        st.warning("Could not find level 1 A-C inter-building corridor connection nodes")
    
    # A-C连接（level3）
    connect_level3 = 'level3'
    a_corr3_name = 'connectToBuildingC-p2'
    a_connect3_node_id = graph.node_id_map.get((a_building_id, 'corridor', a_corr3_name, connect_level3))
    c_corr3_name = 'connectToBuildingA-p0'
    c_connect3_node_id = graph.node_id_map.get((c_building_id, 'corridor', c_corr3_name, connect_level3))
    
    if a_connect3_node_id and c_connect3_node_id:
        coords_a = graph.nodes[a_connect3_node_id]['coordinates']
        coords_c = graph.nodes[c_connect3_node_id]['coordinates']
        distance = euclidean_distance(coords_a, coords_c)
        graph.add_edge(a_connect3_node_id, c_connect3_node_id, distance)
    else:
        st.warning("Could not find level 3 A-C inter-building corridor connection nodes")

    return graph

# Dijkstra算法（保持原有逻辑）
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

# 生成导航路径（保持原有逻辑）
def construct_path(previous_nodes, end_node):
    path = []
    current_node = end_node
    while current_node is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    return path if len(path) > 1 else None

# 导航函数（保持原有逻辑）
def navigate(graph, start_building, start_classroom, start_level, end_building, end_classroom, end_level):
    valid_buildings = ['A', 'B', 'C']
    if start_building not in valid_buildings or end_building not in valid_buildings:
        return None, "Invalid building selection, only Buildings A, B and C are supported", None, None
        
    try:
        start_key = (start_building, start_classroom, start_level)
        end_key = (end_building, end_classroom, end_level)
        
        start_node = graph.node_id_map.get(start_key)
        end_node = graph.node_id_map.get(end_key)
        
        if not start_node:
            start_node = f"{start_building}-classroom-{start_classroom}@{start_level}"
        if not end_node:
            end_node = f"{end_building}-classroom-{end_classroom}@{end_level}"

        if start_node not in graph.nodes:
            return None, f"Starting classroom does not exist: {start_building}{start_classroom}@{start_level}", None, None
        if end_node not in graph.nodes:
            return None, f"Destination classroom does not exist: {end_building}{end_classroom}@{end_level}", None, None

        distances, previous_nodes = dijkstra(graph, start_node)
        path = construct_path(previous_nodes, end_node)

        if path:
            total_distance = distances[end_node]
            simplified_path = []
            path_stairs = set()
            prev_building = None
            
            for node_id in path:
                node_type = graph.nodes[node_id]['type']
                node_name = graph.nodes[node_id]['name']
                node_level = graph.nodes[node_id]['level']
                node_building = graph.nodes[node_id]['building']
                
                if node_type == 'stair':
                    path_stairs.add((node_building, node_name, node_level))
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
                
                elif node_type == 'classroom':
                    simplified_path.append(f"Building {node_building}{node_name}({node_level})")
                
                elif node_type == 'corridor':
                    if 'connectToBuilding' in node_name:
                        if 'connectToBuildingA' in node_name:
                            connected_building = 'A'
                        elif 'connectToBuildingB' in node_name:
                            connected_building = 'B'
                        elif 'connectToBuildingC' in node_name:
                            connected_building = 'C'
                        else:
                            connected_building = 'Other'
                            
                        if prev_building and prev_building != node_building:
                            simplified_path.append(f"Cross corridor from Building {prev_building} to Building {node_building}({node_level})")
                
                if node_type in ['classroom', 'stair', 'corridor']:
                    prev_building = node_building
            
            full_path_str = " → ".join(simplified_path)
            display_options = {
                'start_level': start_level,
                'end_level': end_level,
                'path_stairs': path_stairs,
                'show_all': False,
                'path': path,
                'start_building': start_building,
                'end_building': end_building
            }
            return path, f"Total distance: {total_distance:.2f} units", full_path_str, display_options
        else:
            return None, "No available path between the two classrooms", None, None
    except Exception as e:
        return None, f"Navigation error: {str(e)}", None, None

# 获取教室信息（保持原有逻辑）
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
        st.error(f"Failed to retrieve classroom information: {str(e)}")
        return [], {}, {}

# 重置应用状态（保持原有逻辑）
def reset_app_state():
    st.session_state['display_options'] = {
        'start_level': None,
        'end_level': None,
        'path_stairs': set(),
        'show_all': True,
        'path': [],
        'start_building': None,
        'end_building': None
    }
    st.session_state['current_path'] = None
    if 'path_result' in st.session_state:
        del st.session_state['path_result']

# -------------------------- 3. Streamlit界面逻辑 --------------------------
def main():
    # 页面样式配置（保持原有头部紧凑和右下角作者标记）
    st.markdown("""
        <style>
            /* 头部紧凑显示 */
            .stApp {
                padding-top: 0.0rem !important; 
            }
            header {
                display: none !important;
            }
            /* 作者标记（右下角灰色无框） */
            body {
                position: relative;
                min-height: 100vh;
                margin: 0;
                padding: 0;
            }
            .block-container {
                padding-top: 0.2rem !important;
                padding-left: 1rem;
                padding-right: 1rem;
                max-width: 100%;
                padding-bottom: 80px; 
            }
            .author-tag {
                position: fixed; 
                bottom: 50px;  
                right: 60px;    
                font-size: 16px;
                font-weight: bold;
                color: #666;    
                background: transparent;
                padding: 6px 12px;
                border: none;
                border-radius: 0;
                z-index: 9999;  
            }
        </style>
    """, unsafe_allow_html=True)
    
    # 作者标记
    st.markdown('<div class="author-tag">Created By DANIEL HAN</div>', unsafe_allow_html=True)
    
    # 页面标题
    st.subheader("🏫SCIS Campus Navigation System")
    st.markdown("3D Map & Inter-building Path Planning (A/B/C Building Navigation)")

    # 初始化会话状态
    if 'display_options' not in st.session_state:
        st.session_state['display_options'] = {
            'start_level': None,
            'end_level': None,
            'path_stairs': set(),
            'show_all': True,
            'path': [],
            'start_building': None,
            'end_building': None
        }
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = None

    # 加载数据
    try:
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data is None:
            return
            
        global graph
        graph = build_navigation_graph(school_data)
        building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)
        st.success("✅ Campus data loaded successfully! Initial state shows A/B/C buildings")
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return

    # 界面布局（左侧选择区，右侧地图区）
    col1, col2 = st.columns([1, 6])

    with col1:
        st.markdown("#### 📍 Select Locations")
        
        # 起点选择
        st.markdown("#### Start Point")
        start_building = st.selectbox("Building", building_names, key="start_building")
        start_levels = levels_by_building.get(start_building, [])
        start_level = st.selectbox("Floor", start_levels, key="start_level")
        start_classrooms = classrooms_by_building.get(start_building, {}).get(start_level, [])
        start_classroom = st.selectbox("Classroom", start_classrooms, key="start_classroom")

        # 终点选择
        st.markdown("#### End Point")
        end_building = st.selectbox("Building", building_names, key="end_building")
        end_levels = levels_by_building.get(end_building, [])
        end_level = st.selectbox("Floor", end_levels, key="end_level")
        end_classrooms = classrooms_by_building.get(end_building, {}).get(end_level, [])
        end_classroom = st.selectbox("Classroom", end_classrooms, key="end_classroom")

        # 功能按钮
        nav_button = st.button("🔍 Find Shortest Path", use_container_width=True)
        reset_button = st.button(
            "🔄 Reset View", 
            use_container_width=True,
            help="Click to return to initial state, showing all floors (including Building B) and clearing path"
        )
        
        # 重置按钮逻辑
        if reset_button:
            reset_app_state()
            st.rerun()

    with col2:
        st.markdown("#### 🗺️ 3D Campus Map")
        
        # 导航按钮逻辑
        if nav_button:
            try:
                path, message, simplified_path, display_options = navigate(
                    graph, 
                    start_building, start_classroom, start_level,
                    end_building, end_classroom, end_level
                )
                
                if path and display_options:
                    st.success(f"📊 Navigation result: {message}")
                    st.markdown("##### 🛤️ Path Details")
                    st.info(simplified_path)
                    
                    st.session_state['current_path'] = path
                    st.session_state['display_options'] = display_options
                else:
                    st.error(f"❌ {message}")
            except Exception as e:
                st.error(f"Navigation process error: {str(e)}")
        
        # 显示3D地图（plotly交互式图）
        try:
            if st.session_state['current_path'] is not None:
                fig = plot_3d_map(school_data, st.session_state['display_options'])
            else:
                fig = plot_3d_map(school_data)
            
            # 用Streamlit显示plotly图（支持交互）
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to display map: {str(e)}")

if __name__ == "__main__":
    main()
