import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import traceback
from plotly.graph_objs.layout.scene import Annotation as SceneAnnotation

# 移除了过时的配置项

st.set_page_config(page_title="SCIS Navigation System", layout="wide")

# 颜色配置常量
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral'},
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 4: 'teal', 7: 'orange', 12: 'purple'},
    'corridor_line': {'A': 'cyan', 'B': 'forestgreen', 'C': 'salmon'},
    'corridor_node': 'navy',
    'stair': {
        'Stairs1': '#FF5733', 'Stairs2': '#33FF57', 'Stairs3': '#3357FF',
        'Stairs4': '#FF33F5', 'Stairs5': '#F5FF33', 'StairsB1': '#33FFF5', 'StairsB2': '#FF9933'
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

# 读取校园数据（增加详细调试信息）
def load_school_data_detailed(filename):
    try:
        st.info(f"尝试加载数据文件: {filename}")  # 显示加载进度
        with open(filename, 'r') as f:
            data = json.load(f)
        st.success(f"数据文件加载成功，包含{len(data)}个建筑物")
        return data
    except FileNotFoundError:
        st.error(f"数据文件不存在: {filename}，请检查文件路径")
        return None
    except json.JSONDecodeError:
        st.error(f"数据文件格式错误，不是有效的JSON")
        return None
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        st.error(f"错误详情: {traceback.format_exc()}")
        return None

# 绘制3D地图（简化逻辑，确保基础绘图可用）
def plot_3d_map(school_data, graph, display_options=None):
    try:
        fig = go.Figure()
        st.info("开始绘制3D地图...")  # 显示绘图进度

        if display_options is None:
            display_options = {
                'start_level': None, 'end_level': None, 'path_stairs': set(),
                'show_all': True, 'path': [], 'start_building': None, 'end_building': None
            }
        
        show_all = display_options['show_all']
        building_label_positions = {}

        # 只绘制基础建筑物框架（简化逻辑，确保能显示基础图形）
        for building_id in school_data.keys():
            if not building_id.startswith('building'):
                continue
            building_name = building_id.replace('building', '')
            building_data = school_data[building_id]
            st.info(f"绘制建筑物: {building_name}")  # 显示当前绘制的建筑物

            # 绘制至少一个楼层（确保能看到图形）
            if building_data['levels']:
                level = building_data['levels'][0]  # 只取第一个楼层
                level_name = level['name']
                z = level['z']
                fp = level['floorPlane']

                # 绘制楼层平面（基础图形）
                plane_vertices = [
                    [fp['minX'], fp['minY'], z], [fp['maxX'], fp['minY'], z],
                    [fp['maxX'], fp['maxY'], z], [fp['minX'], fp['maxY'], z], [fp['minX'], fp['minY'], z]
                ]
                x_plane = [p[0] for p in plane_vertices]
                y_plane = [p[1] for p in plane_vertices]
                z_plane = [p[2] for p in plane_vertices]

                # 楼层边框
                fig.add_trace(go.Scatter3d(
                    x=x_plane, y=y_plane, z=z_plane,
                    mode='lines',
                    line=dict(color=COLORS['floor_z'].get(z, 'gray'), width=4),
                    name=f"Building {building_name}-{level_name}"
                ))

                # 楼层填充面
                fig.add_trace(go.Mesh3d(
                    x=x_plane[:-1], y=y_plane[:-1], z=z_plane[:-1],
                    color=COLORS['building'].get(building_name, 'lightgray'),
                    opacity=0.3
                ))

        # 基础布局配置（确保图形能正确渲染）
        fig.update_layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            title="校园3D地图",
            width=1000,
            height=600,
            modebar={'bgcolor': 'rgba(255,255,255,0.7)'}
        )

        st.success("3D地图绘制完成")
        return fig
    except Exception as e:
        st.error(f"绘图失败: {str(e)}")
        st.error(f"错误详情: {traceback.format_exc()}")
        return None

# 图数据结构（保持不变）
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}

    def add_node(self, building_id, node_type, name, level, coordinates):
        building_name = building_id.replace('building', '')
        node_id = f"{building_name}-{node_type}-{name}@{level}"
        self.nodes[node_id] = {
            'building': building_name, 'type': node_type, 'name': name,
            'level': level, 'coordinates': coordinates, 'neighbors': {}
        }
        return node_id

    def add_edge(self, node1_id, node2_id, weight):
        if node1_id in self.nodes and node2_id in self.nodes:
            self.nodes[node1_id]['neighbors'][node2_id] = weight
            self.nodes[node2_id]['neighbors'][node1_id] = weight

# 构建导航图（简化版，确保不影响基础绘图）
def build_navigation_graph(school_data):
    try:
        graph = Graph()
        st.info("开始构建导航图...")
        # 只构建基础节点，避免复杂逻辑出错
        for building_id in school_data.keys():
            if building_id.startswith('building'):
                building_data = school_data[building_id]
                for level in building_data['levels'][:1]:  # 只处理第一个楼层
                    for classroom in level['classrooms'][:1]:  # 只处理第一个教室
                        graph.add_node(building_id, 'classroom', classroom['name'], level['name'], classroom['coordinates'])
        st.success("导航图构建完成")
        return graph
    except Exception as e:
        st.error(f"构建导航图失败: {str(e)}")
        return Graph()  # 返回空图，避免程序中断

# 主函数（增加调试步骤）
def main():
    st.markdown("### 🏫 SCIS校园导航系统")
    st.markdown("#### 调试模式：显示所有加载和绘图步骤")

    # 1. 初始化会话状态
    if 'display_options' not in st.session_state:
        st.session_state['display_options'] = {'show_all': True}
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = None

    # 2. 加载数据（关键步骤，增加重试逻辑）
    school_data = None
    for _ in range(2):  # 尝试加载2次
        school_data = load_school_data_detailed('school_data_detailed.json')
        if school_data:
            break
    if not school_data:
        st.error("无法继续，数据加载失败")
        return

    # 3. 构建导航图
    graph = build_navigation_graph(school_data)

    # 4. 显示3D地图（使用简化版绘图函数）
    st.markdown("#### 🗺️ 3D校园地图")
    try:
        if st.session_state['current_path'] is not None:
            fig = plot_3d_map(school_data, graph, st.session_state['display_options'])
        else:
            fig = plot_3d_map(school_data, graph)
        
        if fig:
            # 强制使用iframe渲染模式（解决Streamlit Cloud兼容性问题）
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
        else:
            st.error("未能生成3D地图对象")
    except Exception as e:
        st.error(f"显示地图失败: {str(e)}")
        st.error(f"错误详情: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
