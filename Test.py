import streamlit as st
import json
import plotly.graph_objects as go
import networkx as nx
from plotly.subplots import make_subplots

# 加载 JSON 数据
with open("floor_plans.json", "r") as f:
    floor_data = json.load(f)

# 页面标题
st.title("校园建筑 A 导航系统")

# 选择楼层
building = "Building A"
floors = list(floor_data[building].keys())
selected_floor = st.selectbox("选择楼层", floors)

# 获取当前楼层房间和连接数据
floor = floor_data[building][selected_floor]
rooms = floor["rooms"]
connections = floor["connections"]

# 创建 3D 散点图展示房间位置
fig_3d = go.Figure()

# 添加房间散点
for room in rooms:
    fig_3d.add_trace(
        go.Scatter3d(
            x=[room["coords"][0]],
            y=[room["coords"][1]],
            z=[0],  # 楼层 z 坐标设为 0，若有多楼层可区分
            mode="markers+text",
            marker=dict(size=10, color="blue"),
            text=room["name"],
            textposition="top center"
        )
    )

# 添加走廊连接
for conn in connections:
    from_room = next((r for r in rooms if r["name"] == conn["from"]), None)
    to_room = next((r for r in rooms if r["name"] == conn["to"]), None)
    if from_room and to_room:
        fig_3d.add_trace(
            go.Scatter3d(
                x=[from_room["coords"][0], to_room["coords"][0]],
                y=[from_room["coords"][1], to_room["coords"][1]],
                z=[0, 0],
                mode="lines",
                line=dict(color="gray", width=2)
            )
        )

# 设置 3D 图布局
fig_3d.update_layout(
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="cube"
    ),
    title=f"{building} - {selected_floor} 3D 布局"
)

# 显示 3D 图
st.plotly_chart(fig_3d)

# 导航功能：选择起点和终点房间
room_names = [room["name"] for room in rooms]
start_room = st.selectbox("起点房间", room_names)
end_room = st.selectbox("终点房间", room_names)

# 构建图结构用于寻路
G = nx.Graph()
for room in rooms:
    G.add_node(room["name"], pos=room["coords"])
for conn in connections:
    G.add_edge(conn["from"], conn["to"])

# 计算最短路径
if start_room != end_room and nx.has_path(G, start_room, end_room):
    path = nx.shortest_path(G, start_room, end_room)
    st.subheader("导航路径")
    st.write(" -> ".join(path))

    # 可视化导航路径
    fig_path = go.Figure()
    # 绘制所有房间
    for room in rooms:
        fig_path.add_trace(
            go.Scatter(
                x=[room["coords"][0]],
                y=[room["coords"][1]],
                mode="markers+text",
                marker=dict(size=10, color="blue"),
                text=room["name"],
                textposition="top center"
            )
        )
    # 绘制所有走廊
    for conn in connections:
        from_room = next((r for r in rooms if r["name"] == conn["from"]), None)
        to_room = next((r for r in rooms if r["name"] == conn["to"]), None)
        if from_room and to_room:
            fig_path.add_trace(
                go.Scatter(
                    x=[from_room["coords"][0], to_room["coords"][0]],
                    y=[from_room["coords"][1], to_room["coords"][1]],
                    mode="lines",
                    line=dict(color="gray", width=2)
                )
            )
    # 绘制导航路径
    path_coords = [next((r["coords"] for r in rooms if r["name"] == n), None) for n in path]
    path_x = [coord[0] for coord in path_coords if coord]
    path_y = [coord[1] for coord in path_coords if coord]
    fig_path.add_trace(
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(size=12, color="red")
        )
    )
    # 设置路径图布局
    fig_path.update_layout(
        xaxis_title="X",
        yaxis_title="Y",
        title=f"从 {start_room} 到 {end_room} 的导航路径"
    )
    st.plotly_chart(fig_path)
elif start_room == end_room:
    st.info("起点和终点是同一房间，无需导航～")
else:
    st.warning("所选房间之间无有效路径！")
