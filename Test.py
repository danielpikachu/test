# app.py

import json, math
from math import dist
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="3D School Pathfinder")
st.title("3D School Pathfinder — Demo")

@st.cache_data
def load_default_map():
    with open("sample_map.json", "r") as f:
        return json.load(f)

# --------- load map JSON (builtin or uploaded) ----------
uploaded = st.sidebar.file_uploader("Upload map JSON", type=["json"])
if uploaded is not None:
    try:
        data = json.load(uploaded)
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")
        data = load_default_map()
else:
    data = load_default_map()

nodes = data.get("nodes", {})
edges = data.get("edges", [])
buildings = data.get("buildings", [])

# --------- build graph ----------
G = nx.Graph()
for name, coord in nodes.items():
    # require x,y; z optional
    x = float(coord.get("x", 0))
    y = float(coord.get("y", 0))
    z = float(coord.get("z", 0))
    G.add_node(name, x=x, y=y, z=z)

for e in edges:
    u = e["from"]; v = e["to"]
    if "weight" in e:
        w = float(e["weight"])
    else:
        a = nodes[u]; b = nodes[v]
        w = dist((a["x"], a["y"], a.get("z",0)), (b["x"], b["y"], b.get("z",0)))
    G.add_edge(u, v, weight=w)

# --------- UI controls ----------
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Route controls")
    if len(nodes) < 2:
        st.info("Map needs at least 2 nodes.")
    start = st.selectbox("Start", sorted(nodes.keys()))
    end = st.selectbox("Destination", sorted(nodes.keys()), index=(1 if len(nodes)>1 else 0))
    avoid_stairs = st.checkbox("Avoid stairs / floor changes (apply penalty)", value=False)
    find_btn = st.button("Find shortest path")

def heuristic(u, v):
    a = nodes[u]; b = nodes[v]
    return dist((a["x"], a["y"], a.get("z",0)), (b["x"], b["y"], b.get("z",0)))

path = None
path_length = None
if find_btn:
    H = G.copy()
    if avoid_stairs:
        for u,v,d in H.edges(data=True):
            if abs(nodes[u].get("z",0) - nodes[v].get("z",0)) > 0.1:
                d["weight"] = d["weight"] * 5.0
    try:
        path = nx.astar_path(H, start, end, heuristic=heuristic, weight="weight")
        path_length = sum(H[u][v]["weight"] for u,v in zip(path, path[1:]))
        st.success(f"Path found: {' → '.join(path)}  — length ≈ {path_length:.1f}")
    except nx.NetworkXNoPath:
        st.error("No path found.")

# --------- 3D visualization ----------
with col2:
    st.subheader("3D map preview")
    fig = go.Figure()

    # helper to build a cuboid mesh for a 4-point footprint (order: CCW)
    def cuboid_mesh_from_quad(footprint, height):
        # footprint: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        verts = []
        for z in (0, height):
            for (x,y) in footprint:
                verts.append((x,y,z))
        xs = [v[0] for v in verts]; ys = [v[1] for v in verts]; zs = [v[2] for v in verts]
        # simple triangulation for quads (bottom 0-3, top 4-7)
        i = [0,0,0,4,5,6,0,1,2,4,5,6]
        j = [1,2,3,5,6,7,4,5,6,0,1,2]
        k = [4,4,4,0,0,0,1,2,3,5,6,7]
        return xs, ys, zs, i, j, k

    # add building meshes
    for b in buildings:
        fp = b.get("footprint", [])
        h = float(b.get("height", 3))
        if len(fp) == 4:
            xs, ys, zs, i, j, k = cuboid_mesh_from_quad(fp, h)
            fig.add_trace(go.Mesh3d(
                x=xs, y=ys, z=zs, i=i, j=j, k=k,
                opacity=0.45, name=b.get("name","building"), hovertext=b.get("name","")
            ))
        else:
            # fallback: draw ground outline
            xs = [p[0] for p in fp] + ([fp[0][0]] if fp else [])
            ys = [p[1] for p in fp] + ([fp[0][1]] if fp else [])
            zs = [0]*len(xs)
            if xs:
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(width=4), opacity=0.4, showlegend=False))

    # draw nodes
    node_names = list(nodes.keys())
    nx_x = [nodes[n]["x"] for n in node_names]
    nx_y = [nodes[n]["y"] for n in node_names]
    nx_z = [nodes[n].get("z",0) for n in node_names]
    fig.add_trace(go.Scatter3d(x=nx_x, y=nx_y, z=nx_z, mode="markers+text",
                               marker=dict(size=4), text=node_names, textposition="top center", name="Nodes"))

    # draw edges
    for u,v in G.edges():
        x0,y0,z0 = nodes[u]["x"], nodes[u]["y"], nodes[u].get("z",0)
        x1,y1,z1 = nodes[v]["x"], nodes[v]["y"], nodes[v].get("z",0)
        fig.add_trace(go.Scatter3d(x=[x0,x1], y=[y0,y1], z=[z0,z1], mode="lines",
                                   line=dict(width=2), opacity=0.25, showlegend=False))

    # highlight path if exists
    if path:
        px = [nodes[n]["x"] for n in path]
        py = [nodes[n]["y"] for n in path]
        pz = [nodes[n].get("z",0) for n in path]
        fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines+markers",
                                   line=dict(width=8), marker=dict(size=5), name="Path"))

    fig.update_layout(scene=dict(aspectmode='data'),
                      margin=dict(l=0, r=0, t=0, b=0), height=700)
    st.plotly_chart(fig, use_container_width=True)

# sidebar json preview
st.sidebar.subheader("Loaded map JSON")
st.sidebar.code(json.dumps(data, indent=2), language="json")