# app.py

import json
import math
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# ---------- Config ----------
DATAFILE = "scis_map.json"  # your external cleaned JSON file
STAIRS_PENALTY = 3.0       # multiplier for edges that change z significantly
AUTO_ADJ_THRESHOLD = 0.01  # adjacency eps (in meters) for touching boxes

st.set_page_config(layout="wide", page_title="SCIS 3D Pathfinder")
st.title("SCIS 3D Pathfinder — 3D Visualization & Routing")

# ---------- Helpers ----------
def load_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def centroid_of_box(coords):
    x = (coords["xMin"] + coords["xMax"]) / 2.0
    y = (coords["yMin"] + coords["yMax"]) / 2.0
    z = (coords["zMin"] + coords["zMax"]) / 2.0
    return (x, y, z)

def box_touch_or_close(a, b, eps=AUTO_ADJ_THRESHOLD):
    axmin, axmax = a["xMin"], a["xMax"]
    aymin, aymax = a["yMin"], a["yMax"]
    bxmin, bxmax = b["xMin"], b["xMax"]
    bymin, bymax = b["yMin"], b["yMax"]
    overlap_x = (axmin <= bxmax + eps) and (bxmin <= axmax + eps)
    overlap_y = (aymin <= bymax + eps) and (bymin <= aymax + eps)
    return overlap_x and overlap_y

def make_cuboid_mesh(coords):
    x0, x1 = coords["xMin"], coords["xMax"]
    y0, y1 = coords["yMin"], coords["yMax"]
    z0, z1 = coords["zMin"], coords["zMax"]
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    i = [0,0,0,4,5,6,0,1,2,4,5,6]
    j = [1,2,3,5,6,7,4,5,6,0,1,2]
    k = [4,4,4,0,0,0,1,2,3,5,6,7]
    return xs, ys, zs, i, j, k

def euclid(a, b):
    return math.dist(a, b)

# ---------- Load Map ----------
path = Path(DATAFILE)
if not path.exists():
    st.error(f"Map file not found: {DATAFILE}. Put your cleaned JSON file beside app.py.")
    st.stop()

school = load_map(DATAFILE)

# ---------- Build Nodes and Edges ----------
nodes = {}
edges = []

for b in school.get("buildings", []):
    b_id = b.get("buildingId")
    for lvl in b.get("levels", []):
        lvl_id = lvl.get("levelId")
        for r in lvl.get("rooms", []):
            rid = r["roomId"]
            coords = r.get("coords") or {
                "xMin": b.get("position", {}).get("xMin", 0)+0.1,
                "xMax": b.get("position", {}).get("xMax", 0)-0.1,
                "yMin": b.get("position", {}).get("yMin", 0)+0.1,
                "yMax": b.get("position", {}).get("yMax", 0)-0.1,
                "zMin": lvl.get("zRange", {}).get("zMin",0),
                "zMax": lvl.get("zRange", {}).get("zMax", lvl.get("zRange", {}).get("zMin",3.5))
            }
            nodes[rid] = {
                "id": rid,
                "label": r.get("name", r.get("roomName", rid)),
                "function": r.get("function",""),
                "building": b_id,
                "level": lvl_id,
                "coords": coords,
                "centroid": centroid_of_box(coords)
            }
        for e in lvl.get("entrances", []):
            eid = e["entranceId"]
            coords = e.get("coords") or {
                "xMin": b.get("position", {}).get("xMin",0),
                "xMax": b.get("position", {}).get("xMin",0)+1,
                "yMin": b.get("position", {}).get("yMin",0),
                "yMax": b.get("position", {}).get("yMin",0)+1,
                "zMin": lvl.get("zRange", {}).get("zMin",0),
                "zMax": lvl.get("zRange", {}).get("zMin",0)+1
            }
            nodes[eid] = {
                "id": eid,
                "label": e.get("entranceName", eid),
                "function": "entrance",
                "building": b_id,
                "level": lvl_id,
                "coords": coords,
                "centroid": centroid_of_box(coords)
            }
        for c in lvl.get("connections", []):
            src = c.get("from")
            dst = c.get("to")
            if src and dst:
                edges.append((src, dst, {"type": c.get("type","hallway"), "meta": c}))

# ---------- Auto-connect adjacent rooms ----------
for b in school.get("buildings", []):
    for lvl in b.get("levels", []):
        level_ids = [r["roomId"] for r in lvl.get("rooms", [])] + [e["entranceId"] for e in lvl.get("entrances",[])]
        for i in range(len(level_ids)):
            for j in range(i+1,len(level_ids)):
                a_id,b_id2 = level_ids[i],level_ids[j]
                if a_id not in nodes or b_id2 not in nodes: continue
                if box_touch_or_close(nodes[a_id]["coords"], nodes[b_id2]["coords"]):
                    w = euclid(nodes[a_id]["centroid"], nodes[b_id2]["centroid"])
                    edges.append((a_id,b_id2,{"type":"adjacency","weight":float(w)}))

# ---------- Build Graph ----------
G = nx.Graph()
for nid, meta in nodes.items():
    G.add_node(nid, **meta)
for u,v,attrs in edges:
    if u not in G or v not in G: continue
    w = attrs.get("weight") or euclid(G.nodes[u]["centroid"], G.nodes[v]["centroid"])
    zu,zv = G.nodes[u]["centroid"][2], G.nodes[v]["centroid"][2]
    if abs(zu-zv) > 0.5 and attrs.get("type") != "elevator":
        w *= STAIRS_PENALTY
    G.add_edge(u,v,weight=float(w),**attrs)

# ---------- Sidebar Controls ----------
st.sidebar.header("Controls")
floors = sorted({d.get("level") for n,d in G.nodes(data=True) if d.get("level")})
floor_choice = st.sidebar.selectbox("Floor", ["All"]+floors)
show_buildings = st.sidebar.multiselect("Buildings", [b["buildingId"] for b in school.get("buildings", [])])
if not show_buildings:
    show_buildings = [b["buildingId"] for b in school.get("buildings", [])]

node_list = list(G.nodes)
def node_label(nid):
    meta = G.nodes[nid]
    return f"{meta.get('label')} ({nid}) - {meta.get('building')} / {meta.get('level')}"

start = st.sidebar.selectbox("Start node", node_list, format_func=node_label)
end = st.sidebar.selectbox("End node", node_list, index=min(1,len(node_list)-1), format_func=node_label)
find_btn = st.sidebar.button("Find path (A*)")
path = None
path_length = None
if find_btn:
    try:
        path = nx.astar_path(G,start,end,heuristic=lambda a,b: euclid(G.nodes[a]["centroid"],G.nodes[b]["centroid"]),weight="weight")
        path_length = sum(G[u][v]["weight"] for u,v in zip(path,path[1:]))
        st.sidebar.success(f"Path ≈ {path_length:.1f} m")
    except nx.NetworkXNoPath:
        st.sidebar.error("No path found")

# ---------- 3D Visualization ----------
fig = go.Figure()
colors = ["lightblue","lightgreen","lightpink","lightyellow","lightgray","wheat","lavender"]
building_ids = [b["buildingId"] for b in school.get("buildings",[])]
palette = {bid: colors[i%len(colors)] for i,bid in enumerate(building_ids)}

for nid,data in G.nodes(data=True):
    if data.get("building") not in show_buildings: continue
    if floor_choice!="All" and data.get("level")!=floor_choice: continue
    xs,ys,zs,i,j,k = make_cuboid_mesh(data["coords"])
    opacity = 0.35
    color = palette.get(data["building"],"lightgray")
    if path and nid in path:
        opacity = 0.9
        color = "red"
    fig.add_trace(go.Mesh3d(x=xs,y=ys,z=zs,i=i,j=j,k=k,color=color,opacity=opacity,hovertext=data["label"],hoverinfo="text"))

for u,v in G.edges():
    u_meta,v_meta = G.nodes[u], G.nodes[v]
    if u_meta.get("building") not in show_buildings or v_meta.get("building") not in show_buildings: continue
    if floor_choice!="All" and not(u_meta.get("level")==floor_choice or v_meta.get("level")==floor_choice): continue
    x0,y0,z0 = u_meta["centroid"]
    x1,y1,z1 = v_meta["centroid"]
    fig.add_trace(go.Scatter3d(x=[x0,x1],y=[y0,y1],z=[z0,z1],mode="lines",line=dict(width=2,color="gray"),opacity=0.25,showlegend=False))

if path:
    px,py,pz = zip(*[G.nodes[n]["centroid"] for n in path])
    fig.add_trace(go.Scatter3d(x=px,y=py,z=pz,mode="lines+markers",line=dict(width=8,color="red"),marker=dict(size=4,color="red"),name="Route"))

fig.update_layout(scene=dict(aspectmode="data",xaxis_title="X",yaxis_title="Y",zaxis_title="Z"),margin=dict(l=0,r=0,t=0,b=0),height=800)
st.plotly_chart(fig,use_container_width=True)

# ---------- Route Details ----------
st.subheader("Route Details")
if path:
    st.markdown("**Step-by-step nodes:**")
    for i,nid in enumerate(path):
        meta = G.nodes[nid]
        st.write(f"{i+1}. {meta['label']} — {meta['building']} / {meta['level']} ({meta['function']})")
    st.markdown(f"**Approx. distance:** {path_length:.1f} meters")
else:
    st.info("Select start and end nodes, then click **Find path (A\*)**.")

# ---------- Export JSON ----------
if st.sidebar.button("Export route JSON") and path:
    out = {"start":start,"end":end,"path":path,"distance_m":float(path_length)}
    st.sidebar.download_button("Download route",data=json.dumps(out,indent=2),file_name="route.json",mime="application/json")
