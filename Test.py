# app.py

import json
import math
from pathlib import Path

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

# ---------- Config ----------
DATAFILE = "scis_map.json"  # put your cleaned JSON here
STAIRS_PENALTY = 3.0       # multiplier for edges that change z significantly
AUTO_ADJ_THRESHOLD = 0.01  # adjacency eps (in meters) for touching boxes

st.set_page_config(layout="wide", page_title="SCIS 3D Pathfinder")
st.title("SCIS 3D Pathfinder — 3D Visualization & Routing")

# ---------- Helpers ----------
def load_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def centroid_of_box(coords):
    # coords: dict with xMin,xMax,yMin,yMax,zMin,zMax
    x = (coords["xMin"] + coords["xMax"]) / 2.0
    y = (coords["yMin"] + coords["yMax"]) / 2.0
    z = (coords["zMin"] + coords["zMax"]) / 2.0
    return (x, y, z)

def box_touch_or_close(a, b, eps=AUTO_ADJ_THRESHOLD):
    # Returns True if boxes a and b overlap or touch in 2D (x/y) on same level
    # we allow small eps gap to treat narrow corridors as connected
    axmin, axmax = a["xMin"], a["xMax"]
    aymin, aymax = a["yMin"], a["yMax"]
    bxmin, bxmax = b["xMin"], b["xMax"]
    bymin, bymax = b["yMin"], b["yMax"]
    # overlap condition in x and y with eps allowance
    overlap_x = (axmin <= bxmax + eps) and (bxmin <= axmax + eps)
    overlap_y = (aymin <= bymax + eps) and (bymin <= aymax + eps)
    return overlap_x and overlap_y

def make_cuboid_mesh(coords, name=None):
    # axis-aligned cuboid from coords dict
    x0, x1 = coords["xMin"], coords["xMax"]
    y0, y1 = coords["yMin"], coords["yMax"]
    z0, z1 = coords["zMin"], coords["zMax"]
    # 8 vertices (bottom 4 then top 4)
    verts = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    # triangles: create faces by triangles (12 triangles)
    # faces defined as vertex indices
    i = [0, 0, 0, 4, 5, 6, 0, 1, 2, 4, 5, 6]
    j = [1, 2, 3, 5, 6, 7, 4, 5, 6, 0, 1, 2]
    k = [4, 4, 4, 0, 0, 0, 1, 2, 3, 5, 6, 7]
    return xs, ys, zs, i, j, k

def euclid(a, b):
    return math.dist(a, b)

# ---------- Load & Parse Map ----------
path = Path(DATAFILE)
if not path.exists():
    st.error(f"Map file not found: {DATAFILE}. Put your cleaned JSON file beside app.py.")
    st.stop()

school = load_map(DATAFILE)

# Collect nodes with coords and metadata
nodes = {}   # node_id -> dict with coords & metadata
edges = []   # (u,v,attrs)

# Iterate buildings -> levels -> rooms
for b in school.get("buildings", []):
    b_id = b.get("buildingId")
    for lvl in b.get("levels", []):
        lvl_id = lvl.get("levelId")
        # rooms
        for r in lvl.get("rooms", []):
            rid = r["roomId"]
            # some JSON variants use coords, some not; require coords present for 3D
            coords = r.get("coords")
            if not coords:
                # fallback: assign a small box using building position if available
                coords = {
                    "xMin": b.get("position", {}).get("xMin", 0) + 0.1,
                    "xMax": b.get("position", {}).get("xMax", 0) - 0.1,
                    "yMin": b.get("position", {}).get("yMin", 0) + 0.1,
                    "yMax": b.get("position", {}).get("yMax", 0) - 0.1,
                    "zMin": lvl.get("zRange", {}).get("zMin", 0),
                    "zMax": lvl.get("zRange", {}).get("zMax", lvl.get("zRange", {}).get("zMin", 3.5))
                }
            nodes[rid] = {
                "id": rid,
                "label": r.get("name", r.get("roomName", rid)),
                "function": r.get("function", ""),
                "building": b_id,
                "level": lvl_id,
                "coords": coords,
                "centroid": centroid_of_box(coords)
            }
        # entrances
        for e in lvl.get("entrances", []):
            eid = e["entranceId"]
            # entraces may not have coords; we try to reuse a nearby room box if available
            coords = e.get("coords")
            if not coords:
                # safe fallback: tiny box at building position or 0
                coords = {
                    "xMin": b.get("position", {}).get("xMin", 0),
                    "xMax": b.get("position", {}).get("xMin", 0) + 1,
                    "yMin": b.get("position", {}).get("yMin", 0),
                    "yMax": b.get("position", {}).get("yMin", 0) + 1,
                    "zMin": lvl.get("zRange", {}).get("zMin", 0),
                    "zMax": lvl.get("zRange", {}).get("zMin", 0) + 1
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
        # explicit connections inside level
        for c in lvl.get("connections", []):
            # connection entries may include from/to or only connId meaning special item
            src = c.get("from")
            dst = c.get("to")
            if src and dst:
                edges.append((src, dst, {"type": c.get("type", "hallway"), "meta": c}))

# ---------- Auto-connect adjacent rooms on same level ----------
# For each building-level, connect rooms whose boxes touch/overlap
for b in school.get("buildings", []):
    b_id = b.get("buildingId")
    for lvl in b.get("levels", []):
        lvl_id = lvl.get("levelId")
        # gather room ids on this level
        level_room_ids = []
        for r in lvl.get("rooms", []):
            level_room_ids.append(r["roomId"])
        for eid in lvl.get("entrances", []):
            level_room_ids.append(eid["entranceId"])
        # pairwise test
        for i in range(len(level_room_ids)):
            for j in range(i + 1, len(level_room_ids)):
                a_id = level_room_ids[i]
                b_id2 = level_room_ids[j]
                if a_id not in nodes or b_id2 not in nodes:
                    continue
                a_coords = nodes[a_id]["coords"]
                b_coords = nodes[b_id2]["coords"]
                # if boxes touch/overlap in XY (same level), connect
                if box_touch_or_close(a_coords, b_coords):
                    # compute euclidean centroid distance as base weight
                    w = euclid(nodes[a_id]["centroid"], nodes[b_id2]["centroid"])
                    edges.append((a_id, b_id2, {"type": "adjacency", "weight": float(w)}))

# ---------- Cross-building explicit corridor connections ----------
# If any connection edges reference buildings across levels, they already exist in edges list built earlier.
# We also add edges where explicit 'connections' specified targetBuildingId (from original cleaned JSON)
# Iterate original JSON for any connection objects that specify targetBuildingId
for b in school.get("buildings", []):
    for lvl in b.get("levels", []):
        for c in lvl.get("connections", []):
            src = c.get("connId") or c.get("from") or None
            # original schema used 'from'/'to'; earlier we added those. Also some have connId with coords
            # If connection specifies targetBuildingId and coords, try to find nearest nodes on each side.
            if c.get("targetBuildingId") and c.get("coords"):
                # find nearest nodes in this building/level to connection coords and nearest nodes in target building same level
                conn_centroid = centroid_of_box(c["coords"])
                # find nearest node inside current building-level
                cand_src = None
                min_dist = float("inf")
                for nid, n in nodes.items():
                    if n["building"] == b.get("buildingId") and n["level"] == lvl.get("levelId"):
                        d = euclid(n["centroid"], conn_centroid)
                        if d < min_dist:
                            min_dist = d; cand_src = nid
                # find candidate in target building (same approximate vertical level)
                cand_dst = None
                min_dist = float("inf")
                tgt_building = c["targetBuildingId"]
                for nid, n in nodes.items():
                    if n["building"] == tgt_building:
                        # prefer same zRange approx
                        d = euclid(n["centroid"], conn_centroid)
                        if d < min_dist:
                            min_dist = d; cand_dst = nid
                if cand_src and cand_dst:
                    edges.append((cand_src, cand_dst, {"type": c.get("type", "corridor"), "weight": euclid(nodes[cand_src]["centroid"], nodes[cand_dst]["centroid"])}))

# ---------- Build networkx Graph ----------
G = nx.Graph()

for nid, meta in nodes.items():
    G.add_node(nid, **meta)

for u, v, attrs in edges:
    if u not in G.nodes or v not in G.nodes:
        # skip invalid edges quietly
        continue
    # determine weight: use provided or centroid distance
    w = attrs.get("weight")
    if w is None:
        w = euclid(G.nodes[u]["centroid"], G.nodes[v]["centroid"])
    # if z differs significantly, apply stairs penalty (unless edge type indicates elevator)
    zu = G.nodes[u]["centroid"][2]; zv = G.nodes[v]["centroid"][2]
    if abs(zu - zv) > 0.5 and attrs.get("type") != "elevator":
        w = float(w) * STAIRS_PENALTY
    G.add_edge(u, v, weight=float(w), **attrs)

# ---------- UI controls ----------
st.sidebar.header("View / Routing Controls")
floors = set()
for n, d in G.nodes(data=True):
    floors.add(d.get("level"))
floors = sorted([f for f in floors if f is not None])
floor_choice = st.sidebar.selectbox("Show floor", ["All"] + floors, index=0)
show_buildings = st.sidebar.multiselect("Show buildings (empty = all)", [b["buildingId"] for b in school.get("buildings", [])])
if not show_buildings:
    show_buildings = [b["buildingId"] for b in school.get("buildings", [])]

# node selection
labels = nx.get_node_attributes(G, "label")
node_list = [n for n in G.nodes()]
def node_label(nid):
    meta = G.nodes[nid]
    return f"{meta.get('label')} ({nid}) - {meta.get('building')} / {meta.get('level')}"

start = st.sidebar.selectbox("Start node", node_list, format_func=node_label)
end = st.sidebar.selectbox("End node", node_list, index=min(1, len(node_list)-1), format_func=node_label)

# find path
find_btn = st.sidebar.button("Find path (A*)")

# ---------- A* search ----------
def heuristic(a, b):
    return euclid(G.nodes[a]["centroid"], G.nodes[b]["centroid"])

path = None
path_length = None
if find_btn:
    try:
        path = nx.astar_path(G, start, end, heuristic=heuristic, weight="weight")
        # compute total length
        path_length = sum(G[u][v]["weight"] for u, v in zip(path, path[1:]))
        st.sidebar.success(f"Path length ≈ {path_length:.1f} m")
    except nx.NetworkXNoPath:
        st.sidebar.error("No path found between selected nodes")

# ---------- 3D Visualization ----------
st.subheader("3D Map — interactive")
fig = go.Figure()

# color palette per building
building_ids = [b["buildingId"] for b in school.get("buildings", [])]
palette = {}
# simple palette generator (cycle)
colors = [
    "lightblue", "lightgreen", "lightpink", "lightyellow", "lightgray", "wheat", "lavender"
]
for i, bid in enumerate(building_ids):
    palette[bid] = colors[i % len(colors)]

# Add room cuboids
for nid, data in G.nodes(data=True):
    bld = data.get("building")
    if bld not in show_buildings:
        continue
    if floor_choice != "All" and data.get("level") != floor_choice:
        continue
    coords = data.get("coords")
    xs, ys, zs, i, j, k = make_cuboid_mesh(coords, name=data.get("label"))
    # slightly vary opacity/colour for path node highlight later
    opacity = 0.35
    color = palette.get(bld, "lightgray")
    # Make path rooms more vivid if in path
    if path and nid in path:
        opacity = 0.9
        color = "red"
    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        opacity=opacity,
        name=f"{data.get('label')} ({nid})",
        hovertext=f"{data.get('label')} ({nid})<br>{data.get('function')}",
        hoverinfo="text",
        flatshading=True,
        showscale=False,
        intensity=None,
        # color is assigned via 'color' prop in Mesh3d by setting facecolor (not directly): fallback to 'color' shorthand
        # Plotly's Mesh3d accepts 'color' for whole mesh
        color=color
    ))

# Draw edges as thin translucent lines (but only those on visible floors/buildings)
for u, v, ed in G.edges(data=True):
    u_meta = G.nodes[u]; v_meta = G.nodes[v]
    if u_meta.get("building") not in show_buildings or v_meta.get("building") not in show_buildings:
        continue
    if floor_choice != "All":
        # show edges only if both endpoints match the floor selection OR if vertical connection to selected floor
        if not (u_meta.get("level") == floor_choice or v_meta.get("level") == floor_choice):
            continue
    x0, y0, z0 = u_meta["centroid"]
    x1, y1, z1 = v_meta["centroid"]
    fig.add_trace(go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1],
                               mode="lines",
                               line=dict(width=2, color="gray"),
                               opacity=0.25,
                               hoverinfo="none",
                               showlegend=False))

# Draw path (if exists) as a bold line
if path:
    px = [G.nodes[n]["centroid"][0] for n in path]
    py = [G.nodes[n]["centroid"][1] for n in path]
    pz = [G.nodes[n]["centroid"][2] for n in path]
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines+markers+text",
                               line=dict(width=8, color="red"),
                               marker=dict(size=4, color="red"),
                               text=[G.nodes[n]["label"] for n in path],
                               textposition="top center",
                               name="Route"))

# Layout tuning
fig.update_layout(scene=dict(aspectmode="data",
                             xaxis=dict(title="X (m)"),
                             yaxis=dict(title="Y (m)"),
                             zaxis=dict(title="Z (m)")),
                  margin=dict(l=0, r=0, t=0, b=0),
                  height=800)

st.plotly_chart(fig, use_container_width=True)

# ---------- Text directions / details ----------
st.subheader("Route details")
if path:
    st.markdown("**Step-by-step (nodes):**")
    for i, nid in enumerate(path):
        meta = G.nodes[nid]
        st.write(f"{i+1}. {meta.get('label')} — {meta.get('building')} / {meta.get('level')} ({meta.get('function')})")
    st.markdown(f"**Approx. total distance:** {path_length:.1f} meters")
else:
    st.info("Select start and end nodes on the left, then click **Find path (A\*)**.")

# ---------- Export route as JSON ----------
if st.sidebar.button("Export route JSON") and path:
    out = {
        "start": start,
        "end": end,
        "path": path,
        "distance_m": float(path_length)
    }
    st.sidebar.download_button("Download route", data=json.dumps(out, indent=2), file_name="route.json", mime="application/json")