import streamlit as st
import json
import networkx as nx
import re
import plotly.graph_objects as go

# ==============================
# Load JSON Data
# ==============================
@st.cache_data
def load_data():
    with open("school_map.json", "r") as f:
        return json.load(f)

school = load_data()

# ==============================
# Build Graph from JSON
# ==============================
def build_graph(school):
    G = nx.Graph()

    for building in school.get("buildings", []):
        for level in building.get("levels", []):
            level_id = level.get("levelId")

            # Add rooms
            for room in level.get("rooms", []):
                G.add_node(room["roomId"], label=room["roomName"], level=level_id, type="room")

            # Add entrances
            for ent in level.get("entrances", []):
                G.add_node(ent["entranceId"], label=ent["entranceName"], level=level_id, type="entrance")

            # Add connections
            for conn in level.get("connections", []):
                conn_id = conn.get("connectionId", "")
                conn_name = conn.get("connectionName", "")

                # Try to find room IDs in the text
                rooms = re.findall(r"[A-Z]\d{2,3}", conn_id + " " + conn_name)

                if len(rooms) >= 2:
                    u, v = rooms[0], rooms[1]
                else:
                    # fallback: link entrance to first room if available
                    if level.get("rooms") and level.get("entrances"):
                        u = level["entrances"][0]["entranceId"]
                        v = level["rooms"][0]["roomId"]
                    else:
                        continue

                weight = 1.0
                attrs = {k: v for k, v in conn.items() if k not in ["weight"]}
                G.add_edge(u, v, weight=weight, **attrs)

    return G

G = build_graph(school)

# ==============================
# Pathfinding
# ==============================
def find_path(G, start, end):
    try:
        return nx.shortest_path(G, source=start, target=end, weight="weight")
    except nx.NetworkXNoPath:
        return None

# ==============================
# 3D Visualization
# ==============================
def plot_school_3d(G, path=None):
    pos = {}
    z_levels = {}
    level_heights = {}

    # Assign Z by level
    for i, building in enumerate(school.get("buildings", [])):
        for j, level in enumerate(building.get("levels", [])):
            level_id = level.get("levelId")
            level_heights[level_id] = j * 10 + i * 50  # space out buildings

    # Assign positions
    for k, node in enumerate(G.nodes()):
        level_id = G.nodes[node].get("level")
        z = level_heights.get(level_id, 0)
        pos[node] = (k % 10 * 10, (k // 10) * 10, z)
        z_levels[node] = z

    # Draw edges
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    # Highlight path
    path_x, path_y, path_z = [], [], []
    if path:
        for i in range(len(path) - 1):
            x0, y0, z0 = pos[path[i]]
            x1, y1, z1 = pos[path[i + 1]]
            path_x.extend([x0, x1, None])
            path_y.extend([y0, y1, None])
            path_z.extend([z0, z1, None])

    fig = go.Figure()

    # Edges
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="gray", width=2),
        name="Connections"
    ))

    # Nodes
    fig.add_trace(go.Scatter3d(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        z=[pos[n][2] for n in G.nodes()],
        mode="markers+text",
        text=[G.nodes[n].get("label", n) for n in G.nodes()],
        textposition="top center",
        marker=dict(size=6, color="blue"),
        name="Nodes"
    ))

    # Path highlight
    if path:
        fig.add_trace(go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode="lines+markers",
            line=dict(color="red", width=6),
            marker=dict(size=8, color="red"),
            name="Shortest Path"
        ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=True
    )

    return fig

# ==============================
# Streamlit UI
# ==============================
st.title(f"Pathfinding App - {school.get('schoolName', 'School')}")

room_ids = [n for n, d in G.nodes(data=True) if d.get("type") == "room"]

col1, col2 = st.columns(2)
with col1:
    start = st.selectbox("Select Start Room", room_ids)
with col2:
    end = st.selectbox("Select End Room", room_ids)

if st.button("Find Path"):
    path = find_path(G, start, end)
    if path:
        st.success(" → ".join(path))
        fig = plot_school_3d(G, path)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No path found.")
