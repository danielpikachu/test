import json
import networkx as nx
import streamlit as st
import plotly.graph_objects as go

# -------- Load JSON --------
uploaded_file = st.file_uploader("Upload your school_map.json", type="json")
if uploaded_file:
    school = json.load(uploaded_file)
else:
    st.warning("Upload a JSON file to display the map.")
    st.stop()

# -------- Create Graph --------
G = nx.Graph()
rooms = {}

for building in school["buildings"]:
    for level in building["levels"]:
        for room in level["rooms"]:
            rid = room["roomId"]
            x = (room["coords"]["xMin"] + room["coords"]["xMax"]) / 2
            y = (room["coords"]["yMin"] + room["coords"]["yMax"]) / 2
            z = (room["coords"]["zMin"] + room["coords"]["zMax"]) / 2
            G.add_node(rid, pos=(x, y, z))
            rooms[rid] = room

        # connections
        if "connections" in level:
            for conn in level["connections"]:
                target = conn["targetBuildingId"]
                # Connect corridor midpoint to nearest node in target building
                x1 = (conn["coords"]["xMin"] + conn["coords"]["xMax"]) / 2
                y1 = (conn["coords"]["yMin"] + conn["coords"]["yMax"]) / 2
                z1 = (conn["coords"]["zMin"] + conn["coords"]["zMax"]) / 2
                # Find nearest room in target building
                target_rooms = [r for b in school["buildings"] if b["buildingId"]==target for l in b["levels"] for r in l["rooms"]]
                nearest = min(target_rooms, key=lambda r: ((x1-(r["coords"]["xMin"]+r["coords"]["xMax"])/2)**2 + (y1-(r["coords"]["yMin"]+r["coords"]["yMax"])/2)**2 + (z1-(r["coords"]["zMin"]+r["coords"]["zMax"])/2)**2)**0.5)
                G.add_edge(conn["name"], nearest["roomId"], weight=1)
                G.add_edge(conn["name"], rid, weight=1)
                G.add_node(conn["name"], pos=(x1, y1, z1))

# -------- Room Selection --------
room_ids = list(rooms.keys())
start_room = st.selectbox("Start Room", room_ids)
end_room = st.selectbox("End Room", room_ids)

# -------- Find Path --------
if start_room and end_room:
    path = nx.shortest_path(G, start_room, end_room, weight="weight")
    st.write("Path:", " → ".join(path))

# -------- Plot 3D Map --------
fig = go.Figure()

# Draw boxes
for room in rooms.values():
    x0, x1 = room["coords"]["xMin"], room["coords"]["xMax"]
    y0, y1 = room["coords"]["yMin"], room["coords"]["yMax"]
    z0, z1 = room["coords"]["zMin"], room["coords"]["zMax"]

    fig.add_trace(go.Mesh3d(
        x=[x0,x1,x1,x0,x0,x1,x1,x0],
        y=[y0,y0,y1,y1,y0,y0,y1,y1],
        z=[z0,z0,z0,z0,z1,z1,z1,z1],
        color='lightblue',
        opacity=0.5,
        name=room["roomId"]
    ))

# Draw path as line
if start_room and end_room:
    px, py, pz = zip(*[G.nodes[n]["pos"] for n in path])
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode='lines+markers', line=dict(color='red', width=5), marker=dict(size=3), name='Path'))

fig.update_layout(scene=dict(
    xaxis_title='X (m)',
    yaxis_title='Y (m)',
    zaxis_title='Z (m)',
), height=800, width=1000)

st.plotly_chart(fig, use_container_width=True)
