import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

st.set_page_config(
    page_title="SCIS Navigation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.switch_backend('Agg')

# --------------------------
# Google Sheets 配置
# --------------------------
SHEET_NAME = 'Navigation visitors'
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

def get_credentials():
    try:
        service_account_info = st.secrets["google_service_account"]
        return Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPE
        )
    except KeyError:
        st.error("Streamlit Secrets中未找到google_service_account配置，请检查TOML格式")
        return None
    except Exception as e:
        st.error(f"密钥加载失败: {str(e)}")
        return None

def init_google_sheet():
    try:
        creds = get_credentials()
        if not creds:
            return None
        client = gspread.authorize(creds)
        
        try:
            sheet = client.open(SHEET_NAME)
        except gspread.exceptions.SpreadsheetNotFound:
            sheet = client.create(SHEET_NAME)
        
        try:
            stats_worksheet = sheet.worksheet("Access_Stats")
        except gspread.exceptions.WorksheetNotFound:
            stats_worksheet = sheet.add_worksheet(title="Access_Stats", rows="1000", cols=3)
            stats_worksheet.append_row(["Timestamp", "Access_Count", "Total_Accesses"])
            stats_worksheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1, 1])
        
        return stats_worksheet
    except Exception as e:
        return None

def update_access_count(worksheet):
    if not worksheet:
        return 0
        
    try:
        records = worksheet.get_all_values()
        if len(records) < 2:
            return 0
            
        last_row = records[-1]
        total = int(last_row[2]) if last_row[2].isdigit() else 0
        new_total = total + 1
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([current_time, 1, new_total])
        
        return new_total
    except Exception as e:
        return 0

def get_total_accesses(worksheet):
    if not worksheet:
        return 0
        
    try:
        records = worksheet.get_all_values()
        if len(records) < 2:
            return 0
            
        last_row = records[-1]
        return int(last_row[2]) if last_row[2].isdigit() else 0
    except Exception as e:
        return 0

# --------------------------
# 配色
# --------------------------
COLORS = {
    'building': {'A': 'lightblue', 'B': 'lightgreen', 'C': 'lightcoral', 'Gate': 'gold'},
    'floor_z': {-9: 'darkgray', -6: 'blue', -3: 'cyan', 2: 'green', 7: 'orange', 12: 'purple', 17: 'teal'},
    'corridor_line': {'A': 'cyan', 'B': 'forestgreen', 'C': 'salmon', 'Gate': 'darkgoldenrod'},
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
        'GateStairs': '#FFD700'
    },
    'stair_label': 'darkred',
    'classroom_label': 'black',
    'path': 'red',
    'start_marker': 'limegreen',
    'start_label': 'green',
    'end_marker': 'magenta',
    'end_label': 'purple',
    'connect_corridor': 'gold',
    'building_label': {'A': 'darkblue', 'B': 'darkgreen', 'C': 'darkred', 'Gate': 'darkgoldenrod'}
}

def load_school_data_detailed(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data file: {str(e)}")
        return None

# ====================== 3D绘图函数 ======================
def plot_3d_map_plotly(school_data, graph=None, display_options=None):
    fig = go.Figure()

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

    building_label_positions = {}
    shown_stairs_legends = set()

    for building_id in school_data.keys():
        if building_id == 'gate':
            building_name = 'Gate'
        elif building_id.startswith('building'):
            building_name = building_id.replace('building', '')
        else:
            continue
            
        building_data = school_data[building_id]
        
        displayed_levels = []
        max_displayed_z = -float('inf')
        max_displayed_y = -float('inf')
        corresponding_x = 0
        
        for level in building_data['levels']:
            level_name = level['name']
            raw_z = level['z']
            z = raw_z + 10
            
            show_level = show_all
            if not show_all:
                if building_name == 'B':
                    show_level = (level_name == 'level1') or any((building_name, s, level_name) in path_stairs for s in ['StairsB1','StairsB2'])
                elif building_name == 'Gate':
                    show_level = True
                else:
                    show_level = (level_name == start_level) or (level_name == end_level)
            
            if show_level:
                displayed_levels.append(level)
                if z > max_displayed_z:
                    max_displayed_z = z
                
                fp = level['floorPlane']
                current_max_y = fp['maxY']
                if current_max_y > max_displayed_y:
                    max_displayed_y = current_max_y
                    corresponding_x = (fp['minX'] + fp['maxX']) / 2
            
            floor_border_color = COLORS['floor_z'].get(raw_z, 'gray')
            building_fill_color = COLORS['building'].get(building_name, 'lightgray')

            if show_level:
                x_vals = [fp['minX'], fp['maxX'], fp['maxX'], fp['minX'], fp['minX']]
                y_vals = [fp['minY'], fp['minY'], fp['maxY'], fp['maxY'], fp['minY']]
                z_vals = [z] * 5

                fig.add_trace(go.Scatter3d(
                    x=x_vals, y=y_vals, z=z_vals,
                    mode='lines',
                    line=dict(color=floor_border_color, width=4),
                    name=f"Building {building_name}-{level_name}",
                    legendgroup=f"Building {building_name}",
                    showlegend=True
                ))
                fig.add_trace(go.Mesh3d(
                    x=x_vals[:4], y=y_vals[:4], z=z_vals[:4],
                    color=building_fill_color, opacity=0.3, showlegend=False
                ))

                for corridor in level['corridors']:
                    points = corridor['points']
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]
                    z_coords = [p[2]+10 for p in points]
                    
                    is_external = corridor.get('type') == 'external'
                    is_connect = 'connectToBuilding' in corridor.get('name','') or 'gateTo' in corridor.get('name','')
                    
                    if is_external:
                        corr_line_color = 'gray'
                        corr_line_width = 5
                        dash = 'dash'
                    elif is_connect:
                        corr_line_color = COLORS['connect_corridor']
                        corr_line_width = 7
                        dash = 'solid'
                    else:
                        corr_line_color = COLORS['corridor_line'].get(building_name, 'gray')
                        corr_line_width = 5
                        dash = 'solid'

                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z_coords, mode='lines',
                        line=dict(color=corr_line_color, width=corr_line_width, dash=dash), showlegend=False
                    ))
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z_coords, mode='markers',
                        marker=dict(color=COLORS['corridor_node'], size=3), showlegend=False
                    ))

                for classroom in level['classrooms']:
                    x, y, _ = classroom['coordinates']
                    w, d = classroom['size']
                    name = classroom['name']
                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z], mode='markers+text',
                        marker=dict(color=building_fill_color, size=7),
                        text=name, textposition="top center", textfont=dict(size=9), showlegend=False
                    ))
                    cx = [x, x+w, x+w, x, x]
                    cy = [y, y, y+d, y+d, y]
                    cz = [z]*5
                    fig.add_trace(go.Scatter3d(
                        x=cx, y=cy, z=cz, mode='lines',
                        line=dict(color=floor_border_color, width=1, dash='dash'), opacity=0.6, showlegend=False
                    ))

            for stair in level['stairs']:
                s_name = stair['name']
                is_path = (building_name, s_name, level_name) in path_stairs
                if show_all or show_level or is_path:
                    x, y, _ = stair['coordinates']
                    color = COLORS['stair'].get(s_name, 'red')
                    size = 12 if is_path else 9
                    legend_name = f"{building_name}-{s_name}"
                    show_legend = legend_name not in shown_stairs_legends
                    if show_legend:
                        shown_stairs_legends.add(legend_name)
                    fig.add_trace(go.Scatter3d(
                        x=[x], y=[y], z=[z], mode='markers+text',
                        marker=dict(color=color, size=size, symbol='diamond', line=dict(width=2)),
                        text=s_name, textposition="top center", textfont=dict(size=9, color='darkred'),
                        name=legend_name, legendgroup="Stairs", showlegend=show_legend
                    ))
        
        if displayed_levels:
            label_z = max_displayed_z + 1.5
            label_y = max_displayed_y + (3 if building_name != 'B' else -2)
            building_label_positions[building_name] = (corresponding_x, label_y, label_z)

    for bld, (x, y, z) in building_label_positions.items():
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z], mode='text',
            text=f"Building {bld}",
            textfont=dict(size=14, color=COLORS['building_label'][bld], family='Arial bold'),
            showlegend=False
        ))

    if path and graph and not show_all:
        try:
            xs, ys, zs, labels = [], [], [], []
            for nid in path:
                c = graph.nodes[nid]['coordinates']
                xs.append(c[0])
                ys.append(c[1])
                zs.append(c[2]+10)
                labels.append(graph.nodes[nid]['name'])
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines+markers',
                line=dict(color=COLORS['path'], width=5), marker=dict(color=COLORS['path'], size=4), name="Path"
            ))
            fig.add_trace(go.Scatter3d(
                x=[xs[0]], y=[ys[0]], z=[zs[0]], mode='markers+text',
                marker=dict(color=COLORS['start_marker'], size=14),
                text=f"Start\n{labels[0]}", textposition="top center", textfont=dict(size=11, color='green'), name="Start"
            ))
            fig.add_trace(go.Scatter3d(
                x=[xs[-1]], y=[ys[-1]], z=[zs[-1]], mode='markers+text',
                marker=dict(color=COLORS['end_marker'], size=14),
                text=f"End\n{labels[-1]}", textposition="top center", textfont=dict(size=11, color='purple'), name="End"
            ))
        except Exception:
            pass

    fig.update_layout(
        title=dict(text="Campus 3D Navigation Map", font=dict(size=22, color="gray"), x=0.5),
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
                   aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.8)),
        margin=dict(l=0, r=0, t=60, b=0), height=880
    )
    return fig

def plot_3d_map(school_data, graph=None, display_options=None):
    fig = plot_3d_map_plotly(school_data, graph, display_options)
    return fig, None

# --------------------------
# 图与寻路
# --------------------------
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}
    def add_node(self, building_id, node_type, name, level, coords):
        bn = 'Gate' if building_id == 'gate' else building_id.replace('building','')
        nid = f"{bn}-{node_type}-{name}@{level}"
        self.nodes[nid] = {'building':bn,'type':node_type,'name':name,'level':level,'coordinates':coords,'neighbors':{}}
        self.node_id_map[(building_id,node_type,name,level)] = nid
        if node_type=='classroom': self.node_id_map[(bn,name,level)]=nid
        return nid
    def add_edge(self,a,b,w):
        if a in self.nodes and b in self.nodes:
            self.nodes[a]['neighbors'][b]=w
            self.nodes[b]['neighbors'][a]=w

def euclidean_distance(a,b,p=15):
    return np.sqrt(sum((ax-bx)**2 for ax,bx in zip(a,b))) + p*abs(a[2]-b[2])

def get_direction_between_nodes(g,c,n):
    cc = g.nodes[c]['coordinates']
    nc = g.nodes[n]['coordinates']
    dx,dy = nc[0]-cc[0], nc[1]-cc[1]
    if abs(dy)>0.1: return "向前" if dy>0 else "向后"
    if abs(dx)>0.1: return "向右" if dx>0 else "向左"
    return ""

def build_navigation_graph(data):
    g=Graph()
    for bid in data:
        if not (bid.startswith('building') or bid=='gate'):continue
        bd=data[bid]
        for lv in bd['levels']:
            lvn=lv['name']
            for cr in lv['classrooms']:
                g.add_node(bid,'classroom',cr['name'],lvn,cr['coordinates'])
            for st in lv['stairs']:
                g.add_node(bid,'stair',st['name'],lvn,st['coordinates'])
            for i,cor in enumerate(lv['corridors']):
                cn=cor.get('name',f'cor{i}')
                for j,p in enumerate(cor['points']):
                    g.add_node(bid,'corridor',f'{cn}-p{j}',lvn,p)
    for bid in data:
        if not (bid.startswith('building') or bid=='gate'):continue
        bn='Gate' if bid=='gate' else bid.replace('building','')
        bd=data[bid]
        for lv in bd['levels']:
            lvn=lv['name']
            cors=[n for n,inf in g.nodes.items() if inf['building']==bn and inf['type']=='corridor' and inf['level']==lvn]
            for i,cor in enumerate(lv['corridors']):
                cn=cor.get('name',f'cor{i}')
                pts=cor['points']
                for j in range(len(pts)-1):
                    a=g.node_id_map.get((bid,'corridor',f'{cn}-p{j}',lvn))
                    b=g.node_id_map.get((bid,'corridor',f'{cn}-p{j+1}',lvn))
                    if a and b: g.add_edge(a,b,euclidean_distance(g.nodes[a]['coordinates'],g.nodes[b]['coordinates'],0))
            for i in range(len(cors)):
                for j in range(i+1,len(cors)):
                    a,b=cors[i],cors[j]
                    d=euclidean_distance(g.nodes[a]['coordinates'],g.nodes[b]['coordinates'],0)
                    if d<3:g.add_edge(a,b,d)
            for cl in [n for n,inf in g.nodes.items() if inf['building']==bn and inf['type']=='classroom' and inf['level']==lvn]:
                best=min(cors,key=lambda x:euclidean_distance(g.nodes[cl]['coordinates'],g.nodes[x]['coordinates'],0))
                g.add_edge(cl,best,euclidean_distance(g.nodes[cl]['coordinates'],g.nodes[best]['coordinates'],0))
            for st in [n for n,inf in g.nodes.items() if inf['building']==bn and inf['type']=='stair' and inf['level']==lvn]:
                best=min(cors,key=lambda x:euclidean_distance(g.nodes[st]['coordinates'],g.nodes[x]['coordinates'],0))
                g.add_edge(st,best,euclidean_distance(g.nodes[st]['coordinates'],g.nodes[best]['coordinates'],0))
        stairs=set((inf['building'],inf['name']) for n,inf in g.nodes.items() if inf['type']=='stair')
        for b,sn in stairs:
            lst=[(n,inf['coordinates'],inf['level']) for n,inf in g.nodes.items() if inf['building']==b and inf['name']==sn and inf['type']=='stair']
            lst.sort(key=lambda x:x[1][2])
            for i in range(len(lst)-1):
                a,b=lst[i][0],lst[i+1][0]
                g.add_edge(a,b,euclidean_distance(g.nodes[a]['coordinates'],g.nodes[b]['coordinates']))
        for bid in data:
            if not (bid.startswith('building') or bid=='gate'):continue
            bd=data[bid]
            for conn in bd.get('connections',[]):
                fn,fl=conn['from']
                tn,tl=conn['to']
                ft='stair' if fn.startswith(('Stairs','GateStairs')) else 'classroom' if any(fn==c['name'] for l in bd['levels'] for c in l.get('classrooms',[])) else 'corridor'
                tbid=bid
                if 'ENTRANCE' in tn:tbid='buildingA'
                if 'connectToBuildingAAndC' in tn:tbid='buildingB'
                if 'SCHOOL CLINIC' in tn:tbid='buildingC'
                if 'connectToBuildingB' in tn:tbid='buildingB'
                if 'connectToBuildingC' in tn:tbid='buildingC'
                tt='stair' if tn.startswith(('Stairs','GateStairs')) else 'classroom' if tbid in data and any(tn==c['name'] for l in data[tbid]['levels'] for c in l.get('classrooms',[])) else 'corridor'
                ffn=fn+'-p0' if ft=='corridor' else fn
                ttn=tn+'-p0' if tt=='corridor' else tn
                f=g.node_id_map.get((bid,ft,ffn,fl))
                t=g.node_id_map.get((tbid,tt,ttn,tl))
                if f and t:g.add_edge(f,t,euclidean_distance(g.nodes[f]['coordinates'],g.nodes[t]['coordinates'],0))
    a,b,c='buildingA','buildingB','buildingC'
    l='level1'
    abn='connectToBuildingB-p1'
    bbn='connectToBuildingAAndC-p1'
    ab=g.node_id_map.get((a,'corridor',abn,l))
    bb=g.node_id_map.get((b,'corridor',bbn,l))
    if ab and bb:g.add_edge(ab,bb,euclidean_distance(g.nodes[ab]['coordinates'],g.nodes[bb]['coordinates'],0))
    bcn='connectToBuildingAAndC-p0'
    cbn='connectToBuildingB-p1'
    bc=g.node_id_map.get((b,'corridor',bcn,l))
    cb=g.node_id_map.get((c,'corridor',cbn,l))
    if bc and cb:g.add_edge(bc,cb,euclidean_distance(g.nodes[bc]['coordinates'],g.nodes[cb]['coordinates'],0))
    ac1n='connectToBuildingC-p3'
    ca1n='connectToBuildingA-p0'
    ac1=g.node_id_map.get((a,'corridor',ac1n,l))
    ca1=g.node_id_map.get((c,'corridor',ca1n,l))
    if ac1 and ca1:g.add_edge(ac1,ca1,euclidean_distance(g.nodes[ac1]['coordinates'],g.nodes[ca1]['coordinates'],0))
    l3='level3'
    ac3n='connectToBuildingC-p2'
    ca3n='connectToBuildingA-p0'
    ac3=g.node_id_map.get((a,'corridor',ac3n,l3))
    ca3=g.node_id_map.get((c,'corridor',ca3n,l3))
    if ac3 and ca3:g.add_edge(ac3,ca3,euclidean_distance(g.nodes[ac3]['coordinates'],g.nodes[ca3]['coordinates'],0))
    return g

def dijkstra(g,s):
    dist={n:float('inf') for n in g.nodes}
    prev={n:None for n in g.nodes}
    dist[s]=0
    q=set(g.nodes)
    while q:
        u=min(q,key=lambda x:dist[x])
        q.remove(u)
        if dist[u]==float('inf'):break
        for v,w in g.nodes[u]['neighbors'].items():
            if dist[v]>dist[u]+w:
                dist[v]=dist[u]+w
                prev[v]=u
    return dist,prev

def construct_path(prev,e):
    p=[]
    while e:
        p.insert(0,e)
        e=prev[e]
    return p if len(p)>1 else None

def navigate(g,sb,sc,sl,eb,ec,el):
    if sb not in 'ABCGate' or eb not in 'ABCGate':return None,'Invalid',None,None
    s=g.node_id_map.get((sb,sc,sl)) or f'{sb}-classroom-{sc}@{sl}'
    e=g.node_id_map.get((eb,ec,el)) or f'{eb}-classroom-{ec}@{el}'
    if s not in g.nodes or e not in g.nodes:return None,'Not exist',None,None
    d,p=dijkstra(g,s)
    path=construct_path(p,e)
    if not path:return None,'No path',None,None
    steps=[]
    stairs=set()
    pb=None
    for i,n in enumerate(path):
        inf=g.nodes[n]
        b,t,nm,l=inf['building'],inf['type'],inf['name'],inf['level']
        if t=='stair':
            stairs.add((b,nm,l))
            steps.append(f"{b}{nm}({l})")
        elif t=='classroom':
            steps.append(f"{b}{nm}({l})")
        pb=b
    return path,f"Total: {d[e]:.1f} units"," → ".join(steps),{'start_level':sl,'end_level':el,'path_stairs':stairs,'show_all':False,'path':path,'start_building':sb,'end_building':eb}

def get_classroom_info(data):
    bns=[]
    for b in data:
        if b=='gate':bns.append('Gate')
        elif b.startswith('building'):bns.append(b.replace('building',''))
    info={}
    lvs={}
    for b in data:
        bn='Gate' if b=='gate' else b.replace('building','')
        info[bn]={}
        lvs[bn]=[]
        for lv in data[b]['levels']:
            lvn=lv['name']
            lvs[bn].append(lvn)
            info[bn][lvn]=[c['name'] for c in lv.get('classrooms',[])]
    return bns,lvs,info

def reset_app_state():
    st.session_state['display_options']={'start_level':None,'end_level':None,'path_stairs':set(),'show_all':True,'path':[],'start_building':None,'end_building':None}
    st.session_state['current_path']=None

# --------------------------
# 主程序
# --------------------------
def main():
    if 'page' not in st.session_state:st.session_state.page='welcome'
    if 'display_options' not in st.session_state:st.session_state.display_options={'start_level':None,'end_level':None,'path_stairs':set(),'show_all':True,'path':[],'start_building':None,'end_building':None}
    if 'current_path' not in st.session_state:st.session_state.current_path=None

    # 只有登录页加载背景
    if st.session_state.page == 'welcome':
        def bg(img):
            with open(img,'rb') as f:
                b64=base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <style>
            [data-testid="stAppViewContainer"]{{
                background:url(data:image/jpeg;base64,{b64}) !important;
                background-size:cover !important;
                background-position:center !important;
                background-attachment:fixed !important;
            }}
            h1{{color:white !important; text-align:center !important; margin-top:22vh !important; font-size:50px !important;}}
            div.stButton>button{{
                margin:0 auto !important; display:block !important;
                width:260px !important; height:60px !important;
                font-size:20px !important; border-radius:12px !important;
                background:#4CAF50 !important; color:white !important; border:none !important;
            }}
            </style>
            """,unsafe_allow_html=True)
        bg("background.jpg")

    # 登录页
    if st.session_state.page == 'welcome':
        if 'ws' not in st.session_state:st.session_state.ws=init_google_sheet()
        total=get_total_accesses(st.session_state.ws)
        st.markdown("<h1>Welcome to SCIS Navigation</h1>",unsafe_allow_html=True)
        if st.button("Enter System"):
            update_access_count(st.session_state.ws)
            st.session_state.page='main'
            st.rerun()

    # 导航页（纯白默认）
    else:
        with st.sidebar:
            st.header("📍 Select Locations")
            data=load_school_data_detailed('school_data_detailed.json')
            if not data:st.error("Load failed");return
            bns,lvs,cls=get_classroom_info(data)
            st.subheader("Start")
            sb=st.selectbox("Building",bns,key='sb')
            sl=st.selectbox("Floor",lvs[sb],key='sl')
            sc=st.selectbox("Classroom",cls[sb][sl],key='sc')
            st.subheader("End")
            eb=st.selectbox("Building",bns,key='eb')
            el=st.selectbox("Floor",lvs[eb],key='el')
            ec=st.selectbox("Classroom",cls[eb][el],key='ec')
            st.divider()
            go_btn=st.button("🔍 Find Path",use_container_width=True)
            rst_btn=st.button("🔄 Reset",use_container_width=True)
            back_btn=st.button("🚪 Back",use_container_width=True)
            if rst_btn:reset_app_state();st.rerun()
            if back_btn:reset_app_state();st.session_state.page='welcome';st.rerun()
        st.markdown("<h2>🏫 SCIS Campus Navigation</h2>",unsafe_allow_html=True)
        g=build_navigation_graph(data)
        st.success("✅ Data loaded")
        if go_btn:
            res=navigate(g,sb,sc,sl,eb,ec,el)
            if res[0]:
                path,msg,text,opts=res
                st.success(f"📊 {msg}")
                st.markdown("#### Path")
                st.info(text)
                st.session_state.current_path=path
                st.session_state.display_options=opts
        fig=plot_3d_map(data,g,st.session_state.display_options)[0]
        st.plotly_chart(fig,use_container_width=True)

if __name__=='__main__':
    main()
