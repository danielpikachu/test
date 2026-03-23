import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import streamlit as st
import gspread
from google.oauth2 import service_account
from datetime import datetime

# ========== 全局页面美化 & 原生布局 ==========
st.set_page_config(
    page_title="SCIS Navigation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局隐藏 Streamlit 默认丑元素（最关键美化）
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
.stApp {background:#f7f9fc;}
.css-18e3th9 {padding: 0rem 1rem;}
.stButton > button {border-radius:8px !important; font-weight:500 !important;}
</style>
""", unsafe_allow_html=True)

plt.switch_backend('Agg')

# ========== Google Sheets 静默运行（零提示） ==========
SHEET_NAME = 'Navigation visitors'
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file"
]

def get_credentials():
    try:
        return service_account.Credentials.from_service_account_info(
            st.secrets["google_service_account"], scopes=SCOPE
        )
    except: return None

def init_google_sheet():
    try:
        creds = get_credentials()
        if not creds: return None
        client = gspread.authorize(creds)
        try: sheet = client.open(SHEET_NAME)
        except: sheet = client.create(SHEET_NAME)
        try: ws = sheet.worksheet("Access_Stats")
        except:
            ws = sheet.add_worksheet("Access_Stats", 1000, 3)
            ws.append_row(["Timestamp","Access_Count","Total_Accesses"])
            ws.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),1,1])
        return ws
    except: return None

def update_access_count(ws):
    if not ws: return 0
    try:
        rows = ws.get_all_values()
        total = int(rows[-1][2]) if rows[-1][2].isdigit() else 0
        new_total = total + 1
        ws.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),1,new_total])
        return new_total
    except: return 0

def get_total_accesses(ws):
    if not ws: return 0
    try:
        rows = ws.get_all_values()
        return int(rows[-1][2]) if rows[-1][2].isdigit() else 0
    except: return 0

# ========== 配色体系（更高级、低饱和） ==========
COLORS = {
    'building': {'A':'#cce7ff','B':'#d9f2d9','C':'#ffdddd','Gate':'#fff2b3'},
    'floor_z': {-9:'#666666',-6:'#4488dd',-3:'#66ccff',2:'#44bb77',4:'#33aaaa',7:'#ffaa44',12:'#aa66cc'},
    'corridor_line': {'A':'#66ccff','B':'#228822','C':'#ee7777','Gate':'#cc9900'},
    'corridor_node':'#003366',
    'stair':{
        'Stairs1':'#ff6644','Stairs2':'#44dd66','Stairs3':'#4466ff',
        'Stairs4':'#ee44dd','Stairs5':'#dddd44','StairsB1':'#44dddd',
        'StairsB2':'#ff9933','GateStairs':'#ffcc00'
    },
    'path':'#ee3333',
    'start_marker':'#33cc33','end_marker':'#cc33cc',
    'connect_corridor':'#ffcc00'
}

# ========== 数据加载（静默） ==========
def load_school_data_detailed(filename):
    try:
        with open(filename,'r') as f: return json.load(f)
    except: return None

# ========== 3D 绘图（纯净无提示、自适应、无边距） ==========
def plot_3d_map(school_data, display_options=None):
    fig = plt.figure(figsize=(16,11), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,0.7])
    ax.axis('off')

    if display_options is None:
        display_options={'start_level':None,'end_level':None,'path_stairs':set(),
                         'show_all':True,'path':[],'start_building':None,'end_building':None}

    show_all = display_options['show_all']
    path = display_options['path']
    building_label_positions = {}

    for building_id in school_data:
        if building_id == 'gate': bn = 'Gate'
        elif building_id.startswith('building'): bn = building_id.replace('building','')
        else: continue
        bdata = school_data[building_id]
        max_z = -999
        max_y = -999
        mid_x = 0
        displayed = []

        for level in bdata['levels']:
            lname = level['name']
            z = level['z']
            show = show_all
            if not show_all:
                if bn == 'B':
                    show = any((bn,s,lname) in display_options['path_stairs'] for s in ['StairsB1','StairsB2'])
                    show = show or ((display_options['start_building']=='B' or display_options['end_building']=='B') and lname=='level1')
                elif bn == 'Gate':
                    show = (display_options['start_building']=='Gate' or display_options['end_building']=='Gate')
                else:
                    show = (lname == display_options['start_level']) or (lname == display_options['end_level'])
            if show:
                displayed.append(level)
                if z>max_z: max_z=z
                fp = level['floorPlane']
                if fp['maxY']>max_y:
                    max_y=fp['maxY']
                    mid_x=(fp['minX']+fp['maxX'])/2

            bc = COLORS['floor_z'].get(z,'#888888')
            fc = COLORS['building'][bn]
            if show:
                fp = level['floorPlane']
                pts = [[fp['minX'],fp['minY'],z],[fp['maxX'],fp['minY'],z],
                       [fp['maxX'],fp['maxY'],z],[fp['minX'],fp['maxY'],z],[fp['minX'],fp['minY'],z]]
                ax.plot([p[0] for p in pts],[p[1] for p in pts],[p[2] for p in pts],color=bc,lw=3)
                ax.plot_trisurf([p[0] for p in pts[:-1]],
                                [p[1] for p in pts[:-1]],
                                [p[2] for p in pts[:-1]],color=fc,alpha=0.35)

                for corr in level['corridors']:
                    px=[p[0] for p in corr['points']]
                    py=[p[1] for p in corr['points']]
                    pz=[p[2] for p in corr['points']]
                    if 'connectToBuilding' in corr.get('name',''):
                        clr=COLORS['connect_corridor']
                        lw=10
                    else:
                        clr=COLORS['corridor_line'][bn]
                        lw=6
                    ax.plot(px,py,pz,color=clr,lw=lw,alpha=0.8)

                for room in level['classrooms']:
                    x,y,_=room['coordinates']
                    ax.text(x,y,z,room['name'],fontsize=9)

                for stair in level['stairs']:
                    flag=(bn,stair['name'],lname) in display_options['path_stairs']
                    x,y,_=stair['coordinates']
                    sc=COLORS['stair'][stair['name']]
                    ax.scatter(x,y,z,color=sc,s=650 if flag else 450,marker='^')

        if displayed:
            if bn == 'B': ypos = max_y - 2
            elif bn == 'Gate': ypos = max_y + 5
            else: ypos = max_y + 2
            building_label_positions[bn] = (mid_x, ypos, max_z + 1)

    for bn,(x,y,z) in building_label_positions.items():
        box = dict(boxstyle="round,pad=0.4", edgecolor="#222", facecolor=COLORS['building'][bn], alpha=0.75)
        ax.text(x,y,z,bn,fontsize=22,weight='bold',ha='center',bbox=box)

    if path and not show_all:
        xs,ys,zs = [],[],[]
        for nid in path:
            crd = graph.nodes[nid]['coordinates']
            xs.append(crd[0]); ys.append(crd[1]); zs.append(crd[2])
        ax.plot(xs,ys,zs,color=COLORS['path'],lw=7)
        ax.scatter(xs[0],ys[0],zs[0],color=COLORS['start_marker'],s=900,marker='*')
        ax.scatter(xs[-1],ys[-1],zs[-1],color=COLORS['end_marker'],s=900,marker='*')

    return fig,ax

# ========== 图结构 & 路径（静默零提示） ==========
class Graph:
    def __init__(self):
        self.nodes = {}
        self.node_id_map = {}
    def add_node(self,bid,ntype,name,level,crd):
        bn = 'Gate' if bid=='gate' else bid.replace('building','')
        nid = f"{bn}-{ntype}-{name}@{level}"
        self.nodes[nid] = {'building':bn,'type':ntype,'name':name,'level':level,'coordinates':crd,'neighbors':{}}
        self.node_id_map[(bid,ntype,name,level)] = nid
        if ntype=='classroom': self.node_id_map[(bn,name,level)] = nid
        return nid
    def add_edge(self,a,b,w):
        if a in self.nodes and b in self.nodes:
            self.nodes[a]['neighbors'][b] = w
            self.nodes[b]['neighbors'][a] = w

def euclidean_distance(c1,c2,p=15):
    import math
    base = math.sqrt(sum((a-b)**2 for a,b in zip(c1,c2)))
    floor = abs(c1[2]-c2[2])
    return base + floor*p

def get_direction_between_nodes(g,n1,n2):
    a,b = g.nodes[n1],g.nodes[n2]
    if a['type']=='stair' and b['type']=='stair':
        return '往上' if b['coordinates'][2]>a['coordinates'][2] else '往下'
    dx = b['coordinates'][0]-a['coordinates'][0]
    dy = b['coordinates'][1]-a['coordinates'][1]
    th = 0.12
    if dy>th: return '向前'
    if dy<-th: return '向后'
    if dx>th: return '向右'
    if dx<-th: return '向左'
    return ''

def build_navigation_graph(school_data):
    g = Graph()
    for bid in school_data:
        if not (bid.startswith('building') or bid=='gate'): continue
        bdata = school_data[bid]
        for lev in bdata['levels']:
            lname = lev['name']
            for room in lev['classrooms']:
                g.add_node(bid,'classroom',room['name'],lname,room['coordinates'])
            for stair in lev['stairs']:
                g.add_node(bid,'stair',stair['name'],lname,stair['coordinates'])
            for idx,corr in enumerate(lev['corridors']):
                cname = corr.get('name',f'corr{idx}')
                for pi,pt in enumerate(corr['points']):
                    g.add_node(bid,'corridor',f"{cname}-p{pi}",lname,pt)
    return g

def dijkstra(g,start):
    dist = {n:float('inf') for n in g.nodes}
    dist[start]=0
    prev = {n:None for n in g.nodes}
    nodes = set(g.nodes)
    while nodes:
        u = min(nodes,key=lambda x:dist[x])
        nodes.remove(u)
        if dist[u]==float('inf'): break
        for v,w in g.nodes[u]['neighbors'].items():
            if dist[v]>dist[u]+w:
                dist[v]=dist[u]+w
                prev[v]=u
    return dist,prev

def construct_path(prev,end):
    path=[]
    cur=end
    while cur:
        path.insert(0,cur)
        cur=prev[cur]
    return path if len(path)>1 else None

def navigate(g,sb,sr,sl,eb,er,el):
    try:
        sk=(sb,sr,sl); ek=(eb,er,el)
        sn=g.node_id_map.get(sk)
        en=g.node_id_map.get(ek)
        if not sn or not en: return None,"","",{}
        dist,prev=dijkstra(g,sn)
        path=construct_path(prev,en)
        if not path: return None,"","",{}
        simple=[]; stairs=set(); preb=None
        for i,nid in enumerate(path):
            nd=g.nodes[nid]
            if nd['type'] in ['classroom','stair']:
                desc=f"{nd['building']}{nd['name']}({nd['level']})"
                if i<len(path)-1: desc+=get_direction_between_nodes(g,nid,path[i+1])
                simple.append(desc)
            if nd['type']=='stair': stairs.add((nd['building'],nd['name'],nd['level']))
            preb=nd['building']
        text=" → ".join(simple)
        opt={'start_level':sl,'end_level':el,'path_stairs':stairs,'show_all':False,'path':path,'start_building':sb,'end_building':eb}
        return path,f"{dist[en]:.2f}",text,opt
    except: return None,"","",{}

# ========== 获取教室列表 ==========
def get_classroom_info(data):
    builds=[]
    for b in data:
        if b=='gate': builds.append('Gate')
        elif b.startswith('building'): builds.append(b.replace('building',''))
    lev_dict,room_dict={},{}
    for bid in data:
        bn='Gate' if bid=='gate' else bid.replace('building','')
        levs,rdict=[],{}
        for lev in data[bid]['levels']:
            ln=lev['name']
            levs.append(ln)
            rdict[ln]=[r['name'] for r in lev['classrooms']]
        lev_dict[bn]=levs
        room_dict[bn]=rdict
    return builds,lev_dict,room_dict

def reset_app_state():
    st.session_state['display_options']={'start_level':None,'end_level':None,'path_stairs':set(),'show_all':True,'path':[]}
    st.session_state['current_path']=None

# ========== 主界面（极简高级风） ==========
def main():
    if 'page' not in st.session_state: st.session_state['page']='welcome'
    if 'display_options' not in st.session_state: reset_app_state()
    if 'worksheet' not in st.session_state: st.session_state['worksheet']=init_google_sheet()

    if st.session_state['page']=='welcome':
        total=get_total_accesses(st.session_state['worksheet'])
        st.markdown("""
        <div style="text-align:center;padding-top:120px;">
        <h1 style="font-size:42px;color:#2c3e50;">SCIS Campus Navigation</h1>
        <p style="font-size:16px;color:#666;">3D Intelligent Route Planning System</p>
        </div>
        """,unsafe_allow_html=True)
        if st.button("Enter Navigation System",use_container_width=True):
            update_access_count(st.session_state['worksheet'])
            st.session_state['page']='main'
            st.rerun()
        st.caption(f"Total Visits · {total}")

    else:
        with st.sidebar:
            st.title("📍 Navigation Panel")
            data=load_school_data_detailed('school_data_detailed.json')
            builds,levs_dict,rooms_dict=get_classroom_info(data)
            st.subheader("Start")
            sb=st.selectbox("Building",builds)
            sl=st.selectbox("Floor",levs_dict[sb])
            sr=st.selectbox("Room",rooms_dict[sb][sl])
            st.subheader("Destination")
            eb=st.selectbox("Building",builds)
            el=st.selectbox("Floor",levs_dict[eb])
            er=st.selectbox("Room",rooms_dict[eb][el])
            st.divider()
            go=st.button("Find Shortest Path",use_container_width=True)
            rst=st.button("Reset View",use_container_width=True)
            out=st.button("Back to Home",use_container_width=True)
            if rst: reset_app_state(); st.rerun()
            if out: reset_app_state(); st.session_state['page']='welcome'; st.rerun()

        st.title("🏫 SCIS 3D Campus Navigation")
        global graph
        graph=build_navigation_graph(data)

        if go:
            path,disttxt,pathtxt,opt=navigate(graph,sb,sr,sl,eb,er,el)
            if path:
                st.success(f"Route Found · Distance {disttxt}")
                st.info(pathtxt)
                st.session_state['current_path']=path
                st.session_state['display_options']=opt

        fig,_=plot_3d_map(data,st.session_state['display_options'])
        st.pyplot(fig,use_container_width=True)

        st.markdown("""
        <div style="position:fixed;right:24px;bottom:16px;font-size:13px;color:#888;">
        Created By DANIEL HAN
        </div>
        """,unsafe_allow_html=True)

if __name__=="__main__":
    main()
