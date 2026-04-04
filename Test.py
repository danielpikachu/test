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
    except Exception:
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
    except Exception:
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
    except Exception:
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


# --------------------------
# Reset state
# --------------------------
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


# --------------------------
# 全局样式
# --------------------------
st.markdown("""
<style>
div.stMarkdown, div.stAlert, div.element-container {
    margin-top: 0px !important;
    margin-bottom: 0px !important;
    padding-top: 0px !important;
    padding-bottom: 0px !important;
}
div.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 0rem !important;
}
::-webkit-scrollbar {display: none !important;}
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem !important;
}
section[data-testid="stSidebar"] button[data-testid="stSidebarCollapseButton"] {
    height: 1.5rem !important;
    min-height: 1.5rem !important;
    margin: 0 0 -5px 0 !important;
}
section[data-testid="stSidebar"] h2 {
    margin-top: -10px !important;
    padding-top: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# --------------------------
# Classroom Info
# --------------------------
def get_classroom_info(school_data):
    try:
        buildings = [b for b in school_data.keys() if b.startswith('building') or b == 'gate']
        building_names = []

        for b in buildings:
            if b == 'gate':
                building_names.append('Gate')
            else:
                building_names.append(b.replace('building', ''))

        classrooms_by_building = {}
        levels_by_building = {}

        for building_id in buildings:
            if building_id == 'gate':
                building_name = 'Gate'
            else:
                building_name = building_id.replace('building', '')

            building_data = school_data[building_id]

            levels = []
            classrooms_by_level = {}

            for level in building_data['levels']:
                level_name = level['name']
                levels.append(level_name)
                classrooms = [c['name'] for c in level['classrooms']]
                classrooms_by_level[level_name] = classrooms

            levels_by_building[building_name] = levels
            classrooms_by_building[building_name] = classrooms_by_level

        return building_names, levels_by_building, classrooms_by_building
    except Exception as e:
        st.error(f"Failed to retrieve classroom information: {str(e)}")
        return [], {}, {}


# --------------------------
# MAIN APP
# --------------------------
def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'welcome'
    if 'display_options' not in st.session_state:
        reset_app_state()
    if 'current_path' not in st.session_state:
        st.session_state['current_path'] = None

    # Welcome Page
    if st.session_state['page'] == 'welcome':
        if 'worksheet' not in st.session_state:
            st.session_state['worksheet'] = init_google_sheet()

        st.markdown("""
        <div style="text-align:center; margin-top:20vh;">
            <h1 style="color:white; font-size:70px;">SCIS Navigation</h1>
            <p style="color:#ccc; font-size:25px;">Navigate the school with ease</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Enter", type="primary", use_container_width=True):
            update_access_count(st.session_state['worksheet'])
            st.session_state['page'] = 'main'
            st.rerun()

    # Main Page
    else:
        with st.sidebar:
            st.header("📍 Select Locations")

            school_data = load_school_data_detailed('school_data_detailed.json')
            if school_data is None:
                st.error("Failed to load school data!")
                return

            building_names, levels_by_building, classrooms_by_building = get_classroom_info(school_data)

            st.subheader("Start Point")
            start_building = st.selectbox("Building", building_names)
            start_level = st.selectbox("Floor", levels_by_building[start_building])
            start_classroom = st.selectbox("Classroom", classrooms_by_building[start_building][start_level])

            st.subheader("End Point")
            end_building = st.selectbox("Building", building_names)
            end_level = st.selectbox("Floor", levels_by_building[end_building])
            end_classroom = st.selectbox("Classroom", classrooms_by_building[end_building][end_level])

            nav_button = st.button("🔍 Find Shortest Path")
            reset_button = st.button("🔄 Reset View")

            if reset_button:
                reset_app_state()
                st.rerun()

        st.success("✅ Campus data loaded successfully!")

        if nav_button:
            st.info(f"Route from {start_building}-{start_classroom} to {end_building}-{end_classroom}")

if __name__ == "__main__":
    main()
