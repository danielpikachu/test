import json
import math
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import streamlit as st

# -------------------------- 配置常量 --------------------------
JSON_FILE_PATH = "school_data_detailed.json"
GOOGLE_SHEETS_CREDENTIALS = "credentials.json"  # 替换为你的凭证文件路径
SHEET_NAME = "SchoolPathFinder"  # Google Sheets 表名

# -------------------------- Google Sheets 配置 --------------------------
def init_google_sheets():
    """初始化Google Sheets连接"""
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            GOOGLE_SHEETS_CREDENTIALS, scope
        )
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).sheet1
        return sheet
    except Exception as e:
        st.error(f"Google Sheets 初始化失败: {str(e)}")
        return None

# -------------------------- 数据加载与解析 --------------------------
def load_school_data():
    """加载学校建筑数据"""
    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"未找到 {JSON_FILE_PATH} 文件")
        return None
    except json.JSONDecodeError:
        st.error("JSON文件格式不正确")
        return None

def get_all_locations(data):
    """获取所有可选位置（教室、楼梯、大门）"""
    locations = []
    
    # 遍历所有建筑
    for building_name, building_data in data.items():
        # 处理普通建筑（A/B/C楼）
        if building_name in ["buildingA", "buildingB", "buildingC"]:
            for level in building_data["levels"]:
                level_name = level["name"]
                # 添加教室
                if "classrooms" in level:
                    for classroom in level["classrooms"]:
                        loc_name = f"{building_name[-1]}-{classroom['name']} ({level_name})"
                        locations.append({
                            "name": loc_name,
                            "building": building_name,
                            "level": level_name,
                            "coords": classroom["coordinates"],
                            "type": "classroom"
                        })
                # 添加楼梯
                if "stairs" in level:
                    for stair in level["stairs"]:
                        loc_name = f"{building_name[-1]}-{stair['name']} ({level_name})"
                        locations.append({
                            "name": loc_name,
                            "building": building_name,
                            "level": level_name,
                            "coords": stair["coordinates"],
                            "type": "stair"
                        })
        # 处理大门
        elif building_name == "gate":
            for level in building_data["levels"]:
                level_name = level["name"]
                # 添加大门
                if "classrooms" in level:
                    for gate in level["classrooms"]:
                        loc_name = f"大门-{gate['name']} ({level_name})"
                        locations.append({
                            "name": loc_name,
                            "building": building_name,
                            "level": level_name,
                            "coords": gate["coordinates"],
                            "type": "gate"
                        })
                # 添加大门楼梯
                if "stairs" in level:
                    for stair in level["stairs"]:
                        loc_name = f"大门-{stair['name']} ({level_name})"
                        locations.append({
                            "name": loc_name,
                            "building": building_name,
                            "level": level_name,
                            "coords": stair["coordinates"],
                            "type": "stair"
                        })
    
    return locations

# -------------------------- 路径计算核心函数 --------------------------
def calculate_distance(coord1, coord2):
    """计算3D坐标距离"""
    return math.sqrt(
        (coord1[0] - coord2[0])**2 +
        (coord1[1] - coord2[1])**2 +
        (coord1[2] - coord2[2])**2
    )

def find_stair_path(data, start_building, start_level, end_building, end_level):
    """查找楼梯路径（原有函数，完整保留）"""
    path = []
    stair_connections = {}
    
    # 构建楼梯连接映射
    for building_name, building_data in data.items():
        if "connections" in building_data:
            for conn in building_data["connections"]:
                from_stair, from_level = conn["from"]
                to_stair, to_level = conn["to"]
                
                if building_name not in stair_connections:
                    stair_connections[building_name] = {}
                if from_level not in stair_connections[building_name]:
                    stair_connections[building_name][from_level] = {}
                stair_connections[building_name][from_level][from_stair] = {
                    "to_building": building_name,
                    "to_level": to_level,
                    "to_stair": to_stair
                }
    
    # 处理同建筑楼层切换
    if start_building == end_building:
        current_level = start_level
        path.append(f"从{start_building} {start_level}出发")
        
        # 查找该建筑的楼层顺序
        building_data = data[start_building]
        levels = sorted(building_data["levels"], key=lambda x: x["z"])
        level_names = [l["name"] for l in levels]
        
        start_idx = level_names.index(start_level)
        end_idx = level_names.index(end_level)
        
        # 向上/向下找楼梯
        step = 1 if end_idx > start_idx else -1
        for i in range(start_idx, end_idx, step):
            current_lvl = level_names[i]
            next_lvl = level_names[i+step]
            
            # 找当前楼层的楼梯
            for level_data in building_data["levels"]:
                if level_data["name"] == current_lvl and "stairs" in level_data:
                    stair_name = level_data["stairs"][0]["name"]
                    path.append(f"乘坐{stair_name}前往{next_lvl}")
                    break
        
        path.append(f"到达{end_building} {end_level}")
        return " → ".join(path)
    
    # 处理跨建筑路径（新增大门连接逻辑）
    else:
        # 先找起始建筑到1楼（大门连接层）的路径
        path.append(f"从{start_building} {start_level}出发")
        building_data = data[start_building]
        levels = sorted(building_data["levels"], key=lambda x: x["z"])
        level_names = [l["name"] for l in levels]
        
        # 到1楼
        if start_level != "level1":
            start_idx = level_names.index(start_level)
            level1_idx = level_names.index("level1")
            step = 1 if level1_idx > start_idx else -1
            
            for i in range(start_idx, level1_idx, step):
                current_lvl = level_names[i]
                next_lvl = level_names[i+step]
                
                for level_data in building_data["levels"]:
                    if level_data["name"] == current_lvl and "stairs" in level_data:
                        stair_name = level_data["stairs"][0]["name"]
                        path.append(f"乘坐{stair_name}前往{next_lvl}")
                        break
        
        # 跨建筑连廊（包含大门）
        if start_building != "gate" and end_building != "gate":
            path.append(f"从{start_building} level1 通过连廊前往大门")
            path.append(f"从大门通过连廊前往{end_building} level1")
        elif start_building == "gate":
            path.append(f"从大门通过连廊前往{end_building} level1")
        elif end_building == "gate":
            path.append(f"从{start_building} level1 通过连廊前往大门")
        
        # 从目标建筑1楼到目标楼层
        if end_level != "level1" and end_building != "gate":
            building_data = data[end_building]
            levels = sorted(building_data["levels"], key=lambda x: x["z"])
            level_names = [l["name"] for l in levels]
            
            level1_idx = level_names.index("level1")
            end_idx = level_names.index(end_level)
            step = 1 if end_idx > level1_idx else -1
            
            for i in range(level1_idx, end_idx, step):
                current_lvl = level_names[i]
                next_lvl = level_names[i+step]
                
                for level_data in building_data["levels"]:
                    if level_data["name"] == current_lvl and "stairs" in level_data:
                        stair_name = level_data["stairs"][0]["name"]
                        path.append(f"乘坐{stair_name}前往{next_lvl}")
                        break
        
        path.append(f"到达{end_building} {end_level}")
        return " → ".join(path)

def find_shortest_path(data, start_loc, end_loc):
    """查找最短路径（包含大门）"""
    # 获取起点和终点坐标
    start_coords = start_loc["coords"]
    end_coords = end_loc["coords"]
    
    # 直接距离
    direct_distance = calculate_distance(start_coords, end_coords)
    
    # 楼梯路径
    stair_path = find_stair_path(
        data,
        start_loc["building"],
        start_loc["level"],
        end_loc["building"],
        end_loc["level"]
    )
    
    return {
        "direct_distance": round(direct_distance, 2),
        "stair_path": stair_path,
        "start": start_loc["name"],
        "end": end_loc["name"]
    }

# -------------------------- Streamlit UI 界面 --------------------------
def main():
    # 页面配置
    st.set_page_config(
        page_title="学校建筑路径查找器（含大门）",
        page_icon="🏫",
        layout="wide"
    )
    
    # 标题
    st.title("🏫 学校建筑路径查找器（含大门）")
    st.divider()
    
    # 初始化会话状态（保存当前路径结果）
    if "current_path_result" not in st.session_state:
        st.session_state.current_path_result = None
    
    # 加载数据
    school_data = load_school_data()
    if not school_data:
        st.stop()
    
    # 获取所有位置
    locations = get_all_locations(school_data)
    location_names = [loc["name"] for loc in locations]
    
    # 初始化Google Sheets
    sheet = init_google_sheets()
    
    # 布局：左右两列
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("路径选择")
        
        # 起点选择
        start_name = st.selectbox(
            "选择起点",
            location_names,
            index=0,
            key="start_select"
        )
        
        # 终点选择
        end_name = st.selectbox(
            "选择终点",
            location_names,
            index=1,
            key="end_select"
        )
        
        # 按钮区域
        st.divider()
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            find_btn = st.button("🔍 查找路径", type="primary")
        
        with col_btn2:
            save_btn = st.button(
                "💾 保存到Google Sheets",
                disabled=(st.session_state.current_path_result is None or sheet is None)
            )
        
        with col_btn3:
            clear_btn = st.button("🗑️ 清空结果")
    
    with col2:
        st.subheader("路径结果")
        
        # 结果显示区域
        result_placeholder = st.empty()
        
        # 清空结果
        if clear_btn:
            st.session_state.current_path_result = None
            result_placeholder.empty()
            st.success("结果已清空！")
        
        # 查找路径
        if find_btn:
            if start_name == end_name:
                st.warning("起点和终点不能相同！")
            else:
                try:
                    # 查找对应的位置数据
                    start_loc = next(loc for loc in locations if loc["name"] == start_name)
                    end_loc = next(loc for loc in locations if loc["name"] == end_name)
                    
                    # 查找路径
                    path_result = find_shortest_path(school_data, start_loc, end_loc)
                    st.session_state.current_path_result = path_result
                    
                    # 显示结果
                    with result_placeholder.container():
                        st.info(f"""
                        **起点**: {path_result['start']}
                        **终点**: {path_result['end']}
                        **直线距离**: {path_result['direct_distance']} 单位
                        """)
                        st.subheader("推荐路径")
                        st.write(path_result['stair_path'])
                    
                except Exception as e:
                    st.error(f"查找路径失败: {str(e)}")
        
        # 保存到Google Sheets
        if save_btn and st.session_state.current_path_result and sheet:
            try:
                # 准备数据
                row_data = [
                    st.session_state.current_path_result['start'],
                    st.session_state.current_path_result['end'],
                    st.session_state.current_path_result['direct_distance'],
                    st.session_state.current_path_result['stair_path']
                ]
                
                # 追加到表格
                sheet.append_row(row_data)
                st.success("✅ 结果已保存到Google Sheets！")
                
            except Exception as e:
                st.error(f"保存失败: {str(e)}")

# -------------------------- 主程序 --------------------------
if __name__ == "__main__":
    main()
