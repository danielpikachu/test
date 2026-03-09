import streamlit as st
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ===================== 原有Google Sheets配置（完全保留）=====================
# Google Sheets 授权配置
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets"
]
# 请替换为你的实际凭证文件路径
CREDS_FILE = "google_credentials.json"  
try:
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    client = gspread.authorize(creds)
    # 替换为你的实际表格名称
    SHEET_NAME = "SchoolBuildingNavigation"  
    sheet = client.open(SHEET_NAME).worksheet("main")  # 替换为你的工作表名称
except Exception as e:
    st.error(f"Google Sheets 初始化失败: {str(e)}")
    client = None
    sheet = None

# ===================== 原有数据加载函数（新增大门口适配）=====================
def load_school_data() -> Dict:
    """加载校园建筑数据（保留原有逻辑，适配大门口节点）"""
    try:
        with open("school_data_detailed.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        # 验证数据结构（保留原有验证逻辑）
        required_keys = ["buildingA", "buildingB", "buildingC", "gate"]  # 新增gate
        for key in required_keys:
            if key not in data:
                st.warning(f"数据文件中缺少{key}节点，请检查JSON结构")
        return data
    except FileNotFoundError:
        st.error("错误：未找到school_data_detailed.json文件，请确认文件路径正确")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"错误：JSON文件格式解析失败 - {str(e)}")
        return {}
    except Exception as e:
        st.error(f"数据加载异常: {str(e)}")
        return {}

# ===================== 原有路径计算函数（完全保留）=====================
def find_stair_path(
    data: Dict, 
    start_building: str, 
    start_level: str, 
    end_building: str, 
    end_level: str,
    start_point: Optional[str] = None,
    end_point: Optional[str] = None
) -> List[Dict]:
    """
    查找包含楼梯的路径（完全保留原有100%逻辑）
    :param data: 校园数据字典
    :param start_building: 起始建筑（新增支持"gate"）
    :param start_level: 起始楼层
    :param end_building: 目标建筑（新增支持"gate"）
    :param end_level: 目标楼层
    :param start_point: 起始点（如教室/楼梯）
    :param end_point: 目标点（如教室/楼梯）
    :return: 路径列表
    """
    # ===== 以下为原有完整逻辑（仅适配gate节点）=====
    path = []
    visited = set()
    # 构建节点唯一标识
    def get_node_id(building: str, level: str, point: str = "") -> str:
        return f"{building}_{level}_{point}" if point else f"{building}_{level}"
    
    # 递归查找路径（原有核心逻辑完全保留）
    def dfs(current_building: str, current_level: str, current_point: str, path_so_far: List):
        current_id = get_node_id(current_building, current_level, current_point)
        if current_id in visited:
            return False
        visited.add(current_id)
        
        # 终止条件
        if (current_building == end_building and 
            current_level == end_level and 
            (not end_point or current_point == end_point)):
            path.extend(path_so_far + [{"building": current_building, "level": current_level, "point": current_point}])
            return True
        
        # 获取当前节点的连接关系（适配gate节点）
        if current_building not in data:
            return False
        building_data = data[current_building]
        if "connections" not in building_data:
            return False
        
        # 遍历所有连接（原有逻辑完全保留）
        for conn in building_data["connections"]:
            from_node = conn["from"]
            to_node = conn["to"]
            
            # 匹配当前节点
            if (from_node[0] == current_point and from_node[1] == current_level) or \
               (current_point is None and from_node[1] == current_level):
                # 递归下一个节点
                next_building = current_building
                next_level = to_node[1]
                next_point = to_node[0]
                
                # 跨建筑连接（适配gate到ABC的连廊）
                if "gateTo" in from_node[0] or "gateTo" in to_node[0]:
                    if "A" in to_node[0]:
                        next_building = "buildingA"
                    elif "B" in to_node[0]:
                        next_building = "buildingB"
                    elif "C" in to_node[0]:
                        next_building = "buildingC"
                    elif "gate" in to_node[0]:
                        next_building = "gate"
                
                if dfs(next_building, next_level, next_point, path_so_far + [
                    {"building": current_building, "level": current_level, "point": current_point}
                ]):
                    return True
        return False
    
    # 启动DFS（适配gate节点）
    dfs(start_building, start_level, start_point if start_point else "", [])
    
    # 路径格式化（原有逻辑保留）
    formatted_path = []
    for idx, step in enumerate(path):
        formatted_path.append({
            "step": idx + 1,
            "building": step["building"],
            "level": step["level"],
            "location": step["point"] if step["point"] else "主走廊",
            "description": f"从{step['building']} {step['level']} {step['point'] if step['point'] else '主走廊'}出发" if idx == 0 else
                           f"到达{step['building']} {step['level']} {step['point'] if step['point'] else '主走廊'}"
        })
    
    return formatted_path if formatted_path else [{"step": 1, "building": "无", "level": "无", "location": "无", "description": "未找到有效路径"}]

# ===================== 原有辅助函数（完全保留，新增gate适配）=====================
def get_building_levels(data: Dict, building: str) -> List[str]:
    """获取建筑的所有楼层（适配gate节点）"""
    if building not in data:
        return []
    return [level["name"] for level in data[building]["levels"]]

def get_level_points(data: Dict, building: str, level: str) -> List[str]:
    """获取楼层内的所有点位（教室/楼梯/走廊，适配gate节点）"""
    if building not in data:
        return []
    building_data = data[building]
    for lvl in building_data["levels"]:
        if lvl["name"] == level:
            points = []
            # 收集教室
            if "classrooms" in lvl:
                points.extend([cls["name"] for cls in lvl["classrooms"]])
            # 收集楼梯
            if "stairs" in lvl:
                points.extend([st["name"] for st in lvl["stairs"]])
            # 收集走廊
            if "corridors" in lvl:
                points.extend([corr["name"] for corr in lvl["corridors"] if "name" in corr])
            return sorted(list(set(points)))
    return []

def sync_data_to_gsheets(data: Dict) -> bool:
    """同步数据到Google Sheets（原有逻辑完全保留）"""
    if not client or not sheet:
        st.error("Google Sheets 未初始化，无法同步")
        return False
    try:
        # 清空原有数据（保留表头）
        sheet.clear()
        headers = ["建筑名称", "楼层", "点位类型", "点位名称", "坐标X", "坐标Y", "坐标Z", "尺寸X", "尺寸Y"]
        sheet.update_cell(1, 1, "|".join(headers))
        
        # 写入数据（适配gate节点）
        row = 2
        buildings = ["buildingA", "buildingB", "buildingC", "gate"]  # 新增gate
        for building in buildings:
            if building not in data:
                continue
            building_name = {
                "buildingA": "A楼",
                "buildingB": "B楼",
                "buildingC": "C楼",
                "gate": "大门口"  # 新增映射
            }.get(building, building)
            
            for level in data[building]["levels"]:
                level_name = level["name"]
                z_coord = level["z"]
                
                # 写入教室数据
                if "classrooms" in level:
                    for cls in level["classrooms"]:
                        coords = cls["coordinates"]
                        size = cls.get("size", [0, 0])
                        sheet.update_cell(row, 1, building_name)
                        sheet.update_cell(row, 2, level_name)
                        sheet.update_cell(row, 3, "教室")
                        sheet.update_cell(row, 4, cls["name"])
                        sheet.update_cell(row, 5, coords[0])
                        sheet.update_cell(row, 6, coords[1])
                        sheet.update_cell(row, 7, coords[2])
                        sheet.update_cell(row, 8, size[0])
                        sheet.update_cell(row, 9, size[1])
                        row += 1
                
                # 写入楼梯数据
                if "stairs" in level:
                    for stair in level["stairs"]:
                        coords = stair["coordinates"]
                        sheet.update_cell(row, 1, building_name)
                        sheet.update_cell(row, 2, level_name)
                        sheet.update_cell(row, 3, "楼梯")
                        sheet.update_cell(row, 4, stair["name"])
                        sheet.update_cell(row, 5, coords[0])
                        sheet.update_cell(row, 6, coords[1])
                        sheet.update_cell(row, 7, coords[2])
                        sheet.update_cell(row, 8, 0)
                        sheet.update_cell(row, 9, 0)
                        row += 1
                
                # 写入走廊数据
                if "corridors" in level:
                    for corr in level["corridors"]:
                        corr_name = corr.get("name", "无名走廊")
                        sheet.update_cell(row, 1, building_name)
                        sheet.update_cell(row, 2, level_name)
                        sheet.update_cell(row, 3, "走廊")
                        sheet.update_cell(row, 4, corr_name)
                        sheet.update_cell(row, 5, "")
                        sheet.update_cell(row, 6, "")
                        sheet.update_cell(row, 7, z_coord)
                        sheet.update_cell(row, 8, 0)
                        sheet.update_cell(row, 9, 0)
                        row += 1
        
        st.success("数据已成功同步到Google Sheets！")
        return True
    except Exception as e:
        st.error(f"同步Google Sheets失败: {str(e)}")
        return False

# ===================== 原有3D可视化函数（完全保留，适配gate）=====================
def plot_3d_navigation(data: Dict, path: List[Dict]) -> None:
    """3D路径可视化（原有逻辑完全保留，新增大门口渲染）"""
    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])
    
    # 定义颜色映射（新增gate颜色）
    color_map = {
        "buildingA": "#FF6B6B",
        "buildingB": "#4ECDC4",
        "buildingC": "#45B7D1",
        "gate": "#8B4513"  # 大门口褐色
    }
    
    # 绘制建筑轮廓（原有逻辑）
    for building in ["buildingA", "buildingB", "buildingC", "gate"]:
        if building not in data:
            continue
        building_data = data[building]
        for level in building_data["levels"]:
            floor_plane = level.get("floorPlane", {})
            min_x = floor_plane.get("minX", 0)
            max_x = floor_plane.get("maxX", 0)
            min_y = floor_plane.get("minY", 0)
            max_y = floor_plane.get("maxY", 0)
            z = level.get("z", 0)
            
            # 绘制楼层平面
            fig.add_trace(go.Scatter3d(
                x=[min_x, max_x, max_x, min_x, min_x],
                y=[min_y, min_y, max_y, max_y, min_y],
                z=[z, z, z, z, z],
                mode="lines",
                name=f"{building}_{level['name']}",
                line=dict(color=color_map[building], width=2),
                opacity=0.5
            ))
    
    # 绘制连廊（适配大门口到ABC的连廊）
    for building in ["buildingA", "buildingB", "buildingC", "gate"]:
        if building not in data:
            continue
        building_data = data[building]
        for level in building_data["levels"]:
            if "corridors" not in level:
                continue
            for corr in level["corridors"]:
                if "points" not in corr:
                    continue
                points = corr["points"]
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                z = [p[2] for p in points]
                corr_color = corr.get("style", {}).get("color", color_map[building])
                line_type = corr.get("style", {}).get("lineType", "solid")
                
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    name=corr.get("name", f"{building}走廊"),
                    line=dict(
                        color=corr_color,
                        width=3,
                        dash=line_type
                    )
                ))
    
    # 绘制路径（原有逻辑）
    if path and path[0]["building"] != "无":
        path_x = []
        path_y = []
        path_z = []
        for step in path:
            building = step["building"]
            level = step["level"]
            point = step["location"]
            
            # 获取点位坐标
            coords = None
            if building in data:
                for lvl in data[building]["levels"]:
                    if lvl["name"] == level:
                        # 查找教室坐标
                        if "classrooms" in lvl:
                            for cls in lvl["classrooms"]:
                                if cls["name"] == point:
                                    coords = cls["coordinates"]
                                    break
                        # 查找楼梯坐标
                        if not coords and "stairs" in lvl:
                            for stair in lvl["stairs"]:
                                if stair["name"] == point:
                                    coords = stair["coordinates"]
                                    break
                        # 默认走廊坐标
                        if not coords and "corridors" in lvl and lvl["corridors"]:
                            coords = lvl["corridors"][0]["points"][0]
                        break
            
            if coords:
                path_x.append(coords[0])
                path_y.append(coords[1])
                path_z.append(coords[2])
        
        if path_x:
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode="lines+markers",
                name="导航路径",
                line=dict(color="red", width=5),
                marker=dict(size=8, color="red")
            ))
    
    # 布局设置（原有逻辑）
    fig.update_layout(
        title="校园3D导航视图",
        width=1000,
        height=800,
        scene=dict(
            xaxis_title="X坐标",
            yaxis_title="Y坐标",
            zaxis_title="Z坐标",
            xaxis=dict(range=[-50, 120]),  # 适配大门口X=-30
            yaxis=dict(range=[-90, 80]),   # 适配大门口Y=-30
            zaxis=dict(range=[-10, 20])
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ===================== 原有UI逻辑（完全保留，新增gate选项）=====================
def main():
    """主函数（Streamlit UI，完全保留原有布局，新增大门口选项）"""
    st.set_page_config(
        page_title="校园导航系统",
        page_icon="🏫",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏫 校园建筑导航系统")
    st.divider()
    
    # 加载数据
    school_data = load_school_data()
    if not school_data:
        st.stop()
    
    # 侧边栏（原有布局完全保留）
    with st.sidebar:
        st.header("导航设置")
        
        # 建筑选择（新增大门口）
        building_options = {
            "buildingA": "A楼",
            "buildingB": "B楼",
            "buildingC": "C楼",
            "gate": "大门口"  # 新增选项
        }
        start_building = st.selectbox(
            "起始建筑",
            options=list(building_options.keys()),
            format_func=lambda x: building_options[x],
            index=3  # 默认选中大门口
        )
        end_building = st.selectbox(
            "目标建筑",
            options=list(building_options.keys()),
            format_func=lambda x: building_options[x],
            index=0  # 默认选中A楼
        )
        
        # 楼层选择（适配大门口）
        start_levels = get_building_levels(school_data, start_building)
        start_level = st.selectbox("起始楼层", start_levels) if start_levels else st.selectbox("起始楼层", ["level1"])
        
        end_levels = get_building_levels(school_data, end_building)
        end_level = st.selectbox("目标楼层", end_levels) if end_levels else st.selectbox("目标楼层", ["level1"])
        
        # 点位选择（适配大门口）
        start_points = get_level_points(school_data, start_building, start_level)
        start_point = st.selectbox("起始点位（可选）", [""] + start_points)
        
        end_points = get_level_points(school_data, end_building, end_level)
        end_point = st.selectbox("目标点位（可选）", [""] + end_points)
        
        # 功能按钮（原有按钮完全保留）
        st.divider()
        query_btn = st.button("🔍 查询路径", type="primary")
        sync_btn = st.button("📤 同步到Google Sheets")
        reset_btn = st.button("🔄 重置设置")
        
        # 重置逻辑
        if reset_btn:
            st.rerun()
    
    # 主内容区（原有布局完全保留）
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("📝 导航结果")
        if query_btn:
            # 调用原有路径查询函数
            path_result = find_stair_path(
                data=school_data,
                start_building=start_building,
                start_level=start_level,
                end_building=end_building,
                end_level=end_level,
                start_point=start_point if start_point else None,
                end_point=end_point if end_point else None
            )
            
            # 展示路径（原有格式）
            if path_result and path_result[0]["building"] != "无":
                for step in path_result:
                    with st.container(border=True):
                        st.write(f"**第{step['step']}步**：{step['description']}")
            else:
                st.warning("未找到有效路径，请检查选择的建筑/楼层/点位")
        
        # 同步数据逻辑
        if sync_btn:
            sync_data_to_gsheets(school_data)
    
    with col2:
        st.subheader("🗺️ 3D可视化视图")
        # 绘制3D视图（适配大门口）
        plot_3d_navigation(school_data, path_result if 'path_result' in locals() else [])
    
    # 数据展示区（原有逻辑，新增大门口数据）
    st.divider()
    with st.expander("📊 建筑数据详情", expanded=False):
        # 转换为DataFrame展示（适配大门口）
        all_data = []
        for building in ["buildingA", "buildingB", "buildingC", "gate"]:
            if building not in school_data:
                continue
            building_data = school_data[building]
            for level in building_data["levels"]:
                level_name = level["name"]
                z = level["z"]
                # 教室数据
                for cls in level.get("classrooms", []):
                    all_data.append({
                        "建筑": building_options[building],
                        "楼层": level_name,
                        "类型": "教室",
                        "名称": cls["name"],
                        "坐标(X,Y,Z)": f"{cls['coordinates'][0]}, {cls['coordinates'][1]}, {cls['coordinates'][2]}",
                        "尺寸(X,Y)": f"{cls.get('size', [0,0])[0]}, {cls.get('size', [0,0])[1]}"
                    })
                # 楼梯数据
                for stair in level.get("stairs", []):
                    all_data.append({
                        "建筑": building_options[building],
                        "楼层": level_name,
                        "类型": "楼梯",
                        "名称": stair["name"],
                        "坐标(X,Y,Z)": f"{stair['coordinates'][0]}, {stair['coordinates'][1]}, {stair['coordinates'][2]}",
                        "尺寸(X,Y)": "0, 0"
                    })
                # 走廊数据
                for corr in level.get("corridors", []):
                    all_data.append({
                        "建筑": building_options[building],
                        "楼层": level_name,
                        "类型": "走廊",
                        "名称": corr.get("name", "无名走廊"),
                        "坐标(X,Y,Z)": "多段路径",
                        "尺寸(X,Y)": "0, 0"
                    })
        
        # 展示数据表格
        df = pd.DataFrame(all_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

# ===================== 程序入口（完全保留）=====================
if __name__ == "__main__":
    main()
