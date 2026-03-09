import streamlit as st
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
import time
import os
from datetime import datetime
warnings.filterwarnings("ignore")

# ===================== 全局配置（原有完整配置）=====================
# 系统基础配置
APP_TITLE = "校园建筑智能导航系统V2.0"
APP_ICON = "🏫"
DEBUG_MODE = False
DATA_FILE_PATH = "school_data_detailed.json"
CREDS_FILE_PATH = "google_credentials.json"
SHEET_MAIN_NAME = "SchoolBuildingNavigation"
SHEET_BACKUP_NAME = "NavigationBackup"
LOG_FILE = "app_logs.txt"

# Plotly库兼容处理（修复导入报错）
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    PLOTLY_CONFIG = {
        'displayModeBar': True,
        'responsive': True,
        'editable': False
    }
except ImportError:
    st.warning("⚠️ 未检测到Plotly库，3D可视化功能将禁用！请执行：pip install plotly")
    PLOTLY_AVAILABLE = False
    # 定义空对象避免代码崩溃
    class DummyPlotly:
        def __getattr__(self, name):
            return lambda **kwargs: None
    go = DummyPlotly()
    px = DummyPlotly()
    make_subplots = lambda **kwargs: DummyPlotly()
    PLOTLY_CONFIG = {}

# ===================== 日志系统（原有完整逻辑）=====================
def write_log(message: str, level: str = "INFO") -> None:
    """写入系统日志（原有完整逻辑）"""
    log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{log_time}] [{level}] {message}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
        if DEBUG_MODE:
            print(log_entry.strip())
    except Exception as e:
        if DEBUG_MODE:
            print(f"日志写入失败: {str(e)}")

# ===================== Google Sheets 完整配置（原有逻辑）=====================
# 授权范围配置
SCOPE = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file"
]

# 初始化Google Sheets客户端
def init_gsheets_client() -> tuple:
    """初始化Google Sheets客户端（原有完整逻辑）"""
    client = None
    main_sheet = None
    backup_sheet = None
    try:
        # 检查凭证文件
        if not os.path.exists(CREDS_FILE_PATH):
            write_log(f"Google凭证文件不存在: {CREDS_FILE_PATH}", "ERROR")
            st.error(f"❌ 找不到Google凭证文件：{CREDS_FILE_PATH}")
            return None, None, None
        
        # 加载凭证
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE_PATH, SCOPE)
        client = gspread.authorize(creds)
        write_log("Google Sheets客户端初始化成功", "INFO")
        
        # 打开主工作表
        main_sheet = client.open(SHEET_MAIN_NAME).worksheet("main")
        write_log(f"成功打开主工作表: {SHEET_MAIN_NAME}", "INFO")
        
        # 打开备份工作表
        try:
            backup_sheet = client.open(SHEET_BACKUP_NAME).worksheet("backup")
            write_log(f"成功打开备份工作表: {SHEET_BACKUP_NAME}", "INFO")
        except Exception as e:
            write_log(f"备份工作表不存在，将创建: {str(e)}", "WARNING")
            backup_sheet = None
            
    except Exception as e:
        error_msg = f"Google Sheets初始化失败: {str(e)}"
        write_log(error_msg, "ERROR")
        st.error(f"❌ {error_msg}")
    
    return client, main_sheet, backup_sheet

# 初始化GS客户端
gs_client, gs_main_sheet, gs_backup_sheet = init_gsheets_client()

# ===================== 数据加载与验证（原有完整逻辑）=====================
def validate_json_structure(data: Dict) -> bool:
    """验证JSON数据结构完整性（原有完整逻辑）"""
    required_root_keys = ["buildingA", "buildingB", "buildingC", "gate"]  # 新增gate验证
    required_level_keys = ["name", "z", "floorPlane", "classrooms", "stairs", "corridors"]
    required_connection_keys = ["from", "to"]
    
    # 验证根节点
    for key in required_root_keys:
        if key not in data:
            write_log(f"JSON结构缺失根节点: {key}", "ERROR")
            st.warning(f"⚠️ 数据文件缺失关键节点：{key}")
            return False
    
    # 验证各建筑层级结构
    for building, building_data in data.items():
        if "levels" not in building_data:
            write_log(f"{building} 缺失levels节点", "ERROR")
            st.warning(f"⚠️ {building} 缺失楼层数据")
            return False
        
        # 验证楼层结构
        for level in building_data["levels"]:
            for lkey in required_level_keys:
                if lkey not in level and lkey != "classrooms":  # classrooms可选
                    write_log(f"{building} {level['name']} 缺失{lkey}节点", "WARNING")
        
        # 验证连接关系
        if "connections" in building_data:
            for conn in building_data["connections"]:
                for ckey in required_connection_keys:
                    if ckey not in conn:
                        write_log(f"{building} 连接关系缺失{ckey}节点", "ERROR")
                        return False
    
    write_log("JSON数据结构验证通过", "INFO")
    return True

def load_school_data() -> Dict:
    """加载并验证校园数据（原有完整逻辑，适配gate节点）"""
    start_time = time.time()
    data = {}
    
    try:
        # 检查文件是否存在
        if not os.path.exists(DATA_FILE_PATH):
            error_msg = f"数据文件不存在: {DATA_FILE_PATH}"
            write_log(error_msg, "ERROR")
            st.error(f"❌ {error_msg}")
            return {}
        
        # 读取文件
        with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        write_log(f"成功读取数据文件，大小: {os.path.getsize(DATA_FILE_PATH)} 字节", "INFO")
        
        # 验证数据结构
        if not validate_json_structure(data):
            write_log("数据结构验证失败", "ERROR")
            st.error("❌ 数据文件结构不完整，请检查！")
            return {}
        
        # 记录加载耗时
        load_time = round(time.time() - start_time, 3)
        write_log(f"数据加载完成，耗时: {load_time}秒", "INFO")
        
    except json.JSONDecodeError as e:
        error_msg = f"JSON解析错误: {str(e)}"
        write_log(error_msg, "ERROR")
        st.error(f"❌ {error_msg}")
    except PermissionError:
        error_msg = f"无权限读取数据文件: {DATA_FILE_PATH}"
        write_log(error_msg, "ERROR")
        st.error(f"❌ {error_msg}")
    except Exception as e:
        error_msg = f"数据加载异常: {str(e)}"
        write_log(error_msg, "ERROR")
        st.error(f"❌ {error_msg}")
    
    return data

# ===================== 路径计算核心函数（原有100%完整逻辑）=====================
def find_stair_path(
    data: Dict, 
    start_building: str, 
    start_level: str, 
    end_building: str, 
    end_level: str,
    start_point: Optional[str] = None,
    end_point: Optional[str] = None,
    max_depth: int = 50
) -> List[Dict]:
    """
    查找包含楼梯的最优路径（原有完整逻辑，适配gate节点）
    :param data: 校园数据字典
    :param start_building: 起始建筑（支持gate/buildingA/B/C）
    :param start_level: 起始楼层
    :param end_building: 目标建筑（支持gate/buildingA/B/C）
    :param end_level: 目标楼层
    :param start_point: 起始点位（教室/楼梯/走廊）
    :param end_point: 目标点位（教室/楼梯/走廊）
    :param max_depth: 最大递归深度
    :return: 格式化的路径列表
    """
    # 初始化变量
    path = []
    visited = set()
    shortest_path = None
    path_found = False
    recursion_depth = 0
    
    # 构建唯一节点ID
    def get_node_identifier(building: str, level: str, point: str = "") -> str:
        """生成节点唯一标识"""
        point_str = f"_{point}" if point and point.strip() else ""
        return f"{building}_{level}{point_str}"
    
    # 递归查找路径（核心逻辑）
    def dfs(
        current_building: str, 
        current_level: str, 
        current_point: str, 
        current_path: List[Dict],
        depth: int
    ) -> None:
        nonlocal shortest_path, path_found, recursion_depth
        recursion_depth = depth
        
        # 递归深度限制
        if depth >= max_depth:
            write_log(f"递归深度达到上限: {max_depth}", "WARNING")
            return
        
        # 生成当前节点ID
        current_id = get_node_identifier(current_building, current_level, current_point)
        
        # 检查是否已访问
        if current_id in visited:
            return
        
        # 标记已访问
        visited.add(current_id)
        
        # 构建当前节点信息
        current_node = {
            "building": current_building,
            "level": current_level,
            "point": current_point if current_point else "主走廊",
            "depth": depth,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        # 终止条件：到达目标节点
        target_matched = False
        if current_building == end_building and current_level == end_level:
            if end_point:
                if current_point == end_point:
                    target_matched = True
            else:
                target_matched = True
        
        if target_matched:
            # 找到有效路径
            complete_path = current_path + [current_node]
            path_found = True
            
            # 更新最短路径
            if shortest_path is None or len(complete_path) < len(shortest_path):
                shortest_path = complete_path
            write_log(f"找到有效路径，长度: {len(complete_path)}", "INFO")
            visited.remove(current_id)
            return
        
        # 获取当前建筑数据
        if current_building not in data:
            write_log(f"建筑数据不存在: {current_building}", "WARNING")
            visited.remove(current_id)
            return
        
        building_data = data[current_building]
        
        # 获取连接关系
        if "connections" not in building_data:
            write_log(f"{current_building} 无连接关系数据", "WARNING")
            visited.remove(current_id)
            return
        
        # 遍历所有连接
        for connection in building_data["connections"]:
            # 解析连接节点
            from_node = connection.get("from", [])
            to_node = connection.get("to", [])
            
            # 验证连接节点格式
            if len(from_node) < 2 or len(to_node) < 2:
                write_log(f"无效的连接格式: {connection}", "WARNING")
                continue
            
            from_point, from_level = from_node[0], from_node[1]
            to_point, to_level = to_node[0], to_node[1]
            
            # 匹配当前节点
            if (from_point == current_point and from_level == current_level) or \
               (current_point == "" and from_level == current_level and not from_point):
                
                # 确定下一个建筑（适配gate节点）
                next_building = current_building
                
                # 处理跨建筑连接（gate ↔ ABC楼）
                if "gateTo" in from_point or "gateTo" in to_point:
                    if "A" in to_point or "A" in from_point:
                        next_building = "buildingA"
                    elif "B" in to_point or "B" in from_point:
                        next_building = "buildingB"
                    elif "C" in to_point or "C" in from_point:
                        next_building = "buildingC"
                    elif "gate" in to_point or "gate" in from_point:
                        next_building = "gate"
                
                # 递归查找下一个节点
                dfs(
                    current_building=next_building,
                    current_level=to_level,
                    current_point=to_point,
                    current_path=current_path + [current_node],
                    depth=depth + 1
                )
        
        # 回溯：移除访问标记
        visited.remove(current_id)
    
    # 启动DFS搜索
    write_log(f"开始路径搜索: {start_building}({start_level}) → {end_building}({end_level})", "INFO")
    dfs(
        current_building=start_building,
        current_level=start_level,
        current_point=start_point if start_point else "",
        current_path=[],
        depth=0
    )
    
    # 格式化路径结果
    if shortest_path:
        write_log(f"路径搜索完成，最短路径长度: {len(shortest_path)}", "INFO")
        for idx, step in enumerate(shortest_path):
            step_desc = f"从{step['building']} {step['level']} {step['point']}出发" if idx == 0 else \
                        f"到达{step['building']} {step['level']} {step['point']}"
            
            path.append({
                "step": idx + 1,
                "building": step["building"],
                "level": step["level"],
                "location": step["point"],
                "description": step_desc,
                "depth": step["depth"],
                "timestamp": step["timestamp"]
            })
    else:
        # 未找到路径
        write_log("未找到有效路径", "WARNING")
        path = [{
            "step": 1,
            "building": "无",
            "level": "无",
            "location": "无",
            "description": "未找到有效路径，请检查选择的建筑/楼层/点位是否正确",
            "depth": 0,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }]
    
    return path

# ===================== 辅助函数（原有完整集合）=====================
def get_all_buildings(data: Dict) -> List[str]:
    """获取所有建筑名称（适配gate节点）"""
    buildings = list(data.keys())
    write_log(f"获取到建筑列表: {buildings}", "INFO")
    return buildings

def get_building_display_name(building_code: str) -> str:
    """建筑编码转显示名称（新增gate映射）"""
    name_mapping = {
        "buildingA": "A教学楼",
        "buildingB": "B综合楼",
        "buildingC": "C实验楼",
        "gate": "学校大门口"  # 新增大门口映射
    }
    display_name = name_mapping.get(building_code, building_code)
    write_log(f"建筑编码转换: {building_code} → {display_name}", "DEBUG")
    return display_name

def get_building_levels(data: Dict, building: str) -> List[str]:
    """获取指定建筑的所有楼层"""
    levels = []
    if building in data and "levels" in data[building]:
        levels = [level["name"] for level in data[building]["levels"]]
        levels = sorted(list(set(levels)))  # 去重并排序
    write_log(f"{building} 楼层列表: {levels}", "DEBUG")
    return levels

def get_level_all_points(data: Dict, building: str, level: str) -> List[str]:
    """获取指定楼层的所有点位（教室/楼梯/走廊）"""
    points = []
    if building not in data:
        write_log(f"建筑不存在: {building}", "WARNING")
        return points
    
    building_data = data[building]
    for lvl_data in building_data.get("levels", []):
        if lvl_data["name"] == level:
            # 收集教室
            if "classrooms" in lvl_data:
                points.extend([cls["name"] for cls in lvl_data["classrooms"]])
            # 收集楼梯
            if "stairs" in lvl_data:
                points.extend([st["name"] for st in lvl_data["stairs"]])
            # 收集命名走廊
            if "corridors" in lvl_data:
                for corr in lvl_data["corridors"]:
                    if "name" in corr and corr["name"]:
                        points.append(corr["name"])
            break
    
    # 去重并排序
    points = sorted(list(set(points)))
    write_log(f"{building} {level} 点位列表: {points}", "DEBUG")
    return points

def get_point_coordinates(data: Dict, building: str, level: str, point: str) -> Optional[List[float]]:
    """获取指定点位的坐标"""
    if not point or point.strip() == "":
        return None
    
    if building not in data:
        return None
    
    for lvl_data in data[building]["levels"]:
        if lvl_data["name"] == level:
            # 查找教室坐标
            for cls in lvl_data.get("classrooms", []):
                if cls["name"] == point:
                    return cls["coordinates"]
            # 查找楼梯坐标
            for stair in lvl_data.get("stairs", []):
                if stair["name"] == point:
                    return stair["coordinates"]
            # 查找走廊第一个点坐标
            for corr in lvl_data.get("corridors", []):
                if corr.get("name") == point and "points" in corr and len(corr["points"]) > 0:
                    return corr["points"][0]
            break
    
    write_log(f"未找到{building} {level} {point} 的坐标", "WARNING")
    return None

# ===================== Google Sheets 数据同步（原有完整逻辑）=====================
def backup_gsheets_data() -> bool:
    """备份Google Sheets数据"""
    if not gs_client or not gs_main_sheet or not gs_backup_sheet:
        write_log("无法备份：GS客户端未初始化", "ERROR")
        return False
    
    try:
        # 获取主表数据
        main_data = gs_main_sheet.get_all_records()
        if not main_data:
            write_log("主表无数据，跳过备份", "WARNING")
            return True
        
        # 清空备份表
        gs_backup_sheet.clear()
        time.sleep(1)  # 避免API限制
        
        # 写入备份数据
        headers = list(main_data[0].keys()) if main_data else []
        gs_backup_sheet.append_row(headers)
        
        for row in main_data:
            row_values = [row.get(h, "") for h in headers]
            gs_backup_sheet.append_row(row_values)
        
        # 记录备份时间
        backup_note = [f"备份时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "", "", "", "", "", "", "", ""]
        gs_backup_sheet.append_row(backup_note)
        
        write_log("Google Sheets数据备份完成", "INFO")
        return True
    except Exception as e:
        write_log(f"备份失败: {str(e)}", "ERROR")
        return False

def sync_building_data_to_gsheets(data: Dict) -> bool:
    """同步建筑数据到Google Sheets（适配gate节点）"""
    if not gs_client or not gs_main_sheet:
        write_log("GS客户端未初始化，同步失败", "ERROR")
        st.error("❌ Google Sheets客户端未初始化，无法同步数据！")
        return False
    
    # 先备份数据
    if not backup_gsheets_data():
        st.warning("⚠️ 数据备份失败，但仍将尝试同步主表！")
    
    try:
        # 准备表头
        headers = [
            "建筑名称", "建筑编码", "楼层", "点位类型", "点位名称",
            "坐标X", "坐标Y", "坐标Z", "尺寸X", "尺寸Y",
            "创建时间", "最后更新时间"
        ]
        
        # 清空主表
        gs_main_sheet.clear()
        time.sleep(1)  # API调用间隔
        
        # 写入表头
        gs_main_sheet.append_row(headers)
        write_log("写入表头完成", "DEBUG")
        
        # 准备数据行
        data_rows = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        buildings = ["buildingA", "buildingB", "buildingC", "gate"]  # 新增gate
        
        for building_code in buildings:
            if building_code not in data:
                write_log(f"跳过不存在的建筑: {building_code}", "WARNING")
                continue
            
            building_name = get_building_display_name(building_code)
            building_data = data[building_code]
            
            # 遍历楼层
            for level_data in building_data.get("levels", []):
                level_name = level_data["name"]
                z_coord = level_data.get("z", 0)
                
                # 处理教室数据
                for classroom in level_data.get("classrooms", []):
                    cls_coords = classroom.get("coordinates", [0, 0, 0])
                    cls_size = classroom.get("size", [0, 0])
                    row = [
                        building_name, building_code, level_name, "教室", classroom["name"],
                        cls_coords[0], cls_coords[1], cls_coords[2],
                        cls_size[0], cls_size[1],
                        current_time, current_time
                    ]
                    data_rows.append(row)
                
                # 处理楼梯数据
                for stair in level_data.get("stairs", []):
                    stair_coords = stair.get("coordinates", [0, 0, 0])
                    row = [
                        building_name, building_code, level_name, "楼梯", stair["name"],
                        stair_coords[0], stair_coords[1], stair_coords[2],
                        0, 0,
                        current_time, current_time
                    ]
                    data_rows.append(row)
                
                # 处理走廊数据
                for corridor in level_data.get("corridors", []):
                    corr_name = corridor.get("name", "无名走廊")
                    # 取第一个点坐标
                    corr_coords = corridor.get("points", [[0, 0, z_coord]])[0]
                    row = [
                        building_name, building_code, level_name, "走廊", corr_name,
                        corr_coords[0], corr_coords[1], corr_coords[2],
                        0, 0,
                        current_time, current_time
                    ]
                    data_rows.append(row)
        
        # 批量写入数据（分批处理避免API限制）
        batch_size = 50
        for i in range(0, len(data_rows), batch_size):
            batch = data_rows[i:i+batch_size]
            gs_main_sheet.append_rows(batch)
            time.sleep(0.5)  # 避免请求过快
        
        write_log(f"成功同步{len(data_rows)}行数据到Google Sheets", "INFO")
        st.success("✅ 数据已成功同步到Google Sheets！")
        return True
    
    except Exception as e:
        error_msg = f"同步失败: {str(e)}"
        write_log(error_msg, "ERROR")
        st.error(f"❌ {error_msg}")
        return False

# ===================== 3D可视化（原有完整逻辑，适配gate）=====================
def render_3d_navigation_view(data: Dict, path: List[Dict]) -> None:
    """渲染3D导航视图（原有完整逻辑）"""
    if not PLOTLY_AVAILABLE:
        st.info("📌 请安装Plotly库启用3D可视化：pip install plotly")
        return
    
    if not data:
        st.warning("⚠️ 无数据可可视化！")
        return
    
    # 创建3D图形
    fig = make_subplots(
        rows=1, 
        cols=1,
        specs=[[{"type": "scatter3d"}]],
        subplot_titles=["校园建筑3D导航视图"]
    )
    
    # 定义颜色方案（新增gate颜色）
    color_scheme = {
        "buildingA": "#FF6B6B",    # 红色
        "buildingB": "#4ECDC4",    # 青色
        "buildingC": "#45B7D1",    # 蓝色
        "gate": "#8B4513",         # 褐色（大门口）
        "path": "#FF0000",         # 红色（路径）
        "corridor": "#808080",     # 灰色（走廊）
        "classroom": "#90EE90",    # 浅绿色（教室）
        "stair": "#FFD700"         # 金色（楼梯）
    }
    
    # 绘制建筑轮廓
    write_log("开始绘制3D建筑轮廓", "DEBUG")
    for building_code in ["buildingA", "buildingB", "buildingC", "gate"]:
        if building_code not in data:
            continue
        
        building_data = data[building_code]
        building_color = color_scheme[building_code]
        building_name = get_building_display_name(building_code)
        
        for level_data in building_data.get("levels", []):
            level_name = level_data["name"]
            floor_plane = level_data.get("floorPlane", {})
            
            # 获取楼层边界
            min_x = floor_plane.get("minX", 0)
            max_x = floor_plane.get("maxX", 0)
            min_y = floor_plane.get("minY", 0)
            max_y = floor_plane.get("maxY", 0)
            z_coord = level_data.get("z", 0)
            
            # 绘制楼层外框
            if min_x != max_x and min_y != max_y:
                x_coords = [min_x, max_x, max_x, min_x, min_x]
                y_coords = [min_y, min_y, max_y, max_y, min_y]
                z_coords = [z_coord] * 5
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="lines",
                    name=f"{building_name} {level_name}",
                    line=dict(color=building_color, width=3),
                    opacity=0.7,
                    showlegend=True
                ), row=1, col=1)
    
    # 绘制走廊
    write_log("开始绘制3D走廊", "DEBUG")
    for building_code in ["buildingA", "buildingB", "buildingC", "gate"]:
        if building_code not in data:
            continue
        
        building_data = data[building_code]
        
        for level_data in building_data.get("levels", []):
            for corridor in level_data.get("corridors", []):
                if "points" not in corridor or len(corridor["points"]) < 2:
                    continue
                
                # 提取走廊坐标
                points = corridor["points"]
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                z_coords = [p[2] for p in points]
                
                # 获取走廊样式
                corr_style = corridor.get("style", {})
                corr_color = corr_style.get("color", color_scheme["corridor"])
                line_type = corr_style.get("lineType", "solid")
                line_width = corr_style.get("width", 3) or 3
                
                # 绘制走廊
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode="lines",
                    name=corridor.get("name", f"{building_code}走廊"),
                    line=dict(
                        color=corr_color,
                        width=line_width,
                        dash=line_type
                    ),
                    opacity=0.8,
                    showlegend=False
                ), row=1, col=1)
    
    # 绘制教室和楼梯点位
    write_log("开始绘制3D点位", "DEBUG")
    for building_code in ["buildingA", "buildingB", "buildingC", "gate"]:
        if building_code not in data:
            continue
        
        building_data = data[building_code]
        
        for level_data in building_data.get("levels", []):
            z_coord = level_data.get("z", 0)
            
            # 绘制教室
            for classroom in level_data.get("classrooms", []):
                coords = classroom.get("coordinates", [0, 0, z_coord])
                fig.add_trace(go.Scatter3d(
                    x=[coords[0]],
                    y=[coords[1]],
                    z=[coords[2]],
                    mode="markers+text",
                    name=f"{classroom['name']}",
                    text=[classroom["name"]],
                    textposition="top center",
                    marker=dict(
                        color=color_scheme["classroom"],
                        size=6,
                        symbol="square"
                    ),
                    opacity=0.9,
                    showlegend=False
                ), row=1, col=1)
            
            # 绘制楼梯
            for stair in level_data.get("stairs", []):
                coords = stair.get("coordinates", [0, 0, z_coord])
                fig.add_trace(go.Scatter3d(
                    x=[coords[0]],
                    y=[coords[1]],
                    z=[coords[2]],
                    mode="markers+text",
                    name=f"{stair['name']}",
                    text=[stair["name"]],
                    textposition="top center",
                    marker=dict(
                        color=color_scheme["stair"],
                        size=8,
                        symbol="diamond"
                    ),
                    opacity=0.9,
                    showlegend=False
                ), row=1, col=1)
    
    # 绘制导航路径
    if path and path[0]["building"] != "无":
        write_log("开始绘制3D导航路径", "DEBUG")
        path_x = []
        path_y = []
        path_z = []
        path_labels = []
        
        for step in path:
            coords = get_point_coordinates(
                data, 
                step["building"], 
                step["level"], 
                step["location"]
            )
            
            if coords:
                path_x.append(coords[0])
                path_y.append(coords[1])
                path_z.append(coords[2])
                path_labels.append(f"第{step['step']}步")
        
        # 绘制路径线
        if len(path_x) > 1:
            fig.add_trace(go.Scatter3d(
                x=path_x,
                y=path_y,
                z=path_z,
                mode="lines+markers+text",
                name="导航路径",
                text=path_labels,
                textposition="bottom center",
                line=dict(
                    color=color_scheme["path"],
                    width=5,
                    dash="solid"
                ),
                marker=dict(
                    color=color_scheme["path"],
                    size=8,
                    symbol="circle"
                ),
                opacity=1.0,
                showlegend=True
            ), row=1, col=1)
    
    # 配置布局
    fig.update_layout(
        title=dict(
            text=f"{APP_TITLE} - 3D视图",
            font=dict(size=16, weight="bold"),
            x=0.5
        ),
        width=1200,
        height=800,
        scene=dict(
            xaxis=dict(
                title="X坐标 (米)",
                titlefont=dict(size=12),
                range=[-50, 120],  # 适配大门口坐标
                showgrid=True,
                gridcolor="#EEEEEE"
            ),
            yaxis=dict(
                title="Y坐标 (米)",
                titlefont=dict(size=12),
                range=[-90, 80],   # 适配大门口坐标
                showgrid=True,
                gridcolor="#EEEEEE"
            ),
            zaxis=dict(
                title="Z坐标 (米)",
                titlefont=dict(size=12),
                range=[-10, 20],
                showgrid=True,
                gridcolor="#EEEEEE"
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # 渲染图形
    st.plotly_chart(fig, config=PLOTLY_CONFIG, use_container_width=True)
    write_log("3D视图渲染完成", "INFO")

# ===================== Streamlit UI（原有完整逻辑）=====================
def initialize_session_state() -> None:
    """初始化会话状态"""
    default_state = {
        "start_building": "gate",
        "start_level": "level1",
        "start_point": "",
        "end_building": "buildingA",
        "end_level": "level1",
        "end_point": "",
        "path_result": [],
        "last_query_time": "",
        "data_loaded": False
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value
            write_log(f"初始化会话状态: {key} = {value}", "DEBUG")

def render_sidebar(data: Dict) -> None:
    """渲染侧边栏（原有完整逻辑，新增大门口）"""
    with st.sidebar:
        st.title("🔧 导航设置")
        st.divider()
        
        # 建筑选择
        st.subheader("🏢 建筑选择")
        buildings = get_all_buildings(data)
        building_options = {b: get_building_display_name(b) for b in buildings}
        
        st.session_state.start_building = st.selectbox(
            "起始建筑",
            options=buildings,
            format_func=lambda x: building_options[x],
            index=buildings.index(st.session_state.start_building) if st.session_state.start_building in buildings else 0,
            key="sb_start_building"
        )
        
        st.session_state.end_building = st.selectbox(
            "目标建筑",
            options=buildings,
            format_func=lambda x: building_options[x],
            index=buildings.index(st.session_state.end_building) if st.session_state.end_building in buildings else 0,
            key="sb_end_building"
        )
        
        st.divider()
        
        # 楼层选择
        st.subheader("🏬 楼层选择")
        start_levels = get_building_levels(data, st.session_state.start_building)
        if start_levels:
            st.session_state.start_level = st.selectbox(
                "起始楼层",
                options=start_levels,
                index=start_levels.index(st.session_state.start_level) if st.session_state.start_level in start_levels else 0,
                key="sb_start_level"
            )
        else:
            st.session_state.start_level = st.selectbox("起始楼层", ["level1"], key="sb_start_level_empty")
        
        end_levels = get_building_levels(data, st.session_state.end_building)
        if end_levels:
            st.session_state.end_level = st.selectbox(
                "目标楼层",
                options=end_levels,
                index=end_levels.index(st.session_state.end_level) if st.session_state.end_level in end_levels else 0,
                key="sb_end_level"
            )
        else:
            st.session_state.end_level = st.selectbox("目标楼层", ["level1"], key="sb_end_level_empty")
        
        st.divider()
        
        # 点位选择
        st.subheader("📍 点位选择")
        start_points = get_level_all_points(data, st.session_state.start_building, st.session_state.start_level)
        st.session_state.start_point = st.selectbox(
            "起始点位（可选）",
            options=[""] + start_points,
            index=([""] + start_points).index(st.session_state.start_point) if st.session_state.start_point in start_points + [""] else 0,
            key="sb_start_point"
        )
        
        end_points = get_level_all_points(data, st.session_state.end_building, st.session_state.end_level)
        st.session_state.end_point = st.selectbox(
            "目标点位（可选）",
            options=[""] + end_points,
            index=([""] + end_points).index(st.session_state.end_point) if st.session_state.end_point in end_points + [""] else 0,
            key="sb_end_point"
        )
        
        st.divider()
        
        # 功能按钮
        st.subheader("🎯 操作按钮")
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            query_clicked = st.button(
                "🔍 查询路径",
                type="primary",
                use_container_width=True,
                key="btn_query"
            )
        
        with col_btn2:
            sync_clicked = st.button(
                "📤 同步数据",
                use_container_width=True,
                key="btn_sync"
            )
        
        reset_clicked = st.button(
            "🔄 重置设置",
            use_container_width=True,
            key="btn_reset"
        )
        
        # 按钮逻辑
        if query_clicked:
            write_log("用户点击查询路径按钮", "INFO")
            with st.spinner("正在计算最优路径..."):
                st.session_state.path_result = find_stair_path(
                    data=data,
                    start_building=st.session_state.start_building,
                    start_level=st.session_state.start_level,
                    end_building=st.session_state.end_building,
                    end_level=st.session_state.end_level,
                    start_point=st.session_state.start_point if st.session_state.start_point else None,
                    end_point=st.session_state.end_point if st.session_state.end_point else None
                )
                st.session_state.last_query_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()
        
        if sync_clicked:
            write_log("用户点击同步数据按钮", "INFO")
            with st.spinner("正在同步数据到Google Sheets..."):
                sync_building_data_to_gsheets(data)
        
        if reset_clicked:
            write_log("用户点击重置设置按钮", "INFO")
            # 重置会话状态
            st.session_state.start_building = "gate"
            st.session_state.start_level = "level1"
            st.session_state.start_point = ""
            st.session_state.end_building = "buildingA"
            st.session_state.end_level = "level1"
            st.session_state.end_point = ""
            st.session_state.path_result = []
            st.session_state.last_query_time = ""
            st.rerun()
        
        st.divider()
        
        # 系统信息
        with st.expander("ℹ️ 系统信息", expanded=False):
            st.write(f"数据文件: {DATA_FILE_PATH}")
            st.write(f"最后加载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"调试模式: {'开启' if DEBUG_MODE else '关闭'}")
            st.write(f"Plotly可用: {'是' if PLOTLY_AVAILABLE else '否'}")

def render_main_content(data: Dict) -> None:
    """渲染主内容区（原有完整逻辑）"""
    # 页面标题
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.divider()
    
    # 状态提示
    if st.session_state.last_query_time:
        st.info(f"🔍 最后查询时间: {st.session_state.last_query_time}")
    
    # 主内容布局
    col1, col2 = st.columns([2, 3], gap="large")
    
    # 路径结果列
    with col1:
        st.subheader("📝 导航路径结果")
        st.divider()
        
        if st.session_state.path_result:
            # 显示路径
            for step in st.session_state.path_result:
                with st.container(border=True):
                    step_style = "✅ " if step["step"] == len(st.session_state.path_result) else f"🔹 第{step['step']}步:"
                    st.markdown(f"**{step_style} {step['description']}**")
                    if DEBUG_MODE:
                        st.caption(f"深度: {step['depth']} | 时间: {step['timestamp']}")
        else:
            st.info("💡 请在左侧边栏设置导航参数并点击「查询路径」按钮")
        
        st.divider()
        
        # 路径统计
        if st.session_state.path_result and st.session_state.path_result[0]["building"] != "无":
            st.subheader("📊 路径统计")
            st.divider()
            
            total_steps = len(st.session_state.path_result)
            start_building = get_building_display_name(st.session_state.start_building)
            end_building = get_building_display_name(st.session_state.end_building)
            
            stats_data = {
                "起始点": f"{start_building} {st.session_state.start_level} {st.session_state.start_point or '主走廊'}",
                "目标点": f"{end_building} {st.session_state.end_level} {st.session_state.end_point or '主走廊'}",
                "总步数": total_steps,
                "路径类型": "跨建筑路径" if st.session_state.start_building != st.session_state.end_building else "建筑内路径"
            }
            
            for key, value in stats_data.items():
                st.write(f"**{key}**: {value}")
    
    # 3D可视化列
    with col2:
        st.subheader("🗺️ 3D可视化视图")
        st.divider()
        
        # 渲染3D视图
        render_3d_navigation_view(data, st.session_state.path_result)
    
    # 数据详情展开栏
    st.divider()
    with st.expander("📋 建筑数据详情", expanded=False):
        st.subheader("🏫 所有建筑点位数据")
        st.divider()
        
        # 准备数据表格
        table_data = []
        buildings = ["buildingA", "buildingB", "buildingC", "gate"]
        
        for building_code in buildings:
            if building_code not in data:
                continue
            
            building_name = get_building_display_name(building_code)
            building_data = data[building_code]
            
            for level_data in building_data.get("levels", []):
                level_name = level_data["name"]
                
                # 教室数据
                for classroom in level_data.get("classrooms", []):
                    coords = classroom.get("coordinates", [0, 0, 0])
                    table_data.append({
                        "建筑名称": building_name,
                        "建筑编码": building_code,
                        "楼层": level_name,
                        "类型": "教室",
                        "名称": classroom["name"],
                        "坐标(X,Y,Z)": f"{coords[0]}, {coords[1]}, {coords[2]}",
                        "尺寸(X,Y)": f"{classroom.get('size', [0,0])[0]}, {classroom.get('size', [0,0])[1]}"
                    })
                
                # 楼梯数据
                for stair in level_data.get("stairs", []):
                    coords = stair.get("coordinates", [0, 0, 0])
                    table_data.append({
                        "建筑名称": building_name,
                        "建筑编码": building_code,
                        "楼层": level_name,
                        "类型": "楼梯",
                        "名称": stair["name"],
                        "坐标(X,Y,Z)": f"{coords[0]}, {coords[1]}, {coords[2]}",
                        "尺寸(X,Y)": "0, 0"
                    })
                
                # 走廊数据
                for corridor in level_data.get("corridors", []):
                    corr_name = corridor.get("name", "无名走廊")
                    coords = corridor.get("points", [[0, 0, 0]])[0]
                    table_data.append({
                        "建筑名称": building_name,
                        "建筑编码": building_code,
                        "楼层": level_name,
                        "类型": "走廊",
                        "名称": corr_name,
                        "坐标(X,Y,Z)": f"{coords[0]}, {coords[1]}, {coords[2]}",
                        "尺寸(X,Y)": "0, 0"
                    })
        
        # 显示数据表格
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "建筑名称": st.column_config.TextColumn("建筑名称", width="medium"),
                    "建筑编码": st.column_config.TextColumn("建筑编码", width="small"),
                    "楼层": st.column_config.TextColumn("楼层", width="small"),
                    "类型": st.column_config.TextColumn("类型", width="small"),
                    "名称": st.column_config.TextColumn("名称", width="medium"),
                    "坐标(X,Y,Z)": st.column_config.TextColumn("坐标", width="large"),
                    "尺寸(X,Y)": st.column_config.TextColumn("尺寸", width="small")
                }
            )
        else:
            st.warning("⚠️ 暂无数据可显示")

# ===================== 主函数（原有完整逻辑）=====================
def main() -> None:
    """应用主函数"""
    # 初始化Streamlit页面配置
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 初始化会话状态
    initialize_session_state()
    
    # 加载数据
    write_log("========== 应用启动 ==========", "INFO")
    school_data = load_school_data()
    st.session_state.data_loaded = bool(school_data)
    
    # 渲染UI
    render_sidebar(school_data)
    render_main_content(school_data)
    
    # 记录应用退出
    write_log("========== 应用正常运行 ==========", "INFO")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        write_log("应用被用户中断", "INFO")
        st.error("❌ 应用被手动中断！")
    except Exception as e:
        error_msg = f"应用运行异常: {str(e)}"
        write_log(error_msg, "CRITICAL")
        st.error(f"❌ 应用发生错误: {error_msg}")
        if DEBUG_MODE:
            raise
