import json
import heapq

class SchoolPathFinder:
    def __init__(self, map_file):
        """初始化路径查找器，加载学校地图数据"""
        self.map_data = self.load_map(map_file)
        self.graph = self.build_graph()
        
    def load_map(self, file_path):
        """从JSON文件加载地图数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误：找不到地图文件 {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"错误：解析地图文件 {file_path} 失败")
            return None
    
    def build_graph(self):
        """构建图结构用于路径查找"""
        if not self.map_data:
            return {}
            
        graph = {}
        
        # 添加各楼层内部的连接
        for floor in self.map_data['floors']:
            floor_num = floor['floor']
            for connection in floor['connections']:
                from_node = f"{floor_num}:{connection['from']}"
                to_node = f"{floor_num}:{connection['to']}"
                distance = connection['distance']
                
                # 添加双向连接
                if from_node not in graph:
                    graph[from_node] = []
                graph[from_node].append((to_node, distance))
                
                if to_node not in graph:
                    graph[to_node] = []
                graph[to_node].append((from_node, distance))
        
        # 添加楼层间的连接（如楼梯）
        for connection in self.map_data['inter_floor_connections']:
            from_node = connection['from']
            to_node = connection['to']
            distance = connection['distance']
            
            if from_node not in graph:
                graph[from_node] = []
            graph[from_node].append((to_node, distance))
            
            if to_node not in graph:
                graph[to_node] = []
            graph[to_node].append((from_node, distance))
            
        return graph
    
 def get_room_id(self, floor, room_name):
        """根据楼层和教室名称获取房间ID"""
        for f in self.map_data['floors']:
            if f['floor'] == floor:
                for room in f['rooms']:
                    if room['name'] == room_name:
                        return f"{floor}:{room['id']}"
        return None
    
    def get_room_name(self, node_id):
        """根据节点ID获取房间名称"""
        floor_num, room_id = node_id.split(':')
        floor_num = int(floor_num)
        
        for f in self.map_data['floors']:
            if f['floor'] == floor_num:
                for room in f['rooms']:
                    if room['id'] == room_id:
                        return room['name']
        return room_id
    
    def dijkstra(self, start, end):
        """使用Dijkstra算法查找最短路径"""
        if start not in self.graph or end not in self.graph:
            return None, 0
            
        # 初始化距离字典，存储从起点到每个节点的最短距离
        distances = {node: float('infinity') for node in self.graph}
        distances[start] = 0
        
        # 初始化优先队列，存储(距离, 节点)
        priority_queue = [(0, start)]
        
        # 存储路径
        previous_nodes = {node: None for node in self.graph}
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            # 如果到达终点，提前退出
            if current_node == end:
                break
                
            # 如果当前距离大于已知最短距离，跳过
            if current_distance > distances[current_node]:
                continue
                
            # 探索邻居节点
            for neighbor, weight in self.graph[current_node]:
                distance = current_distance + weight
                
                # 如果找到更短的路径
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # 重建路径
        path = []
        current = end
        while current is not None:
            path.append(current)
            current = previous_nodes[current]
        
        # 反转路径，从起点到终点
        path.reverse()
        
        # 如果路径只有一个节点且不是起点和终点相同的情况，则表示没有找到路径
        if len(path) == 1 and path[0] != start:
            return None, 0
            
        return path, distances[end]
    
    def find_path(self, start_floor, start_room, end_floor, end_room):
        """查找从起始教室到目标教室的路径"""
        start_node = self.get_room_id(start_floor, start_room)
        end_node = self.get_room_id(end_floor, end_room)
        
        if not start_node or not end_node:
            print("错误：起始教室或目标教室不存在")
            return None, 0
            
        if start_node == end_node:
            print("起始教室和目标教室相同")
            return [start_node], 0
            
        path, distance = self.dijkstra(start_node, end_node)
        
        if not path:
            print("错误：找不到路径")
            return None, 0
            
        return path, distance
    
    def print_path(self, path):
        """打印路径"""
        if not path:
            return
            
        print("路径规划：")
        for i, node in enumerate(path):
            floor, room_id = node.split(':')
            room_name = self.get_room_name(node)
            
            # 显示当前位置
            print(f"{i+1}. 楼层 {floor}，{room_name}")
            
            # 如果不是最后一个节点，显示下一步
            if i < len(path) - 1:
                next_floor, next_room = path[i+1].split(':')
                if floor != next_floor:
                    print("   → 乘坐楼梯前往{}楼".format(next_floor))
                else:
                    print("   → 前往{}".format(self.get_room_name(path[i+1])))


def main():
    # 创建路径查找器实例
    path_finder = SchoolPathFinder('school_map.json')
    
    if not path_finder.map_data:
        return
    
    print("欢迎使用学校教室路径规划系统")
    print("--------------------------")
    
    # 显示所有可用教室
    print("可用教室：")
    for floor in path_finder.map_data['floors']:
        print(f"{floor['floor']}楼：")
        for room in floor['rooms']:
            print(f"  - {room['name']}")
    print("--------------------------")
    
    # 获取用户输入
    try:
        start_floor = int(input("请输入起始楼层："))
        start_room = input("请输入起始教室名称：")
        end_floor = int(input("请输入目标楼层："))
        end_room = input("请输入目标教室名称：")
        
        # 查找路径
        path, distance = path_finder.find_path(start_floor, start_room, end_floor, end_room)
        
        if path:
            print(f"\n找到最短路径，总距离：{distance}单位")
            path_finder.print_path(path)
    except ValueError:
        print("输入错误，请确保楼层为数字")


if __name__ == "__main__":
    main()
