import json
import networkx as nx
import matplotlib.pyplot as plt

class ClassroomNavigator:
    def __init__(self, json_path):
        # 加载JSON数据
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # 构建导航图
        self.graph = nx.Graph()
        self._build_graph()
        
        # 存储节点坐标信息
        self.node_positions = self._get_node_positions()

    def _build_graph(self):
        """从JSON数据构建图网络"""
        # 添加所有楼层的节点和边
        for floor in self.data['floors']:
            # 添加节点
            for node in floor['nodes']:
                self.graph.add_node(
                    node['id'],
                    name=node['name'],
                    floor=floor['floor_id'],
                    type=node['type'],
                    x=node['x'],
                    y=node['y'],
                    z=node['z']
                )
            
            # 添加同层边
            for edge in floor['edges']:
                self.graph.add_edge(
                    edge['from'],
                    edge['to'],
                    weight=edge['distance']
                )
        
        # 添加楼层间连接（楼梯/电梯）
        for edge in self.data['inter_floor_edges']:
            self.graph.add_edge(
                edge['from'],
                edge['to'],
                weight=edge['distance']
            )

    def _get_node_positions(self):
        """提取所有节点的坐标信息"""
        positions = {}
        for floor in self.data['floors']:
            for node in floor['nodes']:
                positions[node['id']] = (node['x'], node['y'], node['z'])
        return positions

    def find_shortest_path(self, start_id, end_id):
        """查找两个节点之间的最短路径"""
        if start_id not in self.graph.nodes or end_id not in self.graph.nodes:
            return None, 0
        
        try:
            path = nx.shortest_path(
                self.graph, 
                source=start_id, 
                target=end_id, 
                weight='weight'
            )
            distance = nx.shortest_path_length(
                self.graph, 
                source=start_id, 
                target=end_id, 
                weight='weight'
            )
            return path, distance
        except nx.NetworkXNoPath:
            return None, 0

    def get_path_details(self, path):
        """获取路径的详细信息（包含名称、楼层等）"""
        if not path:
            return []
        
        details = []
        for node_id in path:
            node_data = self.graph.nodes[node_id]
            details.append({
                'id': node_id,
                'name': node_data['name'],
                'floor': node_data['floor'],
                'type': node_data['type'],
                'coordinates': (node_data['x'], node_data['y'], node_data['z'])
            })
        return details

    def visualize_2d(self, path=None):
        """2D可视化楼层平面图和路径"""
        # 按楼层分别可视化
        for floor in self.data['floors']:
            floor_id = floor['floor_id']
            floor_nodes = [n['id'] for n in floor['nodes']]
            
            # 筛选当前楼层的节点和边
            floor_subgraph = self.graph.subgraph(floor_nodes)
            pos = {n: (self.graph.nodes[n]['x'], self.graph.nodes[n]['y']) 
                   for n in floor_subgraph.nodes}
            
            plt.figure(figsize=(10, 8))
            plt.title(f"Floor {floor_id}: {floor['name']}")
            
            # 绘制所有节点和边
            nx.draw_networkx_edges(floor_subgraph, pos, edge_color='gray', width=1)
            nx.draw_networkx_nodes(floor_subgraph, pos, node_size=300, node_color='lightblue')
            nx.draw_networkx_labels(floor_subgraph, pos, 
                                   labels={n: self.graph.nodes[n]['name'] for n in floor_subgraph.nodes},
                                   font_size=8)
            
            # 如果有路径，高亮显示当前楼层的路径部分
            if path:
                floor_path = [n for n in path if n in floor_nodes]
                if len(floor_path) >= 2:
                    path_edges = list(zip(floor_path[:-1], floor_path[1:]))
                    nx.draw_networkx_edges(floor_subgraph, pos, edgelist=path_edges,
                                          edge_color='red', width=2)
            
            plt.grid(True)
            plt.axis('equal')
            plt.show()

if __name__ == "__main__":
    # 初始化导航系统
    navigator = ClassroomNavigator('floor_plan_data.json')
    
    # 查找从A205到Cafeteria的路径
    start = 'a205'
    end = 'cafe'
    path, distance = navigator.find_shortest_path(start, end)
    
    if path:
        print(f"最短路径从 {navigator.graph.nodes[start]['name']} 到 {navigator.graph.nodes[end]['name']}:")
        print(f"总距离: {distance} 单位")
        print("路径详情:")
        for i, step in enumerate(navigator.get_path_details(path)):
            print(f"  {i+1}. {step['name']} (楼层 {step['floor']})")
        
        # 可视化路径
        navigator.visualize_2d(path)
    else:
        print(f"无法找到从 {start} 到 {end} 的路径")
