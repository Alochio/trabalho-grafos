import sys
import time
import networkx as nx
import heapq
import numpy as np

# Função para ler o arquivo DIMACS
def read_dimacs_file(file_path):
    G = nx.DiGraph()
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            num_nodes, num_edges = map(int, lines[0].strip().split())
            
            for line in lines[1:]:
                u, v, w = map(float, line.strip().split())
                G.add_edge(int(u), int(v), weight=w)
    except MemoryError:
        raise MemoryError("MEMORIA EXCEDIDA")
    return G

# Função para executar com timeout
def run_with_timeout(func, *args, timeout=600):
    start_time = time.time()
    
    try:
        result = func(*args)
    except MemoryError as e:
        return None, str(e), None
    except Exception as e:
        return None, str(e), None
    
    elapsed_time = time.time() - start_time
    
    if elapsed_time > timeout:
        return None, "TEMPO LIMITE", elapsed_time
    
    return result, None, elapsed_time

def dijkstra(graph, start, end):
    # Inicializa distâncias e predecessores
    dist = {node: float('inf') for node in graph.nodes}
    prev = {node: None for node in graph.nodes}
    dist[start] = 0
    
    # Fila de prioridade
    priority_queue = [(0, start)]  # (distância, vértice)
    
    while priority_queue:
        current_dist, u = heapq.heappop(priority_queue)
        
        # Se o vértice já foi processado com uma distância menor, ignore
        if current_dist > dist[u]:
            continue
        
        # Atualiza distâncias dos vizinhos
        for v in graph.neighbors(u):
            weight = graph[u][v].get('weight', 1)
            if dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                prev[v] = u
                heapq.heappush(priority_queue, (dist[v], v))
    
    # Reconstrução do caminho
    path = []
    current = end
    while prev[current] is not None:
        path.insert(0, current)
        current = prev[current]
    if path or start == end:
        path.insert(0, start)
    
    return path, dist[end]

# Algoritmo de Bellman-Ford
def bellman_ford(graph, start, end):
    # Inicialização
    dist = {node: float('inf') for node in graph.nodes}
    prev = {node: None for node in graph.nodes}
    dist[start] = 0
    
    # Relaxamento das arestas
    for _ in range(len(graph.nodes) - 1):
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1)
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                prev[v] = u
    
    # Verificação de ciclos negativos
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        if dist[u] + weight < dist[v]:
            return None, "Erro: Ciclo negativo encontrado"
    
    # Reconstrução do caminho
    path = []
    current = end
    while current is not None:
        path.insert(0, current)
        current = prev[current]
    
    if path[0] != start:
        return None, "Caminho não encontrado"
    
    return path, dist[end]

def floyd_warshall(graph):
    # Cria um mapeamento dos nós para índices
    nodes = list(graph.nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    
    # Inicializa as matrizes de distâncias e predecessores
    V = len(nodes)
    dist = np.full((V, V), np.inf)
    next_node = np.full((V, V), None)
    
    # Preenche as matrizes com os pesos das arestas
    for u in graph.nodes:
        u_idx = node_index[u]
        for v in graph.neighbors(u):
            v_idx = node_index[v]
            dist[u_idx][v_idx] = graph[u][v].get('weight', 1)
            next_node[u_idx][v_idx] = v
    
    # Inicializa a diagonal da matriz de distâncias com 0
    for i in range(V):
        dist[i][i] = 0
    
    # Aplica o algoritmo de Floyd-Warshall
    for k in range(V):
        for i in range(V):
            for j in range(V):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]
    
    # Reconstrói o caminho mais curto
    def reconstruct_path(start, end):
        start_idx = node_index[start]
        end_idx = node_index[end]
        
        if dist[start_idx][end_idx] == np.inf:
            return None, "Caminho não encontrado"
        
        path = []
        while start_idx is not None:
            path.append(nodes[start_idx])
            if start_idx == end_idx:
                break
            start_idx = node_index.get(next_node[start_idx][end_idx], None)
        
        if len(path) == 0 or path[0] != start:
            return None, "Caminho não encontrado"
        
        return path, dist[node_index[start]][node_index[end]]
    
    return reconstruct_path

def main():
    if len(sys.argv) != 4:
        print("Uso: python main.py <arquivo_dimacs> <origem> <destino>")
        return
    
    file_path = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    
    try:
        graph = read_dimacs_file(file_path)
    except MemoryError:
        print("MEMORIA EXCEDIDA")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo: {e}")
        return
    
    # Verificar se os nós existem no grafo
    if start not in graph.nodes or end not in graph.nodes:
        print(f"Erro: Nó {start} ou {end} não encontrado no grafo.")
        return
    
    # Algoritmo de Dijkstra
    print("Processando ...")
    print("-" * 100)
    
    result, error, time_taken = run_with_timeout(dijkstra, graph, start, end, timeout=600)
    if error:
        print("Algoritmo de Dijkstra:")
        print(error)
    else:
        path, length = result
        print("Algoritmo de Dijkstra:")
        print(f"Caminho mínimo: {path}")
        print(f"Custo: {length}")
        print(f"Tempo: {time_taken:.3f}s / {time_taken * 1000:.3f}ms")
    print("-" * 100)
    
    # Algoritmo de Bellman-Ford
    result, error, time_taken = run_with_timeout(bellman_ford, graph, start, end, timeout=600)
    if error:
        print("Algoritmo de Bellman-Ford:")
        print(error)
    else:
        path, length = result
        print("Algoritmo de Bellman-Ford:")
        print(f"Caminho mínimo: {path}")
        print(f"Custo: {length}")
        print(f"Tempo: {time_taken:.3f}s / {time_taken * 1000:.3f}ms")
    print("-" * 100)
    
    # Algoritmo de Floyd-Warshall
    fw_reconstruct_path = floyd_warshall(graph)
    result, error, time_taken = run_with_timeout(lambda: fw_reconstruct_path(start, end), timeout=600)
    if error:
        print("Algoritmo de Floyd-Warshall:")
        print(error)
    else:
        path, length = result
        print("Algoritmo de Floyd-Warshall:")
        print(f"Caminho mínimo: {path}")
        print(f"Custo: {length}")
        print(f"Tempo: {time_taken:.3f}s / {time_taken * 1000:.3f}ms")
    print("-" * 100)

if __name__ == "__main__":
    main()
