import matplotlib.pyplot as plt
import networkx as nx

def read_generic_file(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) == 3:
                u, v, w = parts
                u, v, w = int(u), int(v), float(w)
                G.add_edge(u, v, weight=w)
    return G

def draw_graph(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title('Visualização do Grafo')
    plt.show()

def main():
    file_path = 'data/toy.txt'  # Caminho para o arquivo de entrada
    graph = read_generic_file(file_path)
    draw_graph(graph)

if __name__ == "__main__":
    main()
