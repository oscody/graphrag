from rdflib import Graph
import networkx as nx
import matplotlib.pyplot as plt

# 1. Load the RDF graph using rdflib
g = Graph()
g.parse("markdown_ontologyV2.ttl", format="turtle")

# 2. Convert RDF graph to a NetworkX graph
nx_graph = nx.Graph()

for s, p, o in g:
    # Add nodes for the subject and object
    nx_graph.add_node(s)
    nx_graph.add_node(o)
    # Add an edge labeled by the predicate
    nx_graph.add_edge(s, o, label=str(p))

# 3. Visualize with Matplotlib
pos = nx.spring_layout(nx_graph)  # position the nodes using a force-directed layout

# Draw the network nodes and edges
nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', edge_color='gray')

# Draw edge labels for easier interpretation
edge_labels = nx.get_edge_attributes(nx_graph, 'label')
nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red')

# Show the plot
plt.show()