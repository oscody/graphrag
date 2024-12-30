#!/usr/bin/env python

import os
from datetime import datetime
import urllib.parse
import frontmatter

# RDFlib imports
from rdflib import Graph, RDF, RDFS, Namespace, URIRef, Literal
from rdflib.namespace import XSD

# NetworkX / Matplotlib imports (for visualization)
import networkx as nx
import matplotlib.pyplot as plt

# --------------------
# 1. Define Namespaces
# --------------------
EX = Namespace('http://example.org/')
SCHEMA = Namespace('http://schema.org/')

# -------------------------------
# 2. Function to create valid URIs
# -------------------------------
def create_valid_uri(base_uri: str, text: str) -> URIRef:
    """
    Sanitizes the given text and appends it to the base URI.
    """
    sanitized_text = urllib.parse.quote(text.strip().replace(' ', '_'))
    return URIRef(f"{base_uri}/{sanitized_text}")

# ---------------------------------------
# 3. Function to build RDF graph from MD
# ---------------------------------------
def build_rdf_graph_from_markdown(directory: str) -> Graph:
    """
    Walks through a directory of markdown files, extracts frontmatter metadata,
    and builds an RDF graph (rdflib.Graph).
    """
    g = Graph()
    # Bind common prefixes
    g.bind('ex', EX)
    g.bind('schema', SCHEMA)

    # Process all markdown files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = frontmatter.load(file)
                
                # Extract metadata with fallbacks
                title = content.get('title', 'Untitled')
                date_val = content.get('date', datetime.now().date().isoformat())
                tags = content.get('tags', [])
                body = content.content

                # Create URI for this document
                doc_uri = create_valid_uri("http://example.org/document", title)

                # Add metadata to the graph
                g.add((doc_uri, RDF.type, EX.Document))
                g.add((doc_uri, SCHEMA.name, Literal(title, datatype=XSD.string)))
                g.add((doc_uri, SCHEMA.datePublished, Literal(str(date_val), datatype=XSD.date)))
                g.add((doc_uri, SCHEMA.text, Literal(body, datatype=XSD.string)))

                # Add Tags
                for tag in tags:
                    tag_uri = create_valid_uri("http://example.org/tag", tag)
                    g.add((tag_uri, RDF.type, EX.Tag))
                    g.add((tag_uri, RDFS.label, Literal(tag, datatype=XSD.string)))
                    g.add((doc_uri, SCHEMA.about, tag_uri))

    return g

# --------------------------------------------
# 4. Function to convert RDF graph -> NetworkX
# --------------------------------------------
def rdflib_to_networkx(rdf_graph: Graph) -> nx.Graph:
    """
    Converts an rdflib.Graph into a NetworkX Graph object for visualization.
    """
    nx_graph = nx.Graph()

    for s, p, o in rdf_graph:
        # Convert to strings for consistency (otherwise might be URIRefs, BNodes, Literals)
        s_str = str(s)
        p_str = str(p)
        o_str = str(o)

        # Ensure nodes exist in the NX graph
        nx_graph.add_node(s_str)
        nx_graph.add_node(o_str)

        # Add edge with the predicate as an attribute
        nx_graph.add_edge(s_str, o_str, label=p_str)

    return nx_graph

# --------------------------------
# 5. Visualization helper function
# --------------------------------
def visualize_networkx_graph(nx_graph: nx.Graph, title: str = "RDF Graph Visualization"):
    """
    Visualizes a NetworkX graph using Matplotlib. Draws node labels and
    edge labels (predicate names).
    """
    plt.figure(figsize=(12, 8))
    # Spring layout tends to be a decent "force-directed" layout
    pos = nx.spring_layout(nx_graph, k=0.5, iterations=50)

    # Draw nodes and edges
    nx.draw(
        nx_graph,
        pos,
        with_labels=True,
        node_color='lightblue',
        edge_color='gray',
        font_size=8
    )

    # Draw predicate labels on edges
    edge_labels = nx.get_edge_attributes(nx_graph, 'label')
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=edge_labels, font_color='red', font_size=7)

    plt.title(title)
    plt.axis("off")
    plt.show()

# ----------------
# 7. Main Function
# ----------------
def main():
    # 7.1 Build and serialize the graph
    directory = "/Users/bogle/Dev/gitcode/chatgpt-markdown/test/"
    output_ttl = "markdown.ttl"

    print("Building RDF graph from Markdown files...")
    g = build_rdf_graph_from_markdown(directory)

    print(f"Serializing graph to {output_ttl}...")
    g.serialize(destination=output_ttl, format="turtle")

    # 7.2 Reload the RDF graph
    print("Reloading RDF graph from Turtle file...")
    loaded_graph = Graph()
    loaded_graph.parse(output_ttl, format="turtle")

    # 7.3 Convert to NetworkX for visualization
    print("Converting RDF graph to NetworkX...")
    nx_graph = rdflib_to_networkx(loaded_graph)

    print("Visualizing the graph...")
    visualize_networkx_graph(nx_graph, title="Markdown Ontology Graph")

if __name__ == "__main__":
    main()