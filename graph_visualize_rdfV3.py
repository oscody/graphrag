#!/usr/bin/env python
"""
Improved RDF -> NetworkX Visualization Script
---------------------------------------------

Features:
1. Shorten URIs for more readable node/edge labels.
2. Color-code nodes based on RDF types (Document vs. Tag).
3. Fine-tune spring layout parameters (k, iterations).
4. Smaller font size for labels; offset edge labels slightly.
"""

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
# 2b. Helper to shorten long URIs
# ---------------------------------------
def shorten_uri(uri: str) -> str:
    """
    Returns the local part of the URI if possible.
    E.g. "http://example.org/document/YouTube" -> "YouTube"
         "http://schema.org/text" -> "text"
    """
    # If there's a #, split on #; otherwise split on last '/'
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.rsplit("/", 1)[-1]
    return uri

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
    Labels are shortened for readability.
    """
    nx_graph = nx.Graph()

    for s, p, o in rdf_graph:
        # Convert each RDF term to a shortened string
        s_str = shorten_uri(str(s))
        p_str = shorten_uri(str(p))
        o_str = shorten_uri(str(o))

        # Ensure nodes exist in the NX graph
        nx_graph.add_node(s_str)
        nx_graph.add_node(o_str)

        # Add edge with the predicate as an attribute
        nx_graph.add_edge(s_str, o_str, label=p_str)

    return nx_graph

# ---------------------------------------------
# 5. Visualization function: color-coding node
# ---------------------------------------------
def visualize_networkx_graph(rdf_graph: Graph, nx_graph: nx.Graph, title: str = "RDF Graph Visualization"):
    """
    Visualizes a NetworkX graph using Matplotlib. Draws node labels
    and edge labels (predicate names). Color-codes nodes based on RDF type.
    """

    # 1) Identify which nodes are Documents vs. Tags (or unknown).
    document_nodes = set()
    tag_nodes = set()

    for s in rdf_graph.subjects(RDF.type, EX.Document):
        document_nodes.add(shorten_uri(str(s)))
    for s in rdf_graph.subjects(RDF.type, EX.Tag):
        tag_nodes.add(shorten_uri(str(s)))

    # 2) Generate a layout. Adjust k, iterations, and seed if needed
    #    to reduce overlap or to get a consistent layout.
    pos = nx.spring_layout(nx_graph, k=1.0, iterations=100, seed=42)

    plt.figure(figsize=(12, 8))

    # 3) Draw the edges first
    nx.draw_networkx_edges(nx_graph, pos, edge_color='gray')

    # 4) Draw different node sets with different colors
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=document_nodes,
        node_color='lightblue',
        label='Documents'
    )

    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        nodelist=tag_nodes,
        node_color='lightgreen',
        label='Tags'
    )

    # 5) Draw any "other" nodes (those neither Document nor Tag)
    other_nodes = set(nx_graph.nodes()) - document_nodes - tag_nodes
    if other_nodes:
        nx.draw_networkx_nodes(
            nx_graph,
            pos,
            nodelist=other_nodes,
            node_color='lightgray',
            label='Other'
        )

    # 6) Draw labels on the nodes
    nx.draw_networkx_labels(nx_graph, pos, font_size=8)

    # 7) Draw the edge labels, slightly offset from the edges
    edge_labels = nx.get_edge_attributes(nx_graph, 'label')
    nx.draw_networkx_edge_labels(
        nx_graph,
        pos,
        edge_labels=edge_labels,
        font_color='red',
        font_size=7,
        label_pos=0.5  # 0=near source, 1=near target
    )

    plt.title(title)
    plt.legend(scatterpoints=1)
    plt.axis("off")
    plt.show()

# ----------------
# 6. Main Function
# ----------------
def main():
    # 1) Build and serialize the graph
    directory = "/Users/bogle/Dev/gitcode/chatgpt-markdown/test/"
    output_ttl = "markdown.ttl"

    print("Building RDF graph from Markdown files...")
    g = build_rdf_graph_from_markdown(directory)

    print(f"Serializing graph to {output_ttl}...")
    g.serialize(destination=output_ttl, format="turtle")

    # 2) Reload the RDF graph
    print("Reloading RDF graph from Turtle file...")
    loaded_graph = Graph()
    loaded_graph.parse(output_ttl, format="turtle")

    # 3) Convert to NetworkX for visualization
    print("Converting RDF graph to NetworkX...")
    nx_graph = rdflib_to_networkx(loaded_graph)

    # 4) Visualize with color-coding and improved layout
    print("Visualizing the graph...")
    visualize_networkx_graph(loaded_graph, nx_graph, title="Markdown Ontology Graph")

if __name__ == "__main__":
    main()