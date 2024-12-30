# graph_kg_rag.py

import os
import frontmatter
from rdflib import Graph, RDF, RDFS, Namespace, URIRef, Literal
from rdflib.namespace import XSD
from datetime import datetime
import urllib.parse


# Define Namespaces
ex = Namespace('http://example.org/')
schema = Namespace('http://schema.org/')

# Initialize RDF Graph
g = Graph()
g.bind('ex', ex)
g.bind('schema', schema)

def create_valid_uri(base_uri, text):
    sanitized_text = urllib.parse.quote(text.strip().replace(' ', '_'))
    return URIRef(f"{base_uri}/{sanitized_text}")

directory = "/Users/bogle/Dev/gitcode/chatgpt-markdown/test/"

# Parse Markdown Files -> Build RDF Graph
for filename in os.listdir(directory):
    if filename.endswith(".md"):
        with open(os.path.join(directory, filename), 'r') as file:
            content = frontmatter.load(file)
            
            # Extract Metadata
            title = content.get('title', 'Untitled')
            date_val = content.get('date', datetime.now().date().isoformat())
            tags = content.get('tags', [])
            body = content.content
            
            # Create URI for the document
            doc_uri = create_valid_uri("http://example.org/document", title)
            
            # Add Metadata to the Graph
            g.add((doc_uri, RDF.type, URIRef(ex.Document)))
            g.add((doc_uri, schema.name, Literal(title, datatype=XSD.string)))
            g.add((doc_uri, schema.datePublished, Literal(str(date_val), datatype=XSD.date)))
            g.add((doc_uri, schema.text, Literal(body, datatype=XSD.string)))
            
            # Add Tags
            for tag in tags:
                tag_uri = create_valid_uri("http://example.org/tag", tag)
                g.add((tag_uri, RDF.type, URIRef(ex.Tag)))
                g.add((tag_uri, RDFS.label, Literal(tag, datatype=XSD.string)))
                g.add((doc_uri, schema.about, tag_uri))

# Serialize the Graph
g.serialize(destination="markdown_ontologyV2.ttl", format="turtle")


# Re-load our saved graph
g = Graph()
g.parse("markdown_ontologyV2.ttl", format="turtle")

# Namespaces
ex = Namespace('http://example.org/')
schema = Namespace('http://schema.org/')

def local_semantic_search(search_term):
    """
    Perform a SPARQL query on the local RDF graph to find 
    Documents whose title, body text, or tags match 'search_term'.
    """
    # We lowercase the search term in the FILTER to do case-insensitive matching
    # Using CONTAINS to do substring matching in SPARQL.
    query = f"""
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?doc ?title ?text ?tagLabel
    WHERE {{
      ?doc a ex:Document ;
           schema:name ?title ;
           schema:text ?text ;
           schema:about ?tag .
      ?tag rdfs:label ?tagLabel .
      
      FILTER (
        CONTAINS(LCASE(?title), LCASE("{search_term}")) ||
        CONTAINS(LCASE(?text), LCASE("{search_term}")) ||
        CONTAINS(LCASE(?tagLabel), LCASE("{search_term}"))
      )
    }}
    """
    results = g.query(query)

    # Group results by document
    doc_map = {}
    for row in results:
        doc_uri = row['doc']
        title_str = str(row['title'])
        text_str = str(row['text'])
        tag_str = str(row['tagLabel'])

        if doc_uri not in doc_map:
            doc_map[doc_uri] = {
                'title': title_str,
                'body': text_str,
                'tags': set()
            }
        doc_map[doc_uri]['tags'].add(tag_str)

    # Return the aggregated results
    return doc_map

# Example usage
search_query = "sixth"

matched_docs = local_semantic_search(search_query)

if not matched_docs:
    print(f"No documents found matching '{search_query}'")
else:
    print(f"Documents matching '{search_query}':\n")
    for doc_uri, info in matched_docs.items():
        print(f"Document URI: {doc_uri}")
        print(f"Title: {info['title']}")
        print(f"Tags: {list(info['tags'])}")
        # Print a snippet of the body
        snippet = (info['body'][:150] + "...") if len(info['body']) > 150 else info['body']
        print(f"Snippet of text: {snippet}")
        print("------")
