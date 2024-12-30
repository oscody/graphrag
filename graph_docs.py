# https://github.com/SteveHedden/kg_llm/blob/main/VectorVsKG.ipynb


### Turn metadata into a KG


from rdflib import Graph, RDF, RDFS, Namespace, URIRef, Literal
from rdflib.namespace import SKOS, XSD
import pandas as pd
import urllib.parse
import random
from datetime import datetime, timedelta

# Create a new RDF graph
g = Graph()

# Define namespaces
schema = Namespace('http://schema.org/')
ex = Namespace('http://example.org/')
prefixes = {
    'schema': schema,
    'ex': ex,
    'skos': SKOS,
    'xsd': XSD
}
for p, ns in prefixes.items():
    g.bind(p, ns)

# Define classes and properties
Article = URIRef(ex.Article)
MeSHTerm = URIRef(ex.MeSHTerm)
g.add((Article, RDF.type, RDFS.Class))
g.add((MeSHTerm, RDF.type, RDFS.Class))

title = URIRef(schema.name)
abstract = URIRef(schema.description)
date_published = URIRef(schema.datePublished)
access = URIRef(ex.access)

g.add((title, RDF.type, RDF.Property))
g.add((abstract, RDF.type, RDF.Property))
g.add((date_published, RDF.type, RDF.Property))
g.add((access, RDF.type, RDF.Property))

# Function to clean and parse MeSH terms
def parse_mesh_terms(mesh_list):
    if pd.isna(mesh_list):
        return []
    return [term.strip().replace(' ', '_') for term in mesh_list.strip("[]'").split(',')]

# Function to create a valid URI
def create_valid_uri(base_uri, text):
    if pd.isna(text):
        return None
    sanitized_text = urllib.parse.quote(text.strip().replace(' ', '_').replace('"', '').replace('<', '').replace('>', '').replace("'", "_"))
    return URIRef(f"{base_uri}/{sanitized_text}")

# Function to generate a random date within the last 5 years
def generate_random_date():
    start_date = datetime.now() - timedelta(days=5*365)
    random_days = random.randint(0, 5*365)
    return start_date + timedelta(days=random_days)

# Function to generate a random access value between 1 and 10
def generate_random_access():
    return random.randint(1, 10)

# Load your DataFrame here
# df = pd.read_csv('your_data.csv')

# Loop through each row in the DataFrame and create RDF triples
for index, row in df.iterrows():
    article_uri = create_valid_uri("http://example.org/article", row['Title'])
    if article_uri is None:
        continue
    
    # Add Article instance
    g.add((article_uri, RDF.type, Article))
    g.add((article_uri, title, Literal(row['Title'], datatype=XSD.string)))
    g.add((article_uri, abstract, Literal(row['abstractText'], datatype=XSD.string)))
    
    # Add random datePublished and access
    random_date = generate_random_date()
    random_access = generate_random_access()
    g.add((article_uri, date_published, Literal(random_date.date(), datatype=XSD.date)))
    g.add((article_uri, access, Literal(random_access, datatype=XSD.integer)))
    
    # Add MeSH Terms
    mesh_terms = parse_mesh_terms(row['meshMajor'])
    for term in mesh_terms:
        term_uri = create_valid_uri("http://example.org/mesh", term)
        if term_uri is None:
            continue
        
        # Add MeSH Term instance
        g.add((term_uri, RDF.type, MeSHTerm))
        g.add((term_uri, RDFS.label, Literal(term.replace('_', ' '), datatype=XSD.string)))
        
        # Link Article to MeSH Term
        g.add((article_uri, schema.about, term_uri))

# Serialize the graph to a file (optional)
g.serialize(destination='ontology.ttl', format='turtle')

### Semantic search using KG
from SPARQLWrapper import SPARQLWrapper, JSON

def get_concept_triples_for_term(term):
    sparql = SPARQLWrapper("https://id.nlm.nih.gov/mesh/sparql")
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
    PREFIX mesh: <http://id.nlm.nih.gov/mesh/>

    SELECT ?subject ?p ?pLabel ?o ?oLabel
    FROM <http://id.nlm.nih.gov/mesh>
    WHERE {{
        ?subject rdfs:label "{term}"@en .
        ?subject ?p ?o .
        FILTER(CONTAINS(STR(?p), "concept"))
        OPTIONAL {{ ?p rdfs:label ?pLabel . }}
        OPTIONAL {{ ?o rdfs:label ?oLabel . }}
    }}
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    triples = set()  # Using a set to avoid duplicate entries
    for result in results["results"]["bindings"]:
        obj_label = result.get("oLabel", {}).get("value", "No label")
        triples.add(obj_label)
    
    # Add the term itself to the list
    triples.add(term)
    
    return list(triples)  # Convert back to a list for easier handling

def get_narrower_concepts_for_term(term):
    sparql = SPARQLWrapper("https://id.nlm.nih.gov/mesh/sparql")
    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
    PREFIX mesh: <http://id.nlm.nih.gov/mesh/>

    SELECT ?narrowerConcept ?narrowerConceptLabel
    WHERE {{
        ?broaderConcept rdfs:label "{term}"@en .
        ?narrowerConcept meshv:broaderDescriptor ?broaderConcept .
        ?narrowerConcept rdfs:label ?narrowerConceptLabel .
    }}
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    concepts = set()  # Using a set to avoid duplicate entries
    for result in results["results"]["bindings"]:
        subject_label = result.get("narrowerConceptLabel", {}).get("value", "No label")
        concepts.add(subject_label)
    
    return list(concepts)  # Convert back to a list for easier handling

def get_all_narrower_concepts(term, depth=2, current_depth=1):
    # Create a dictionary to store the terms and their narrower concepts
    all_concepts = {}

    # Initial fetch for the primary term
    narrower_concepts = get_narrower_concepts_for_term(term)
    all_concepts[term] = narrower_concepts
    
    # If the current depth is less than the desired depth, fetch narrower concepts recursively
    if current_depth < depth:
        for concept in narrower_concepts:
            # Recursive call to fetch narrower concepts for the current concept
            child_concepts = get_all_narrower_concepts(concept, depth, current_depth + 1)
            all_concepts.update(child_concepts)
    
    return all_concepts

# Fetch alternative names and narrower concepts
term = "Mouth Neoplasms"
alternative_names = get_concept_triples_for_term(term)
all_concepts = get_all_narrower_concepts(term, depth=4)  # Adjust depth as needed

# Output alternative names
print("Alternative names:", alternative_names)
print()

# Output narrower concepts
for broader, narrower in all_concepts.items():
    print(f"Broader concept: {broader}")
    print(f"Narrower concepts: {narrower}")
    print("---")


def flatten_concepts(concepts_dict):
    flat_list = []

    def recurse_terms(term_dict):
        for term, narrower_terms in term_dict.items():
            flat_list.append(term)
            if narrower_terms:
                recurse_terms(dict.fromkeys(narrower_terms, []))  # Use an empty dict to recurse
    
    recurse_terms(concepts_dict)
    return flat_list

# Flatten the concepts dictionary
flat_list = flatten_concepts(all_concepts)


#Convert the MeSH terms to URI
def convert_to_mesh_uri(term):
    formatted_term = term.replace(" ", "_").replace(",", "_").replace("-", "_")
    return URIRef(f"http://example.org/mesh/_{formatted_term}_")


# Convert terms to URIs
mesh_terms = [convert_to_mesh_uri(term) for term in flat_list]


from rdflib import URIRef

query = """
PREFIX schema: <http://schema.org/>
PREFIX ex: <http://example.org/>

SELECT ?article ?title ?abstract ?datePublished ?access ?meshTerm
WHERE {
  ?article a ex:Article ;
           schema:name ?title ;
           schema:description ?abstract ;
           schema:datePublished ?datePublished ;
           ex:access ?access ;
           schema:about ?meshTerm .

  ?meshTerm a ex:MeSHTerm .
}
"""

# Dictionary to store articles and their associated MeSH terms
article_data = {}

# Run the query for each MeSH term
for mesh_term in mesh_terms:
    results = g.query(query, initBindings={'meshTerm': mesh_term})

    # Process results
    for row in results:
        article_uri = row['article']

        if article_uri not in article_data:
            article_data[article_uri] = {
                'title': row['title'],
                'abstract': row['abstract'],
                'datePublished': row['datePublished'],
                'access': row['access'],
                'meshTerms': set()
            }

        # Add the MeSH term to the set for this article
        article_data[article_uri]['meshTerms'].add(str(row['meshTerm']))

# Rank articles by the number of matching MeSH terms
ranked_articles = sorted(
    article_data.items(),
    key=lambda item: len(item[1]['meshTerms']),
    reverse=True
)

# Get the top 3 articles
top_3_articles = ranked_articles[:3]

# Output results
for article_uri, data in top_3_articles:
    print(f"Title: {data['title']}")
    print(f"Abstract: {data['abstract']}")
    print("MeSH Terms:")
    for mesh_term in data['meshTerms']:
        print(f"  - {mesh_term}")
    print()



# Similarity search using a KG


from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, Namespace, SKOS
import urllib.parse

# Define namespaces
schema = Namespace('http://schema.org/')
ex = Namespace('http://example.org/')
rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')

# Function to calculate Jaccard similarity and return overlapping terms
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union) if len(union) != 0 else 0
    return similarity, intersection

# Load the RDF graph
g = Graph()
g.parse('ontology.ttl', format='turtle')

def get_article_uri(title):
    # Convert the title to a URI-safe string
    safe_title = urllib.parse.quote(title.replace(" ", "_"))
    return URIRef(f"http://example.org/article/{safe_title}")

def get_mesh_terms(article_uri):
    query = """
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?meshTerm
    WHERE {
      ?article schema:about ?meshTerm .
      ?meshTerm a ex:MeSHTerm .
      FILTER (?article = <""" + str(article_uri) + """>)
    }
    """
    results = g.query(query)
    mesh_terms = {str(row['meshTerm']) for row in results}
    return mesh_terms

def find_similar_articles(title):
    article_uri = get_article_uri(title)
    mesh_terms_given_article = get_mesh_terms(article_uri)

    # Query all articles and their MeSH terms
    query = """
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?article ?meshTerm
    WHERE {
      ?article a ex:Article ;
               schema:about ?meshTerm .
      ?meshTerm a ex:MeSHTerm .
    }
    """
    results = g.query(query)

    mesh_terms_other_articles = {}
    for row in results:
        article = str(row['article'])
        mesh_term = str(row['meshTerm'])
        if article not in mesh_terms_other_articles:
            mesh_terms_other_articles[article] = set()
        mesh_terms_other_articles[article].add(mesh_term)

    # Calculate Jaccard similarity
    similarities = {}
    overlapping_terms = {}
    for article, mesh_terms in mesh_terms_other_articles.items():
        if article != str(article_uri):
            similarity, overlap = jaccard_similarity(mesh_terms_given_article, mesh_terms)
            similarities[article] = similarity
            overlapping_terms[article] = overlap

    # Sort by similarity and get top 5
    top_similar_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Print results
    print(f"Top 15 articles similar to '{title}':")
    for article, similarity in top_similar_articles:
        print(f"Article URI: {article}")
        print(f"Jaccard Similarity: {similarity:.4f}")
        print(f"Overlapping MeSH Terms: {overlapping_terms[article]}")
        print()

# Example usage
article_title = "Gingival metastasis as first sign of multiorgan dissemination of epithelioid malignant mesothelioma."
find_similar_articles(article_title)





# Function to combine titles and abstracts
def combine_abstracts(top_3_articles):
    combined_text = "".join(
        [f"Title: {data['title']} Abstract: {data['abstract']}" for article_uri, data in top_3_articles]
    )
    return combined_text

# Combine abstracts from the top 3 articles
combined_text = combine_abstracts(top_3_articles)
print(combined_text)


import pandas as pd
import openai

# Set up your OpenAI API key
api_key = "XXX"
openai.api_key = api_key


def generate_summary(combined_text):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Summarize the key information here in bullet points. Make it understandable to someone without a medical degree:\n\n{combined_text}",
        max_tokens=1000,
        temperature=0.3
    )
    
    # Get the raw text output
    raw_summary = response.choices[0].text.strip()
    
    # Split the text into lines and clean up whitespace
    lines = raw_summary.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    # Join the lines back together with actual line breaks
    formatted_summary = '\n'.join(lines)
    
    return formatted_summary

# Generate and print the summary
summary = generate_summary(combined_text)
print(summary)



# Step 3: use a vector-powered knowledge graph to test data retrieval
# Create a column to store URIs for each article


# Function to create a valid URI
def create_valid_uri(base_uri, text):
    if pd.isna(text):
        return None
    # Encode text to be used in URI
    sanitized_text = urllib.parse.quote(text.strip().replace(' ', '_').replace('"', '').replace('<', '').replace('>', '').replace("'", "_"))
    return URIRef(f"{base_uri}/{sanitized_text}")

# Add a new column to the DataFrame for the article URIs
df['Article_URI'] = df['Title'].apply(lambda title: create_valid_uri("http://example.org/article", title))


# Test that the URI works - we are able to find all mesh terms associated with a given uri

from rdflib import Graph, Namespace, URIRef

# Assuming your RDF graph (g) is already loaded

# Define namespaces
schema = Namespace('http://schema.org/')
ex = Namespace('http://example.org/')
rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')

def get_mesh_terms_for_article(graph, article_uri):
    # Define the SPARQL query using the article URI
    query = f"""
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?meshTermLabel
    WHERE {{
      <{article_uri}> a ex:Article ;
               schema:about ?meshTerm .
      ?meshTerm rdfs:label ?meshTermLabel .
    }}
    """
    
    # Execute the query
    results = graph.query(query)
    
    # Extract the MeSH terms from the results
    mesh_terms = [str(row['meshTermLabel']) for row in results]
    
    return mesh_terms

# Example usage with the provided URI
article_uri = "http://example.org/article/Expression_of_p53_and_coexistence_of_HPV_in_premalignant_lesions_and_in_cervical_cancer."
mesh_terms = get_mesh_terms_for_article(g, article_uri)

# Output the results
print(f"MeSH terms associated with the article '{article_uri}':")
for term in mesh_terms:
    print(term)



# Create the schema for new embedding which includes mesh tags and URI

class_obj = {
    # Class definition
    "class": "articles_with_abstracts_and_URIs",

    # Property definitions
    "properties": [
        {
            "name": "title",
            "dataType": ["text"],
        },
        {
            "name": "abstractText",
            "dataType": ["text"],
        },
        {
            "name": "meshMajor",
            "dataType": ["text"],
        },
        {
            "name": "Article_URI",
            "dataType": ["text"],
        },
    ],

    # Specify a vectorizer
    "vectorizer": "text2vec-openai",

    # Module settings
    "moduleConfig": {
        "text2vec-openai": {
            "vectorizeClassName": True,
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
        },
        "qna-openai": {
          "model": "gpt-3.5-turbo-instruct"
        },
        "generative-openai": {
          "model": "gpt-3.5-turbo"
        }
    },
}

client.schema.create_class(class_obj)


import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Replace infinity values with NaN and then fill NaN values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna('', inplace=True)

# Convert columns to string type
df['Title'] = df['Title'].astype(str)
df['abstractText'] = df['abstractText'].astype(str)
df['meshMajor'] = df['meshMajor'].astype(str)
df['Article_URI'] = df['Article_URI'].astype(str)


# Log the data types
logging.info(f"Title column type: {df['Title'].dtype}")
logging.info(f"abstractText column type: {df['abstractText'].dtype}")
logging.info(f"meshMajor column type: {df['meshMajor'].dtype}")
logging.info(f"Article_URI column type: {df['Article_URI'].dtype}")


with client.batch(
    batch_size=10,  # Specify batch size
    num_workers=2,   # Parallelize the process
) as batch:
    for index, row in df.iterrows():
        try:
            question_object = {
                "title": row.Title,
                "abstractText": row.abstractText,
                "meshMajor": row.meshMajor,
                "article_URI": row.Article_URI,
            }
            batch.add_data_object(
                question_object,
                class_name="articles_with_abstracts_and_URIs",
                uuid=generate_uuid5(question_object)
            )
        except Exception as e:
            logging.error(f"Error processing row {index}: {e}")


# Semantic search with vectorized KG

response = (
    client.query
    .get("articles_with_abstracts_and_URIs", ["title","abstractText","meshMajor","article_URI"])
    .with_additional(["id"])
    .with_near_text({"concepts": ["mouth neoplasms"]})
    .with_limit(10)
    .do()
)

print(json.dumps(response, indent=4))


# Extract article URIs
article_uris = [article["article_URI"] for article in response["data"]["Get"]["Articles_with_abstracts_and_URIs"]]



from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

# Define namespaces
schema = Namespace('http://schema.org/')
ex = Namespace('http://example.org/')
rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')
xsd = Namespace('http://www.w3.org/2001/XMLSchema#')

def get_articles_after_date(graph, article_uris, date_cutoff):
    # Create a dictionary to store results for each URI
    results_dict = {}

    # Define the SPARQL query using a list of article URIs and a date filter
    uris_str = " ".join(f"<{uri}>" for uri in article_uris)
    query = f"""
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?article ?title ?datePublished
    WHERE {{
      VALUES ?article {{ {uris_str} }}
      
      ?article a ex:Article ;
               schema:name ?title ;
               schema:datePublished ?datePublished .
      
      FILTER (?datePublished > "{date_cutoff}"^^xsd:date)
    }}
    """
    
    # Execute the query
    results = graph.query(query)
    
    # Extract the details for each article
    for row in results:
        article_uri = str(row['article'])
        results_dict[article_uri] = {
            'title': str(row['title']),
            'date_published': str(row['datePublished'])
        }
    
    return results_dict

date_cutoff = "2023-01-01"
articles_after_date = get_articles_after_date(g, article_uris, date_cutoff)

# Output the results
for uri, details in articles_after_date.items():
    print(f"Article URI: {uri}")
    print(f"Title: {details['title']}")
    print(f"Date Published: {details['date_published']}")
    print()


# Similarity search with vectorized KG

response = (
    client.query
    .get("articles_with_abstracts_and_URIs", ["title","abstractText","meshMajor","article_URI"])
    .with_near_object({
        "id": "37b695c4-5b80-5f44-a710-e84abb46bc22"
    })
    .with_limit(50)
    .with_additional(["distance"])
    .do()
)

print(json.dumps(response, indent=2))


# Assuming response is the data structure with your articles
article_uris = [URIRef(article["article_URI"]) for article in response["data"]["Get"]["Articles_with_abstracts_and_URIs"]]
from rdflib import URIRef

# Constructing the SPARQL query with a FILTER for the article URIs
query = """
PREFIX schema: <http://schema.org/>
PREFIX ex: <http://example.org/>

SELECT ?article ?title ?abstract ?datePublished ?access ?meshTerm
WHERE {
  ?article a ex:Article ;
           schema:name ?title ;
           schema:description ?abstract ;
           schema:datePublished ?datePublished ;
           ex:access ?access ;
           schema:about ?meshTerm .

  ?meshTerm a ex:MeSHTerm .

  # Filter to include only articles from the list of URIs
  FILTER (?article IN (%s))
}
"""


# Convert the list of URIRefs into a string suitable for SPARQL
article_uris_string = ", ".join([f"<{str(uri)}>" for uri in article_uris])

# Insert the article URIs into the query
query = query % article_uris_string

# Dictionary to store articles and their associated MeSH terms
article_data = {}

# Run the query for each MeSH term
for mesh_term in mesh_terms:
    results = g.query(query, initBindings={'meshTerm': mesh_term})

    # Process results
    for row in results:
        article_uri = row['article']

        if article_uri not in article_data:
            article_data[article_uri] = {
                'title': row['title'],
                'abstract': row['abstract'],
                'datePublished': row['datePublished'],
                'access': row['access'],
                'meshTerms': set()
            }

        # Add the MeSH term to the set for this article
        article_data[article_uri]['meshTerms'].add(str(row['meshTerm']))

# Rank articles by the number of matching MeSH terms
ranked_articles = sorted(
    article_data.items(),
    key=lambda item: len(item[1]['meshTerms']),
    reverse=True
)


# Output results
for article_uri, data in ranked_articles:
    print(f"Title: {data['title']}")
    print(f"Abstract: {data['abstract']}")
    print("MeSH Terms:")
    for mesh_term in data['meshTerms']:
        print(f"  - {mesh_term}")
    print()

# List to store the URIs of the ranked articles
ranked_article_uris = [URIRef(article_uri) for article_uri, data in ranked_articles]
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD, SKOS

# Assuming your RDF graph (g) is already loaded

# Define namespaces
schema = Namespace('http://schema.org/')
ex = Namespace('http://example.org/')
rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')

def filter_articles_by_access(graph, article_uris, access_values):
    # Construct the SPARQL query with a dynamic VALUES clause
    uris_str = " ".join(f"<{uri}>" for uri in article_uris)
    query = f"""
    PREFIX schema: <http://schema.org/>
    PREFIX ex: <http://example.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?article ?title ?abstract ?datePublished ?access ?meshTermLabel
    WHERE {{
      VALUES ?article {{ {uris_str} }}
      
      ?article a ex:Article ;
               schema:name ?title ;
               schema:description ?abstract ;
               schema:datePublished ?datePublished ;
               ex:access ?access ;
               schema:about ?meshTerm .
      ?meshTerm rdfs:label ?meshTermLabel .
      
      FILTER (?access IN ({", ".join(map(str, access_values))}))
    }}
    """
    
    # Execute the query
    results = graph.query(query)
    
    # Extract the details for each article
    results_dict = {}
    for row in results:
        article_uri = str(row['article'])
        if article_uri not in results_dict:
            results_dict[article_uri] = {
                'title': str(row['title']),
                'abstract': str(row['abstract']),
                'date_published': str(row['datePublished']),
                'access': str(row['access']),
                'mesh_terms': []
            }
        results_dict[article_uri]['mesh_terms'].append(str(row['meshTermLabel']))
    
    return results_dict

access_values = [3,5,7]
filtered_articles = filter_articles_by_access(g, ranked_article_uris, access_values)

# Output the results
for uri, details in filtered_articles.items():
    print(f"Article URI: {uri}")
    print(f"Title: {details['title']}")
    print(f"Abstract: {details['abstract']}")
    print(f"Date Published: {details['date_published']}")
    print(f"Access: {details['access']}")
    print()

# RAG with a vectorized KG

response = (
    client.query
    .get("Articles_with_abstracts_and_URIs", ["title", "abstractText",'article_URI','meshMajor'])
    .with_near_text({"concepts": ["therapies for mouth neoplasms"]})
    .with_limit(3)
    .with_generate(grouped_task="Summarize the key information here in bullet points. Make it understandable to someone without a medical degree.")
    .do()
)

print(response["data"]["Get"]["Articles_with_abstracts_and_URIs"][0]["_additional"]["generate"]["groupedResult"])


# Extract article URIs
article_uris = [article["article_URI"] for article in response["data"]["Get"]["Articles_with_abstracts_and_URIs"]]

# Function to filter the response for only the given URIs
def filter_articles_by_uri(response, article_uris):
    filtered_articles = []
    
    articles = response['data']['Get']['Articles_with_abstracts_and_URIs']
    for article in articles:
        if article['article_URI'] in article_uris:
            filtered_articles.append(article)
    
    return filtered_articles

# Filter the response
filtered_articles = filter_articles_by_uri(response, article_uris)

# Output the filtered articles
print("Filtered articles:")
for article in filtered_articles:
    print(f"Title: {article['title']}")
    print(f"URI: {article['article_URI']}")
    print(f"Abstract: {article['abstractText']}")
    print(f"MeshMajor: {article['meshMajor']}")
    print("---")


# ---

# RAG on vectorized knowledge graph with filters

response = (
    client.query
    .get("articles_with_abstracts_and_URIs", ["title", "abstractText", "meshMajor", "article_URI"])
    .with_additional(["id"])
    .with_near_text({"concepts": ["therapies for mouth neoplasms"]})
    .with_limit(20)
    .do()
)

# Assuming response is the data structure with your articles
article_uris = [URIRef(article["article_URI"]) for article in response["data"]["Get"]["Articles_with_abstracts_and_URIs"]]


from rdflib import URIRef

# Constructing the SPARQL query with a FILTER for the article URIs


query = """
PREFIX schema: <http://schema.org/>
PREFIX ex: <http://example.org/>

SELECT ?article ?title ?abstract ?datePublished ?access ?meshTerm
WHERE {
  ?article a ex:Article ;
           schema:name ?title ;
           schema:description ?abstract ;
           schema:datePublished ?datePublished ;
           ex:access ?access ;
           schema:about ?meshTerm .

  ?meshTerm a ex:MeSHTerm .

  # Filter to include only articles from the list of URIs
  FILTER (?article IN (%s))
}
"""


# Convert the list of URIRefs into a string suitable for SPARQL
article_uris_string = ", ".join([f"<{str(uri)}>" for uri in article_uris])

# Insert the article URIs into the query
query = query % article_uris_string

# Dictionary to store articles and their associated MeSH terms
article_data = {}

# Run the query for each MeSH term
for mesh_term in mesh_terms:
    results = g.query(query, initBindings={'meshTerm': mesh_term})

    # Process results
    for row in results:
        article_uri = row['article']

        if article_uri not in article_data:
            article_data[article_uri] = {
                'title': row['title'],
                'abstract': row['abstract'],
                'datePublished': row['datePublished'],
                'access': row['access'],
                'meshTerms': set()
            }

        # Add the MeSH term to the set for this article
        article_data[article_uri]['meshTerms'].add(str(row['meshTerm']))

# Rank articles by the number of matching MeSH terms
ranked_articles = sorted(
    article_data.items(),
    key=lambda item: len(item[1]['meshTerms']),
    reverse=True
)


# Output results
for article_uri, data in ranked_articles:
    print(f"Title: {data['title']}")
    print(f"Abstract: {data['abstract']}")
    print("MeSH Terms:")
    for mesh_term in data['meshTerms']:
        print(f"  - {mesh_term}")
    print()




# Function to combine titles and abstracts
def combine_abstracts(ranked_articles):
    combined_text = "".join(
        [f"Title: {data['title']} Abstract: {data['abstract']}" for article_uri, data in ranked_articles]
    )
    return combined_text

# Combine abstracts from the top 3 articles
combined_text = combine_abstracts(ranked_articles)
print(combined_text)



# Function to combine titles and abstracts into one chunk of text
def combine_abstracts(ranked_articles):
    combined_text = " ".join(
        [f"Title: {data['title']} Abstract: {data['abstract']}" for _, data in ranked_articles]
    )
    return combined_text

# Combine abstracts from the filtered articles
combined_text = combine_abstracts(ranked_articles)
print(combined_text)




# Generate and print the summary
summary = generate_summary(combined_text)
print(summary)