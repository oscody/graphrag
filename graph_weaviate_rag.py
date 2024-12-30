# graph_weaviate_rag.py

import os
import frontmatter
from datetime import datetime, timezone

import weaviate
from weaviate.classes.config import Property, DataType
from weaviate.classes.config import Configure

collection_name = "MarkdownDocument"
directory = "/Users/bogle/Dev/gitcode/chatgpt-markdown/test/"


# --- Weaviate Client Usage in a single session ---
client = weaviate.connect_to_local()
try:
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    markdown_collection = client.collections.create(
        collection_name,
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(
            api_endpoint="http://host.docker.internal:11434",
            model="nomic-embed-text",
        ),
        generative_config=Configure.Generative.ollama(
            api_endpoint="http://host.docker.internal:11434",
            model="llama3.2",
        ),
        properties=[
            Property(name="title", data_type=DataType.TEXT),
            Property(name="date", data_type=DataType.DATE),
            Property(name="tags", data_type=DataType.TEXT_ARRAY, vectorize_property_name=True),
            Property(name="body", data_type=DataType.TEXT),
        ]
    )

    markdown_collection = client.collections.get("MarkdownDocument")

    # Insert data
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            with open(os.path.join(directory, filename), 'r') as file:
                content = frontmatter.load(file)
                title = content.get('title', 'Untitled')

                # Convert to string in case frontmatter returns a datetime
                date_str = str(content.get('date', datetime.now().date().isoformat()))
                try:
                    date_obj = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
                except ValueError:
                    date_obj = datetime.now(timezone.utc)

                tags = content.get('tags', [])
                body = content.content

                markdown_collection.data.insert({
                    "title": title,
                    "date": date_obj.isoformat(),
                    "tags": tags,
                    "body": body,
                })

                print(f"Added {filename.split('.')[0]} to Weaviate")

    markdown_collection = client.collections.get("MarkdownDocument")

    # Query example frist query should return no found results secondquery should return found results
    query = "give me 5 eight-grade spelling be words mentioned in eight Grade Spelling Bee document "
    # query = "give me 5 sixth-grade spelling be words mentioned in  6th Grade Spelling Bee document "
    response = markdown_collection.query.near_text(
        query=query,
        limit=2
    )

    # for obj in response.objects:
    #     print(f"Title: {obj.properties['title']}")
    #     print(f"Date: {obj.properties['date']}")
    #     print(f"Tags: {', '.join(obj.properties['tags'])}")
    #     print(f"Body: {obj.properties['body'][:200]}...")
    #     print()


    results = markdown_collection.generate.near_text(
        query=query,
        limit=2,
        grouped_task=f"Answer the question: {query}? only using the given context in {{text}}"
    )
    print(f"------llm answer------",results.generated)

finally:
    # Ensure the client is closed to avoid ResourceWarnings
    client.close()