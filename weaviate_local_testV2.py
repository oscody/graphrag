import weaviate
from weaviate.classes.config import Configure
import os


client = weaviate.connect_to_local()                                

print(client.is_ready())  


collection_name = "DemoCollection"


llmQuestion = "How can you find the process ID (PID) of JarvisV2"

if client.collections.exists(collection_name):
    client.collections.delete(collection_name)


collection = client.collections.create(
    collection_name,
    vectorizer_config=Configure.Vectorizer.text2vec_ollama(     # Configure the Ollama embedding integration
        api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
        model="nomic-embed-text",                               # The model to use
    ),
    generative_config=Configure.Generative.ollama(              # Configure the Ollama generative integration
        api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
        model="llama3.2",                                       # The model to use
    )
)


docs_dir = "/Users/bogle/Dev/gitcode/chatgpt-markdown/test"

markdown_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]


with collection.batch.dynamic() as batch:
    for filename in markdown_files:
        with open(os.path.join(docs_dir, filename), "r") as f:
            content = f.read()
            batch.add_object(
                {
                    "content": content,
                    "class_name": filename.split(".")[0],
                }
            )
            print(f"Added {filename.split('.')[0]} to Weaviate")

results = collection.query.near_text(
    query=llmQuestion,
    limit=1
)
print(results.objects[0].properties)



query=llmQuestion
results = collection.generate.near_text(
    query=query,
    limit=2,
    grouped_task=f"Answer the question: {query}? only using the given context in {{text}}"
)
print(f"------llm answer------",results.generated)


client.close()  