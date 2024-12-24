import weaviate
from weaviate.classes.config import Configure

client = weaviate.connect_to_local()                                

print(client.is_ready())  


# client.collections.create(
#         "DemoCollection",
#         vectorizer_config=[
#             Configure.NamedVectors.text2vec_ollama(
#                 name="title_vector",
#                 source_properties=["title"],
#                 api_endpoint=ollamaapi,
#             ).model_dump(),  # Use model_dump() for Pydantic V2 compatibility
#             Configure.Generative.ollama(
#                 api_endpoint=ollamaapi,
#                 model="llama3.2"
#             ).model_dump(),  # Use model_dump() for Pydantic V2 compatibility
#         ],
#         properties=[
#             Property(name="fileName", data_type=DataType.TEXT),
#             Property(name="content", data_type=DataType.TEXT),
#         ]
# )


client.collections.create(
    name="DemoCollection",
    vectorizer_config=Configure.Vectorizer.text2vec_ollama(     # Configure the Ollama embedding integration
        api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
        model="nomic-embed-text",                               # The model to use
    ),
    generative_config=Configure.Generative.ollama(              # Configure the Ollama generative integration
        api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
        model="llama3.2",                                       # The model to use
    )
)


demoCollection = client.collections.get("DemoCollection")



client.close()  