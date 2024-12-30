import weaviate
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from weaviate.classes.config import Configure
from weaviate.classes.config import Property, DataType


load_dotenv()
repo_path = "/Users/bogle/Dev/gitcode/chatgpt-markdown/test"
wcd_url = os.getenv("wcd_url")
wcd_api_key = os.getenv("wcd_api_key")
ollamaapi="http://192.168.5.3:11434"



# print('url',wcd_url)
# print('wcd_api_key',wcd_api_key)


loader = DirectoryLoader(
    repo_path,
    glob="**/*.md",  # Use glob pattern to match all Markdown files
    loader_cls=UnstructuredMarkdownLoader  # Specify the loader class
)

# Load documents
documents = loader.load()

# Debugging output
print(f"Number of documents loaded: {len(documents)}")


client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    
    auth_credentials=Auth.api_key(wcd_api_key),            
)

print(client.is_ready())  



client.collections.create(
    "DemoCollection",
    vectorizer_config=[
        Configure.NamedVectors.text2vec_ollama(
            name="title_vector",
            source_properties=["title"],
            api_endpoint=ollamaapi,
        )
    ],
    properties=[ 
        Property(name="fileName", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
    ]
)


client.close()  