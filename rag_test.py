from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os

repo_path = "/Users/bogle/Dev/gitcode/chatgpt-markdown/test"


loader = DirectoryLoader(
    repo_path,
    glob="**/*.md",  # Use glob pattern to match all Markdown files
    loader_cls=UnstructuredMarkdownLoader  # Specify the loader class
)

# Load documents
documents = loader.load()

# Debugging output
print(f"Number of documents loaded: {len(documents)}")
for doc in documents:
    print(f"Document Metadata: {doc.metadata}")
    print(f"Document Content: {doc.page_content[:100]}...")  # Print a snippet of the content