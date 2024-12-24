# pip install weaviate-client needed to work 

import weaviate
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv


load_dotenv()

wcd_url = os.getenv("wcd_url")
wcd_api_key = os.getenv("wcd_api_key")

import weaviate

client = weaviate.connect_to_local()

print(client.is_ready())

client.close()  