# pip install weaviate-client needed to work 

import weaviate
from weaviate.classes.init import Auth
import os
from dotenv import load_dotenv


load_dotenv()

wcd_url = os.getenv("wcd_url")
wcd_api_key = os.getenv("wcd_api_key")

# print('url',wcd_url)
# print('wcd_api_key',wcd_api_key)

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    
    auth_credentials=Auth.api_key(wcd_api_key),            
)

print(client.is_ready())

client.close()  