import weaviate

client = weaviate.connect_to_local()  

client.collections.delete("DemoCollection")  


client.close()  