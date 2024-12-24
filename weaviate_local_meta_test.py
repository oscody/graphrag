import weaviate

client = weaviate.connect_to_local()

try:
    meta_info = client.get_meta()
    print(meta_info)

finally:
    client.close()