import ollama

# Initialize the Ollama client
ollama_client = ollama.Client()

# Define the model to use
model_name = "llama3.2"

# Define a simple prompt
prompt = "Why is the sky blue answer in 1 sentence?"

# Prepare the messages with role specification
messages = [
    {"role": "user", "content": prompt}
]

try:
    # Send the request to the Ollama model
    response = ollama_client.chat(
        model=model_name,
        messages=messages
    )

    # Extract and print the response content
    answer = response['message']['content'].strip()
    print(f"Model response: {answer}")

except Exception as e:
    print(f"An error occurred: {e}")