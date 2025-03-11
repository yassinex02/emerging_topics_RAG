import requests

def main():
    base_url = "http://localhost:8000"  # Adjust if needed

    # Upload sample texts
    texts = [
        "The capital of France is Paris. France is in Europe.",
        "Don Quixote was written by Miguel de Cervantes in the early 17th century.",
        "Python is a popular programming language created by Guido van Rossum."
    ]
    upload_payload = {"texts": texts}
    print("Uploading documents...\n")
    resp_upload = requests.post(f"{base_url}/upload", json=upload_payload)
    print("Status code /upload:", resp_upload.status_code)
    print("Response /upload:", resp_upload.json())

    # Example questions
    questions = [
        "What is the capital of France?",
        "Who created the Python language?",
        "Who wrote Don Quixote?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        generate_payload = {"new_message": {"role": "user", "content": q}}
        resp_generate = requests.post(f"{base_url}/generate", json=generate_payload)
        print("Status code /generate:", resp_generate.status_code)
        if resp_generate.ok:
            data = resp_generate.json()
            print("Generated response:", data.get("generated_text"))
        else:
            print("Error:", resp_generate.text)

if __name__ == "__main__":
    main()