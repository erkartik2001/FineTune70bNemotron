from flask import Flask, request, jsonify
from llm_model import LlamaModel
from flask_httpauth import HTTPBasicAuth
import os
import threading

app = Flask(__name__)
auth = HTTPBasicAuth()

users = {
    "admin": "42robots"
}

@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
token = "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"
llama = LlamaModel(model_name, token)

@app.before_first_request
def initialize_model():
    llama.load_model()
    print("Base model loaded.")

@app.route('/api/call70b', methods=['POST', 'GET'])
@auth.login_required
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('maxlength',100)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        response_text = llama.generate_text(prompt,max_length)
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/finetune70b', methods=["POST", "GET"])
@auth.login_required
def finetune70b():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".txt"):
        return jsonify({"error": "Only .txt files are allowed"}), 400

    file_path = os.path.join("temp", file.filename)
    os.makedirs("temp", exist_ok=True)
    file.save(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        def fine_tune_thread():
            try:
                print("Processing training data...")
                training_data = llama.process_text(file_content)
                output_dir = "./fine_tuned_model"
                
                print("Starting fine-tuning...")
                llama.fine_tune(training_data, output_dir=output_dir)

                print("Loading fine-tuned model...")
                llama.load_fine_tuned_model(output_dir)
                print("Fine-tuned model is now active.")
            except Exception as e:
                print(f"Error during fine-tuning: {e}")

        threading.Thread(target=fine_tune_thread).start()
        return jsonify({"message": "Fine-tuning started. Model will be updated once completed."}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(file_path)

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
