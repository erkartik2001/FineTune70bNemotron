from flask import Flask, request, jsonify
from llm_model import LlamaModel
from flask_httpauth import HTTPBasicAuth
import os

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
llama.load_model()

@app.route('/api/call70b', methods=['POST','GET'])
@auth.login_required
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    
    response_text = llama.generate_text(prompt)
    
    return jsonify({"response": response_text})


@app.route('/api/finetune70b',methods=["POST","GET"])
@auth.login_required
def finetune70b():
    if 'file' not in request.files:
        return jsonify({"error":"No file part in request"}), 400
    
    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error":"No file selected"}), 400
    
    if not file.filename.endswith(".txt"):
        return jsonify({"error":"Only .txt files are allowed"}), 400
    
    file_path = os.path.join("temp",file.filename)
    os.makedirs("temp",exist_ok=True)
    file.save(file_path)

    try:
        llm_model.fine_tune(train_file=file_path)
        return jsonify({"message":"Training completed"}), 200
    
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    
    finally:
        os.remove(file_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
