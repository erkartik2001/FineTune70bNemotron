from flask import Flask, request, jsonify
from llm_model import LLMModel
from flask_httpauth import HTTPBasicAuth

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

llm_model = LLMModel("nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
llm_model.load_model()

@app.route('/api/call70b', methods=['POST'])
@auth.login_required
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    
    response_text = llm_model.query_model(prompt)
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
