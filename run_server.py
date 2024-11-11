from flask import Flask, request, jsonify
from call_nemotron import query_model

app = Flask(__name__)


app.route("/queryllm", methods=["GET"])
def run_nemotron():

    try:
        data = request.get_json()
        query = data.get("query")
        response = query_model(input_text=query)

        return jsonify({"response":response}), 200
    
    except Exception as e:
        return jsonify({"error":"error occured while processing the request","details":str(e)}), 500
    

if __name__ == "__main__":
    app.run(port=5000,debug=True)