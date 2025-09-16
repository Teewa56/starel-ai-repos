from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from .trrain_rag_model import main
#production only
app = Flask(__name__)

CORS(app, resources={r"/ask": {"origins": "htttps://starel-frontend.vercel.app"}})

@app.route("/ask", methods=['POST'])
def ask():
    try:
        user_prompt = request.json.get("prompt", "")
        if not user_prompt: 
            return jsonify({"error": "user prompt not specified"})

        response = main(user_prompt)
    except Exception as e:
        print("Error while getting response from model: ",e)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)