
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello, world!'

@app.route('/summarize', methods=['POST']) 
def summarize():  
    text = request.form.get('text') # text that the user input
    model = request.form.get('model') # summarization model that user chooses

    return jsonify({"result": text, "model": model}) # returns a json of text


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001,debug=True)  # Enable reloader and debugger