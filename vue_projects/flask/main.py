from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)

@app.route("/receive", methods=['post'])
def form():
    file = request.files['file']
    filename = secure_filename(file.filename)
    print(file.filename)
    
    file_path = './storages/'
    # os.makedirs(file_path, exists_ok=True)
    file.save(os.path.join(file_path, filename))

    # files = request.files
    # file = files.get('file')
    
    # with open(os.path.abspath(f'./{file.filename}'), 'wb') as f:
    #     f.write(file.content)
    response = jsonify("File received and saved!")
    return response

@app.route("/uploader")
def send():
    return render_template('upload.htm')

@app.route("/")
def hello():
    return render_template('./upload.htm')

if __name__ == "__main__":
    app.run()