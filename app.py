from flask import Flask, render_template, request, redirect, url_for
import os
from plate_recognition.detector import process_image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message="Không có file nào được tải lên")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message="Không file nào được chọn")
        if not allowed_file(file.filename):
            return render_template('index.html', message="File không hợp lệ. Vui lòng chọn ảnh jpg, jpeg, png, bmp.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result_data = process_image(filepath)
        plates = result_data.get('plates', [])

        if not plates:
            return render_template('index.html', message="Không có biển số nào được nhận diện")

        if all(not plate.get("text") for plate in plates):
            return render_template('index.html', **result_data, message="Không nhận diện được ký tự nào từ biển số")

        return render_template('index.html', **result_data)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)