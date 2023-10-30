from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('model.h5')
target_img = os.path.join(os.getcwd(), 'static/images')

@app.route('/')
def index_view():
    return render_template('index.html')

# Allow files with extensions: png, jpg, and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXT

# Function to load and prepare the image in the right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join('static/images', filename)
        file.save(file_path)
        img = read_image(file_path)
        class_prediction = model.predict(img)
        class_index = np.argmax(class_prediction, axis=1)

        # Map class indices to fruit names
        fruit_classes = ["Apple", "Banana", "Orange"]
        fruit = fruit_classes[class_index[0]]

        return render_template('predict.html', fruit=fruit, prob=class_prediction, user_image=file_path)
    else:
        return "Unable to read the file. Please check the file extension"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)
