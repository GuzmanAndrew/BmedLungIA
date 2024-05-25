# Import Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import Keras
from tensorflow.keras.preprocessing import image

# Import python files
import numpy as np
import os
from werkzeug.utils import secure_filename
from model_loader import cargarModelo

UPLOAD_FOLDER = '../images/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

port = int(os.getenv('PORT', 5000))
print("Port recognized: ", port)

# Initialize the application service
app = Flask(__name__)
CORS(app)
loaded_model, graph = cargarModelo()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a route


@app.route('/')
def main_page():
    return '¡Servicio REST activo!'


@app.route('/model/cancer/', methods=['POST'])
def default():
    data = {"success": False}
    if request.method == "POST":
        if 'file' not in request.files:
            print('No file part')
            return jsonify(data)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return jsonify(data)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            filename = UPLOAD_FOLDER + '/' + filename
            print("\nfilename:", filename)

            image_to_predict = image.load_img(filename, target_size=(224, 224))
            test_image = image.img_to_array(image_to_predict)
            test_image = np.expand_dims(test_image, axis=0)
            test_image = test_image.astype('float32')
            test_image /= 255

            with graph.as_default():
                result = loaded_model.predict(test_image)[0][0]

            prediction = 1 if result >= 0.5 else 0
            CLASSES = ['Normal', 'Cáncer']

            ClassPred = CLASSES[prediction]
            ClassProb = result

            print("Pedicción:", ClassPred)
            print("Prob: {:.2%}".format(ClassProb))
            score_formatted = "{:.2%}".format(ClassProb)

            data["predictions"] = []
            r = {"label": ClassPred, "score": score_formatted}
            data["predictions"].append(r)

            data["success"] = True

    return jsonify(data)


# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, threaded=False)
