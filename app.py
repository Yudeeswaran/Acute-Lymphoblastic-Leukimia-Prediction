
from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
vgg16_model = None
svm_model = None

def load_models():
    global vgg16_model, svm_model
    try:
        # Load the pre-trained VGG16 model
        vgg16_model = load_model('VGG16.keras')
        # Load the SVM model from the .pkl file
        with open('model.pkl', 'rb') as f:
            svm_model = pickle.load(f)
    except Exception as e:
        print("Error loading models:", e)

def map_stage(prediction):
    if prediction == 1:
        return "Early Stage"
    elif prediction == 2:
        return "Pre-B Stage"
    elif prediction == 3:
        return "Pro-B Stage"
    elif prediction == 0:
        return "Benign Stage"
    else:
        return "Unknown Stage"

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file.filename == '':
                return render_template('predict.html', prediction='No file selected')

            if file:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Read the image file
                img = image.load_img(file_path, target_size=(224, 224))
                img_array = image.img_to_array(img)

                # Expand the dimensions to match the input shape of VGG16
                img_array_expanded = np.expand_dims(img_array, axis=0)

                # Preprocess the image for VGG16
                img_array_preprocessed = vgg16_preprocess(img_array_expanded)

                # Make predictions using VGG16
                vgg16_predictions = vgg16_model.predict(img_array_preprocessed)

                # Flatten the predictions
                vgg16_predictions_flat = vgg16_predictions.flatten().reshape(1, -1)

                # Pass the predictions through SVM model
                svm_predictions = svm_model.predict(vgg16_predictions_flat)

                # Map the predictions to stage labels
                formatted_predictions = map_stage(svm_predictions[0])

                return render_template('predict.html', prediction=formatted_predictions, image_url=file_path)
        except Exception as e:
            print("Prediction error:", e)
            return render_template('predict.html', prediction='Prediction error occurred')
    return render_template('predict.html')

@app.route('/learn')
def learn_page():
    return render_template('learn.html')

@app.route('/support')
def support_page():
    return render_template('support.html')

@app.route('/bc')
def page1():
    return render_template('breast_cancer.html')

@app.route('/cc')
def page5():
    return render_template('co_cancer.html')

@app.route('/lc')
def page2():
    return render_template('lung.html') 

@app.route('/pc')
def page3():
    return render_template('prostate.html')

@app.route('/ALL')
def page4():
    return render_template('ALL.html')

@app.route('/pac')
def page6():
    return render_template('pancreatic_cancer.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    load_models()
    app.run(debug=True)

