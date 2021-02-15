from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("./model"))
from preprocess import *
from load import *

global model
model = init()


mean_age = 127.32431789340102
std_age = 41.18136765542683
full_path = '/storage/upload/'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = full_path


@app.route('/')
def upload_f():
    return render_template('upload.html')


def finds(file_name, gender_pre_process):
    
    if 'original' in gender_pre_process:
        file_name_to_predict = 'pre_' + file_name
        img_preprocessing(full_path, file_name, file_name_to_predict)
    else:
        file_name_to_predict = file_name

    if 'male' in gender_pre_process:
        female = np.array([0])
    else:
        female = np.array([1])
        
    img = load_img(full_path + file_name_to_predict, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = tensorflow.expand_dims(img_array, 0)  # Create batch axis
    return str(round(mean_age + std_age*(model.predict([img_array, female]))[0][0], 2))


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        gender_pre_process = request.form.to_dict()
        gender_pre_process = list(gender_pre_process.values())
        print(gender_pre_process)

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        val = finds(secure_filename(f.filename), gender_pre_process)
        return render_template('upload.html', prediction_text='Predicted age in month:- {}'.format(val))
        
if __name__ == '__main__':

    app.run(debug=True, port=8000, host='0.0.0.0')
  
