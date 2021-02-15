import tensorflow
import os

def init():
    json_file = open('/storage/upload/model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()

    # use Keras model_from_json to make a loaded model
    loaded_model = tensorflow.keras.models.model_from_json\
                                                (loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("/storage/upload/model.h5")
    print("Loaded Model from disk")

    # compile and evaluate loaded model
    loaded_model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return loaded_model
