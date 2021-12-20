import numpy as np
from tensorflow.keras.models import model_from_json  # import the saved JSON format model


class FacialExpressionModel(object):
    Emotions_List = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, json_model, weights_model):
        # load the model stored during Training
        with open(json_model, "r") as json_file:
            load_json_model = json_file.read()
            self.loaded_model = model_from_json(load_json_model)

        # load the stored weight model
        self.loaded_model.load_weights(weights_model)

    def predict_emotion(self, img):
        # emotion predicting using pre-trained model
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.Emotions_List[np.argmax(self.preds)]

    def return_probabs(self, img):
        # returns the probabilities of each emotion
        self.preds = self.loaded_model.predict(img)
        return self.preds
