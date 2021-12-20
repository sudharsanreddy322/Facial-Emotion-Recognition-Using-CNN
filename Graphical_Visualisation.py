import cv2
import numpy as np
import matplotlib.pyplot as plt
from load_model import FacialExpressionModel

model = FacialExpressionModel("training_model.json", "model_weights.h5")
ide_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def Emotion_Analysis(img):

    path = "static/" + str(img)
    image = cv2.imread(path)

    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scaleFactor = 1.3
    minNeighbors = 5

    faces = ide_face.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

    # when no face is detected in the frame
    if len(faces) == 0:
        return [img]

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(face, (48, 48))

        prediction = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

        Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!", "Angry": "?", "Disgust": "#", "Neutral": ".",
                   "Fear": "~"}

        Text = str(prediction) + Symbols[str(prediction)]
        Text_Color = (180, 105, 255)

        Thickness = 2
        font_scale = 1
        font_type = cv2.FONT_HERSHEY_SIMPLEX

        # inserting the text on Image
        cv2.putText(image, Text, (x, y), font_type, font_scale, Text_Color, Thickness)

        # Finding the Coordinates and Radius of Circle
        xc = int((x + x + w) / 2)
        yc = int((y + y + h) / 2)
        radius = int(w / 2)

        # Drawing the Circle on the Image
        cv2.circle(image, (xc, yc), radius, (0, 255, 0), Thickness)

        # Saving the Predicted Image
        path = "static/" + "pred" + str(img)
        cv2.imwrite(path, image)

        # List of Emotions
        EMOTIONS = ["Angry", "Disgust",
                    "Fear", "Happy",
                    "Neutral", "Sad",
                    "Surprise"]

        # finding the probability of each emotion
        probability = model.return_probabs(roi[np.newaxis, :, :, np.newaxis])

        # converting the array into list
        data = probability.tolist()[0]

        # initializing the figure for Bar Graph
        plt.figure(figsize=(8, 5))
        plt.bar(EMOTIONS, data, color='green', width=0.4)

        plt.xlabel("Types of Emotions")
        plt.ylabel("Probability")
        plt.title("Facial Emotion Recognition")

        # saving the Bar Plot
        path = "static/" + "bar_plot" + str(img)
        plt.savefig(path)

    # Returns a list containing the names of Original, Predicted, Bar Plot Images
    return [img, "pred" + img, "bar_plot" + img, prediction]
