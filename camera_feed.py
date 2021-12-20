import cv2
import numpy as np
from load_model import FacialExpressionModel

model = FacialExpressionModel("training_model.json", "model_weights.h5")
# loading the haarcascade face identifier
ide_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)  # takes the video feed from webcam

    def __del__(self):
        self.video.release()  # to release the video feed

    def get_frame(self):
        # reading the video and in the form of frames
        _, frame = self.video.read()
        # converting the Color image to Gray Scale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # image size is reduced by 30% at each image scale to specific size
        scaleFactor = 1.3
        minNeighbors = 5

        faces = ide_face.detectMultiScale(gray_frame, scaleFactor, minNeighbors)

        # iterating through the detected faces
        for (x, y, w, h) in faces:
            # cropping the face from whole image
            face = gray_frame[y:y + h, x:x + w]
            # resize the image accordingly to use pretrained model
            face = cv2.resize(face, (48, 48))
            # predict the emotion
            # print(face.shape)
            prediction = model.predict_emotion(face[np.newaxis, :, :, np.newaxis])
            Symbols = {"Happy": ":)", "Sad": ":}", "Surprise": "!!",
                       "Angry": "?", "Disgust": "#", "Neutral": ".", "Fear": "~"}

            Text = str(prediction) + Symbols[str(prediction)]
            Text_Color = (0, 255, 0)
            Thickness = 2
            font_size = 1
            font_type = cv2.FONT_HERSHEY_SIMPLEX

            # place the text on image
            cv2.putText(frame, Text, (x, y), font_type, font_size, Text_Color, Thickness)

            # finding the coordinates and radius of circle
            xc = int((x + x + w) / 2)
            yc = int((y + y + h) / 2)
            radius = int(w / 2)

            # Drawing the Circle on the Image
            cv2.circle(frame, (xc, yc), radius, (0, 255, 0), Thickness)
        # cv2.imshow("image", frame)
        # cv2.waitKey(10000)
        # Encoding the Image into a memory buffer
        _, jpeg = cv2.imencode('.jpg', frame)

        # Returning the image as a bytes object
        return jpeg.tobytes()


# x = VideoCamera().get_frame()
