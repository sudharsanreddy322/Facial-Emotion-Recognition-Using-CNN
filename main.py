import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from urllib.request import Request, urlopen
from flask import Flask, render_template, Response, request, redirect, flash

from camera_feed import VideoCamera
from Graphical_Visualisation import Emotion_Analysis

# initializing the flask
app = Flask(__name__)

# When serving files, we set the cache control max age to zero number of seconds
# for refreshing the Cache
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

save_img = 'static'
check_extension_types = {'png', 'jpg', 'jpeg', 'gif'}
app.config['save_img'] = save_img


def generate_frame(camera):
    while True:
        frame = camera.get_frame()
        yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'


def allowed_file(filename):
    return ('.' in filename and filename.rsplit('.', 1)[1].lower() in check_extension_types)


@app.route('/')
def Start():
    return render_template('Start.html')


@app.route('/video_feed')
def video_feed():
    # streaming video live from webcam
    return Response(generate_frame(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/RealTime', methods=['POST'])
def RealTime():
    # real time video capturing from home page
    return render_template('RealTime.html')


@app.route('/takeimage', methods=['POST'])
def takeimage():
    # takes frame and finds emotions in it

    v = VideoCamera()
    _, frame = v.video.read()
    save_to = "static/"
    cv2.imwrite(save_to + "capture" + ".jpg", frame)

    result = Emotion_Analysis("capture.jpg")

    # if faces are not detected, return same image
    if len(result) == 1:
        return render_template('NoDetection.html', orig=result[0])

    return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2])


@app.route('/ManualUpload', methods=['POST'])
def ManualUpload():
    return render_template('ManualUpload.html')


@app.route('/uploadimage', methods=['POST'])
def uploadimage():
    if request.method == 'POST':

        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # If no file selected, submit empty file with no filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['save_img'], filename))

            result = Emotion_Analysis(filename)

            if len(result) == 1:
                return render_template('NoDetection.html', orig=result[0])

            # link = provide_url(result[3])

            return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2])


@app.route('/imageurl', methods=['POST'])
def imageurl():
    # Fetch the Image from the Provided URL
    url = request.form['url']
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    # saving the image
    webpage = urlopen(req).read()
    arr = np.asarray(bytearray(webpage), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    save_to = "static/"
    cv2.imwrite(save_to + "url.jpg", img)

    result = Emotion_Analysis("url.jpg")

    if len(result) == 1:
        return render_template('NoDetection.html', orig=result[0])

    return render_template('Visual.html', orig=result[0], pred=result[1], bar=result[2])


if __name__ == '__main__':
    app.run()
