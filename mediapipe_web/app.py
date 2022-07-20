from mediapipe_code import output_motion
from flask import Flask, Response, url_for, redirect
import cv2
import numpy as np
import mediapipe as mp
import threading

outputFrame = None
lock = threading.Lock()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('detect_motion'))

@app.route('/reload')
def detect_motion():
    global outputFrame, lock
    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not video.isOpened():
        video=cv2.VideoCapture(0)
    if not video.isOpened():
        raise IOError("Cannot open webcame") 

    while video.isOpened():
        hasFrame, frame = video.read()
        if not hasFrame:
            cv2.waitKey()
            break
        else:
            outputFrame = output_motion(frame)
            with lock:
                outputFrame = outputFrame.copy()

def gen():
    global outputFrame, lock
    # Traverse the frames of the output video stream
    while True:
      # Wait until the thread lock is acquired
        with lock:
         # Check whether there is content in the output. If there is no content, skip this process
            if outputFrame is None:
                continue

         # Compress the output to jpeg format
        (flag, jpeg) = cv2.imencode(".jpg", outputFrame)
        frame=jpeg.tobytes()
         # Make sure the output is compressed correctly
        if not flag:
            continue

        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, threaded=True)
