#Title: OBJECT CLASSIFIER
#Developer: Vishwas Puri
#Purpose: A program that uses your webcam to find and detect many objects, shown in your camera.

#It uses Mobile net SSD and a famous categorized data set called “coco.names”.

#This program is made using python supported by streamlit.
import streamlit as st
import mediapipe as mp
import cv2

#defined the face recognition model
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
st.set_page_config(layout="wide")

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
st.write("Press start to turn on camera and show it some objects!")

#defining mediapipe's inbuilt pose recogignition models
thres = 0.45  # Threshold to detect object
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

#getting all the cateogarized names from the coco.names data set
classNames = []
classFile = "coco.names"
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

#the classification models are being defined
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

#defining the quality of the detection using its weight and config files, respectively
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def objectClassifier():
    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            #get details of the detection
            classIds, confs, bbox = net.detect(img, confThreshold=thres)
            if len(classIds) != 0:
                #draw a rectangle over the object and write its catogarized name over it
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if confidence > 0.6:
                        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        # cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        # Show Confidence Level


            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # setting up streamlit camera configuration
    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "style": {"margin": "0 100", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )

    # Info Block
    st.write("If camera doesn't turn on, please ensure that your camera permissions are on!")
    with st.expander("Steps to enable permission"):
        st.write("1. Click the lock button at the top left of the page")
        st.write("2. Slide the camera slider to on")
        st.write("3. Reload your page!")

    st.subheader("Possible Output Objects")
    st.write("'Coco.names' Data List, https://github.com/pjreddie/darknet/blob/master/data/coco.names")
if __name__ == "__main__":
    objectClassifier()


