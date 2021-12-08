# import streamlit as st
# import mediapipe as mp
# import cv2
# import time
#
# # st.set_page_config(layout="wide")
#
#
# thres = 0.45 # Threshold to detect object
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils
#
# classNames= []
# classFile = "coco.names"
# with open(classFile,'rt') as f:
#     classNames = f.read().split('\n')
#
# configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
# weightsPath = 'frozen_inference_graph.pb'
#
# net = cv2.dnn_DetectionModel(weightsPath,configPath)
# net.setInputSize(320,320)
# net.setInputScale(1.0/ 127.5)
# net.setInputMean((127.5, 127.5, 127.5))
# net.setInputSwapRB(True)
#
# from streamlit_webrtc import (
#     AudioProcessorBase,
#     RTCConfiguration,
#     VideoProcessorBase,
#     WebRtcMode,
#     webrtc_streamer,
# )
# try:
#     from typing import Literal
# except ImportError:
#     from typing_extensions import Literal  # type: ignore
# import av
#
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )
# st.write("Press start to turn on Camera!")
# st.write("If camera doesn't turn on, click the select device button, change the camera input and reload your screen!")
#
# col = st.empty()
#
# start = col.button('Start')
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)
#
# camOn = False
#
# if start:
#     with st.spinner('Turning camera on!'):
#         time.sleep(3)
#     stop = col.button("Stop")
#     camOn = True
#
#
# while camOn:
#     _, frame = camera.read()
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     classIds, confs, bbox = net.detect(frame, confThreshold=thres)
#     if len(classIds) != 0:
#
#         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
#             print(classNames[classId-1])
#             if confidence > 0.6:
#                 cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
#                 cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
#                             cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#                 # cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
#                 # cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
#                 # Show Confidence Level
#     FRAME_WINDOW.image(img)
#     if stop:
#         st.success("Turning camera off!")
#         break
#
# # else:
# #     st.write('Camera Stopped!')
#
#
#
#
#
#
#
#

import cv2
import streamlit as st
import time


col = st.empty()

start = col.button('Turn on Camera')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

camOn = False

if start:
    with st.spinner('Turning camera on!'):
        time.sleep(3)
    stop = col.button("Turn off Camera")
    camOn = True


while camOn:
    ret, frame = camera.read()
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.rectangle(img, (5,5),(100,100), color=(0, 255, 0), thickness=2)
        cv2.putText(img, "dsa", (100,100),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        FRAME_WINDOW.image(img)
    if stop:
        break



#Info Block
st.write("If camera doesn't turn on, please ensure that your camera permissions are on!")
with st.expander("Steps to enable permission"):
    st.write("1. Click the lock button at the top left of the page")
    st.write("2. Slide the camera slider to on")
    st.write("3. Reload your page!")
