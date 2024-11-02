# import cv2
# import time
# import requests


# url = "http://192.168.190.26/stream"
# timeout = 10  # Thời gian chờ trước khi cảnh báo (s)

# def detect_person():
#     first_seen = None
#     while True:
#         img_resp = requests.get(url)
#         img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
#         frame = cv2.imdecode(img_arr, -1)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         people_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
#         people = people_cascade.detectMultiScale(gray, 1.1, 4)

#         if len(people) > 0:
#             if first_seen is None:
#                 first_seen = time.time()
#             elif time.time() - first_seen > timeout:
#                 print("Cảnh báo: Phát hiện người trong khu vực!")
#         else:
#             first_seen = None

#         for (x, y, w, h) in people:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.imshow("ESP32-CAM", frame)
#         if cv2.waitKey(1) == 27:
#             break
#     cv2.destroyAllWindows()

# detect_person()



import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
 
url='http://192.168.190.26/cam-hi.jpg'
im=None
 
def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
 
        cv2.imshow('live transmission',im)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
        
def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        im = cv2.imdecode(imgnp,-1)
 
        bbox, label, conf = cv.detect_common_objects(im)
        im = draw_bbox(im, bbox, label, conf)
 
        cv2.imshow('detection',im)
        key=cv2.waitKey(5)
        if key==ord('q'):
            break
            
    cv2.destroyAllWindows()
 
 
 
if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
            f1= executer.submit(run1)
            f2= executer.submit(run2)

