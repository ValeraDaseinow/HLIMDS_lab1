import cv2
import numpy as np
import time
from tensorflow import keras

isBgCaptured = 0
# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

gesture_names = {-1: 'Unknown',
                 0: 'A',
                 1: 'F',
                 2: 'W'}

model = keras.models.load_model(r'C:\intelFPGA_lite\17.0\HLIMDS\Lab1\my_model')

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def take_background():
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    time.sleep(2)
    isBgCaptured = 1
    print('Background captured')


# In[21]:


def predict_rgb_image_vgg(image):
    pred = model.predict(image)
    try:
        p = [np.where(r>0.75)[0][0] for r in pred]
    except:
        p = [-1]
    return p[0]

camera = cv2.VideoCapture(0)
camera.set(10, 200)
while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    if isBgCaptured:
        frame_without_bg = remove_background(frame)
        frame_without_bg = frame_without_bg[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        frame_without_bg = cv2.cvtColor(frame_without_bg, cv2.COLOR_BGR2GRAY)
        
        #blur = cv2.GaussianBlur(frame_without_bg, (blurValue, blurValue), 0)
        #ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('2', frame_without_bg)
    cv2.imshow('1', frame)
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit all windows at any time
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')
    elif k == 32:
        # If space bar pressed
        if isBgCaptured:
            # copies 1 channel BW image to all 3 RGB channels
            target = np.stack((frame_without_bg,) * 3, axis=-1)
            target = cv2.resize(target, (160, 160))
            target = target.reshape(1, 160, 160, 3)
            target.shape
            p = predict_rgb_image_vgg(target)
            try:
                cv2.putText(frame_without_bg, gesture_names[p], (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255))
                print(gesture_names[p])
            except:
                print("bonk")
            cv2.imshow('3', frame_without_bg)
    elif k == ord('r'):  # press 'r' to reset the background
        time.sleep(1)
        bgModel = None
        isBgCaptured = 0
        print('Reset background')
cv2.destroyAllWindows()
camera.release()
