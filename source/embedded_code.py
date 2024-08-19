import threading
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from gpiozero import PWMOutputDevice, DigitalOutputDevice
import os
import time

os.environ['QT_QPA_PLATFORM'] = 'xcb'

PWMA = 13
AIN1 = 19
AIN2 = 26
PWMB = 12
BIN1 = 20
BIN2 = 16

L_Motor = PWMOutputDevice(PWMA)
R_Motor = PWMOutputDevice(PWMB)

AIN1_pin = DigitalOutputDevice(AIN1)
AIN2_pin = DigitalOutputDevice(AIN2)
BIN1_pin = DigitalOutputDevice(BIN1)
BIN2_pin = DigitalOutputDevice(BIN2)

speedSet = 0.5
classNames = {0: 'normal', 1:'burned'}

camera = cv2.VideoCapture(2)
camera2 = cv2.VideoCapture(0)
camera.set(3, 640)
camera.set(4, 480)
camera2.set(3, 640)
camera2.set(4, 480)

image = None
image2 = None
image_lock = threading.Lock()
image_ok = False
box_size = 0
carState = "stop"
running = True
imagednn = None
_, annotated_frame = camera.read()

def motor_control(command, speed=0.7):
    if command == "go":
        L_Motor.value = speed
        AIN2_pin.on()
        AIN1_pin.off()
        R_Motor.value = speed
        BIN2_pin.on()
        BIN1_pin.off()
    elif command == "back":
        L_Motor.value = speed
        AIN2_pin.off()
        AIN1_pin.on()
        R_Motor.value = speed
        BIN2_pin.off()
        BIN1_pin.on()
    elif command == "stop":
        L_Motor.value = 0
        R_Motor.value = 0
    elif command == "right":
        L_Motor.value = speed
        AIN2_pin.on()
        AIN1_pin.off()
        R_Motor.value = speed
        BIN2_pin.off()
        BIN1_pin.on()
    elif command == "left":
        L_Motor.value = speed
        AIN2_pin.off()
        AIN1_pin.on()
        R_Motor.value = speed
        BIN2_pin.on()
        BIN1_pin.off()
        

def id_class_name(class_id, classes):
    return classes.get(class_id, 'Unknown')

def img_preprocess(image):
    height, _, _ = image.shape
    image = image[:,:,:]
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = cv2.resize(image, (128,96))
    image = cv2.GaussianBlur(image,(5,5),0)
    #_, image = cv2.threshold(image,160,255,cv2.THRESH_BINARY_INV)
    image = image / 255
    return image

def capture_and_preprocess():
    global image, image2, image_ok, imagednn
    while running:
        ret1, img1 = camera.read()
        ret2, img2 = camera2.read()
        with image_lock:
            if ret1:
                img1 = img1[:,:,:]
                imagednn = img1.copy()
            if ret2:
                image2 = img2
            image_ok = ret1 and ret2

def opencvdnn_thread():
    global image_ok, box_size, carState, running, imagednn
    global annotated_frame
    file_path = '/home/test/test/test_auto_stop/'
    i=0
    model = YOLO("/home/test/model-test/models/ciga_changseop_ncnn_model")
    while running:
        with image_lock:
            if image_ok:
                results = model(imagednn)
                annotated_frame = imagednn

                for result in results:
                    try:
                        #print(result.boxes.conf[result.boxes.conf>0.5])
                        
                        annotated_frame = result.plot()
                        if (result.boxes.conf>0.5).any()  and carState == 'go':
                            carState = 'stop'
                            print('auto stop')
                            cv2.imwrite("%s_%05d.png" % (file_path, i), annotated_frame)
                            i += 1
                    except Exception as e:
                        print(f"Error plotting results: {e}")
                    
            image_ok = False

def main():
    global carState, running, imagednn, image2
    global annotated_frame
    #model_path = '/home/test/Downloads/AI_CAR/model/lane_navigation_final.h5'
    model_path = '/home/test/Downloads/model.h5'
    model = load_model(model_path)
    pre_state = None
    cur_state = None
    try:
        while running:
            with image_lock:
                # Check if both images are ready
                if image2 is not None and imagednn is not None:
                    preprocessed = img_preprocess(image2)
                    X = np.asarray([preprocessed])
                    steering_angle = int(model.predict(X)[0])
                    print("predict angle:", steering_angle)
                    if carState == "go":
                        if 67 <= steering_angle <= 105:
                            print("go")
                            cur_state = "go"
                            if pre_state != cur_state :
                                motor_control("go", 1.0)
                                time.sleep(0.2)
                            motor_control("go", speedSet)
                            pre_state = "go"
                        elif steering_angle > 105:
                            print("right")
                            motor_control("right", 1.0)
                            time.sleep(0.2)
                            motor_control("go", 0.7)
                            pre_state = "right"
                        elif steering_angle < 67:
                            print("left")
                            motor_control("left", 1.0)
                            time.sleep(0.2)
                            motor_control("go", 0.7)
                            pre_state = "left"
                    elif carState == "stop":
                        motor_control("stop")

                    #cv2.imshow('preprocessed', preprocessed)
                    cv2.imshow('imagednn', annotated_frame)

                    keyValue = cv2.waitKey(1)
                    if keyValue == ord('q'):
                        running = False
                        break
                    elif keyValue == 82:
                        print("go")
                        carState = "go"
                    elif keyValue == 84:
                        print("stop")
                        carState = "stop"
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        running = False

if __name__ == '__main__':
    capture_thread = threading.Thread(target=capture_and_preprocess)
    dnn_thread = threading.Thread(target=opencvdnn_thread)

    capture_thread.start()
    dnn_thread.start()

    main()

    capture_thread.join()
    dnn_thread.join()
    camera.release()
    camera2.release()
    cv2.destroyAllWindows()
