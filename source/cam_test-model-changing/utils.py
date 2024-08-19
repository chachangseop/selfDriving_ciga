import time

import cv2
from ultralytics.utils.plotting import Colors, Annotator


class FrameRateCalculator:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.is_fps_updated = False

    def get_fps(self):
        elapsed_time = time.time() - self.start_time
        self.frame_count += 1
        self.is_fps_updated = False

        if elapsed_time > 1:    # calculate frame every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
            self.is_fps_updated = True
            
        return self.is_fps_updated, int(self.fps)


def draw_fps(img, fps):
    """
    초당 프레임 수(FPS)를 화면에 그린다.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    ret_img = cv2.putText(img, str(fps), (7, 35), font, 0.8, (100, 255, 0), 2, cv2.LINE_AA)
    return ret_img

def draw_annotation(img, label_names, results):
    """
    탐지한 객체 박스를 화면에 그린다.
    """
    annotator = None

    for r in results:
        colors = Colors()
        annotator = Annotator(img)

        boxes = r.boxes
        for box in boxes:
            box_coordinate = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            class_index = box.cls
            color_index = int(class_index) % colors.n
            annotator.box_label(box_coordinate, label_names[int(class_index)], colors.palette[color_index])
        
    if annotator is not None:
        annotated_img = annotator.result()
    else:
        annotated_img = img.copy()
    
    return annotated_img


if __name__ == "__main__":
    print("This is not executable program.")