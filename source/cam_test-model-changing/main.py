import sys
import argparse

import cv2
from pathlib import Path
from ultralytics import YOLO
import yaml

import utils
from utils import FrameRateCalculator


########## 상수 ##########
DEFAULT_BASE_MODEL_PATH = "./models/yolov8n.pt"
DEFAULT_MODEL_FORMAT = "yolo"
DEFAULT_IMG_SIZE = 640


########## 메인 ##########
if __name__ == "__main__":
    ##### 전달 인자 확인 #####
    # debug: 디버깅 활성화
    # gui: 카메라 이미지를 윈도우로 표시한다
    # model: 사용할 모델 경로
    # format: 사용할 라이브러리
    # size: 추론 모델에 넘길 이미지 크기
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("-m", "--model", dest="model", action="store")
    parser.add_argument("-f", "--format", dest="format", action="store")
    parser.add_argument("-s", "--size", dest="size", action="store")
    args = parser.parse_args()
    
    ##### 기본값 설정 #####
    base_model_path = Path(args.model) if args.model else Path(DEFAULT_BASE_MODEL_PATH)
    model_format = args.format if args.format else DEFAULT_MODEL_FORMAT
    img_size = int(args.size) if args.size else DEFAULT_IMG_SIZE

    # PyTorch 모델 불러오기
    if args.model is not None and not base_model_path.exists():
        sys.exit(f"Error : Model not exist > {base_model_path}")
    if base_model_path.suffix != ".pt":
        sys.exit(f"Error : Use PyTorch model(.pt) >> {base_model_path}")
    
    base_model = YOLO(base_model_path)
    label_map = base_model.names

    # 모델 변환
    # OpenVINO
    if model_format == "openvino":
        model_path = Path(base_model_path.parent, f"{base_model_path.stem}_openvino_model")
        model_size = img_size
        if model_path.exists():
            with open(str(model_path) + "/metadata.yaml", "r") as f:
                yaml_data = yaml.safe_load(f)
                model_size = yaml_data["imgsz"][0]
            if model_size != img_size:
                base_model.export(format="openvino", imgsz=img_size)
        else:
            base_model.export(format="openvino", imgsz=img_size)
        model = YOLO(model_path)
    # NCNN
    # elif model_format == "ncnn":
    #     model_path = Path(base_model_path.parent, f"{base_model_path.stem}_ncnn_model")
    #     if model_path.exists():
    #         with open(str(model_path) + "/metadata.yaml", "r") as f:
    #             yaml_data = yaml.safe_load(f)
    #             model_size = yaml_data["imgsz"][0]
    #         if model_size != img_size:
    #             base_model.export(format="ncnn", imgsz=img_size)
    #     else:
    #         base_model.export(format="ncnn", imgsz=img_size)
    #     sys.exit("TEST")
    # TensorFlow Lite
    # elif model_format == "tflite":
    #     base_model.export(format="tflite", dynamic=True)
    #     sys.exit("TEST")
    # ONNX
    elif model_format == "onnx":
        model_path = Path(base_model_path.parent, f"{base_model_path.stem}.onnx")
        if not model_path.exists():
            base_model.export(format="onnx", dynamic=True)
        model = YOLO(model_path)
    # YOLO
    elif model_format == "yolo":
        model = base_model
    else:
        sys.exit(f"Error : Not supported format. Use 'openvino', 'tflite', 'onnx' or 'yolo' >> {model_format}")
    
    print(f"\nProgram start: {base_model_path}, format={model_format}, imgsz={img_size}\n")

    # 추론  
    cap = cv2.VideoCapture(0)
    fps_calc = FrameRateCalculator()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            is_frame_updated, fps = fps_calc.get_fps()

            results = model(frame, imgsz=img_size, verbose=args.debug)

            if args.gui:
                annotated_frame = utils.draw_annotation(frame, label_map, results)
                result_frame = utils.draw_fps(annotated_frame, fps)
                cv2.imshow("Press q to close", result_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    
    cap.release()
    cv2.destroyAllWindows()
