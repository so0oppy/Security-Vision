# 실시간 보안 감시 및 침입 탐지 시스템
# -----------------------------------

import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import time

# YOLO 모델 로드
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg") # 가중치, 설정파일
layer_names = net.getLayerNames() # YOLO의 전체 레이어 이름 가져오기
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()] # net.getUnconnectedOutLayers(): 출력 레이어(감지 결과 나오는 레이어) 가져오기

# coco 클래스 로드 (사람 감지)
with open("coco.names", "r") as file: # YOLO가 학습된 80개의 객체 이름 저장
    classes = [line.strip() for line in file.readlines()]

# cap = cv2.VideoCapture(0) # 기본 카메라
video_path = '145-1_cam01_trespass01_place01_day_summer.mp4'
cap = cv2.VideoCapture(video_path) # 웹캠 대신 영상파일 사용
# output_path = '%s_output.mp4' % video_path.split('.')[0]

# 원본 영상의 프레임 속도 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / fps  # 초 단위로 변환
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter 객체 생성
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

frame_skip = 10  # 프레임을 특정 개수 건너뛰고 처리(처리 속도 향상)
frame_count = 0  # 처리한 프레임의 카운터

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Frame not read correctly.")
        break
    
    frame_count += 1

    # 매 'frame_skip'번째 프레임만 처리
    if frame_count % frame_skip != 0:
        continue  # 건너뛰기

    # print("Frame shape:", frame.shape) # Frame shape: (2160, 3840, 3)
    # print("Frame dtype:", frame.dtype) # Frame dtype: uint8

    # 출력 프레임 축소
    frame = cv2.resize(frame, (640, 480))

    # YOLO 입력 전처리
    height, width, _ = frame.shape # 프레임의 높이, 너비, 채널(RGB) 값 가져오기
    # 감시 영역 지정
    monitor_x1 = int(width * 0.55)
    monitor_x2 = int(width * 0.3)
    monitor_y = int(height * 0.4)

    # 감시 영역 시각적으로 표시 (사다리꼴)
    points = np.array([[0, monitor_y], # 상단 좌측
                    [monitor_x1, monitor_y], # 상단 우측
                    [monitor_x2, height], # 하단 우측
                    [0, height]], # 하단 좌측
                    dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0, 0, 255), thickness=2) # 테두리만 그림
    
    # 흑백 이미지라면, 컬러(RGB)로 변환
    if frame.shape[2] == 1:  # 흑백 이미지라면(1채널)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # cv2.dnn.blobFromImage: YOLO가 사용 가능하도록 프레임 정규화(0~1사이 값으로 변환)
    # (416, 416): YOLO의 입력 크기(고정)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False) 
    # print("Blob shape:", blob.shape) # Blob shape: (1, 3, 416, 416)
    
    net.setInput(blob) # YOLO에 입력 데이터 전달
    outputs = net.forward(output_layers) # YOLO가 감지한 객체 정보 outputs에 저장

    class_ids = [] 
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:] # 객체별 확률 값 가져오기 / * YOLO모델의 출력 구조(detection): [center_x, center_y, width, height, confidence, class1_score, class2_score, ...]
            class_id = np.argmax(scores) # 가장 높은 확률을 가진 객체 클래스 찾기
            confidence = scores[class_id] # 해당 클래스의 신뢰도 가져오기

            if confidence > 0.5 and classes[class_id] == "person": # 신뢰도 50% 이상, 사람일 경우만 감지
                # dectection에서의 center_x, center_y, width, height값은 정규화된 값(0~1)이기 때문에 이미지 크기(width, height) 기준으로 픽셀 단위로 변환 필요 
                # YOLO는 바운딩 박스를 이미지 크기에 대한 비율로 반환
                # center_x * width → 실제 이미지에서의 중심 x 좌표 (픽셀)
                # w * width → 실제 바운딩 박스 너비 (픽셀)
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int") # 픽셀좌표는 정수여야 함
                # YOLO는 바운딩 박스의 중심좌표(center_x, center_y)를 반환하지만,
                # OpenCV에서 사각형을 그릴 땐 좌상단 좌표와 너비/높이 사용 
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-Maximum Suppression(비최대 억제) 적용 : 가장 신뢰도 높은 박스만 남기고 겹치는 박스 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 신뢰도 50%이상만, IOU(Interaction Over Union, 교차비율) 40%이상 -> 중복판단 
    # 더 엄격한 필터링 -> 신뢰도↑, IOU↓
    # 더 많은 객체 감지 -> 신뢰도↓, IOU↑

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 침입 감지 조건: 사람이 왼쪽 30%, 높이 50% 이내로 들어오면 감지
            if (x < monitor_x1 or x < monitor_x2) and (y+h) > monitor_y:
                cv2.putText(frame, "INTRUSION DETECTED!!", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                winsound.Beep(1000, 500) # 경고음

    cv2.imshow("Person Detection", frame)
    # 프레임 간 실제 지연 시간 계산
    elapsed_time = time.time() - start_time
    delay = max(int((frame_time - elapsed_time) * 1000), 1)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()