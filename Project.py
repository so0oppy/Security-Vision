# 실시간 보안 감시 및 침입 탐지 시스템
# -----------------------------------

import winsound
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from ultralytics import YOLO
import torch
from torchvision import models, transforms
from PIL import Image
from scipy.spatial.distance import cosine
import time

# YOLO 모델 로드
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg") # 가중치, 설정파일
layer_names = net.getLayerNames() # YOLO의 전체 레이어 이름 가져오기
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()] # net.getUnconnectedOutLayers(): 출력 레이어(감지 결과 나오는 레이어) 가져오기

# coco 클래스 로드 (사람 감지)
with open("coco.names", "r") as file: # YOLO가 학습된 80개의 객체 이름 저장
    classes = [line.strip() for line in file.readlines()]

# ResNet 모델 로드
model = models.resnet50(pretrained = True)
model.eval() # 평가 모드로 전환

# 이미지 전처리
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # 배치 차원 추가

# 의상 특징 추출 함수
def get_clothing_feature(image):
    image = Image.fromarray(image)  # OpenCV 이미지 -> PIL 이미지 변환
    input_tensor = preprocess_image(image)
    
    with torch.no_grad():  # 그래디언트 계산 방지
        output = model(input_tensor)  # 모델을 통한 예측
    return output

clothing_feature = None
clothing_features = set() # 중복 저장 피하기 위함
cnt = 0 # 같은 사람이 머뭇거린 횟수

# Optical Flow 설정
prev_frame = None # 이전 프레임
prev_gray = None # 이전 프레임의 흑백 이미지 저장 변수
stagnant_threshold = 4 
stagnant_frames = 0  

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

def compute_iou(box1, box2):
    # IOU(Intersection Over Union) 계산 : 두 바운딩 박스의 겹치는 정도
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Frame not read correctly.")
        break
    
    stagnant_detected = False # 머뭇거림 여부 초기화
    intrusion_detected = False # 감시 구역 내 진입 여부 초기화

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
    
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # results = pose.process(frame_rgb)

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

            if confidence > 0.3 and classes[class_id] == "person": # 신뢰도 30% 이상, 사람일 경우만 감지
                # dectection에서의 center_x, center_y, width, height값은 정규화된 값(0~1)이기 때문에 이미지 크기(width, height) 기준으로 픽셀 단위로 변환 필요 
                # YOLO는 바운딩 박스를 이미지 크기에 대한 비율로 반환
                # center_x * width → 실제 이미지에서의 중심 x 좌표 (픽셀)
                # w * width → 실제 바운딩 박스 너비 (픽셀)
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int") # 픽셀좌표는 정수여야 함
                # YOLO는 바운딩 박스의 중심좌표(center_x, center_y)를 반환하지만,
                # OpenCV에서 사각형을 그릴 땐 좌상단 좌표와 너비/높이 사용 
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bbox = (x, y, w, h)
                boxes.append(bbox)
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-Maximum Suppression(비최대 억제) 적용 : 가장 신뢰도 높은 박스만 남기고 겹치는 박스 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) # 신뢰도 50%이상만, IOU(Interaction Over Union, 교차비율) 40%이상 -> 중복판단 
    # 더 엄격한 필터링 -> 신뢰도↑, IOU↓
    # 더 많은 객체 감지 -> 신뢰도↓, IOU↑

    if len(indexes) > 0:
        # 사람 감지 루프
        # for obj_id, (tracker, bbox, last_seen) in tracking_objects.items():
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            # (x, y, w, h) = map(int, bbox)
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 침입 감지 조건1: 머뭇거림 감지---------------------------------------
            if prev_frame is None:
                prev_frame = frame
            # 전체 프레임을 그레이스케일로 변환
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 사람 영역만 추출
            person_prev_gray = prev_gray[y:y+h, x:x+w]
            person_frame_gray = frame_gray[y:y+h, x:x+w]

            # cv2.calcOpticalFlowFarneback() 함수는 두 그레이스케일 이미지(prev_gray, gray) 간의 Optical Flow 계산 
            # Farneback 알고리즘을 사용하여 흐름 추정, 각 파라미터는 흐름 계산의 정확도와 속도에 영향을 미침
            flow = cv2.calcOpticalFlowFarneback(person_prev_gray, person_frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # flow는 각 화살표의 (x, y) 벡터를 포함하는 배열
            # cv2.cartToPolar() 함수는 이 벡터를 Polar 좌표계로 변환하여 magnitude(크기)와 angle(각도)을 구함 
            # 이 때 사용되는 flow[..., 0]과 flow[..., 1]은 각각 x, y 방향의 흐름을 나타냄
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            avg_movement = np.mean(magnitude) # magnitude의 평균 움직임 구함
            
            if avg_movement < stagnant_threshold: # 머뭇거림 처리
                stagnant_frames += 1 # 머뭇거리는 프레임 수 계산
            else:
                stagnant_frames = 0  # 머뭇거림 상태 종료

            stagnant_detected = stagnant_frames >= 15  # 15프레임 이상 머뭇거리면 감지

            if stagnant_detected == True:
                # 사람의 의상 특징 추출
                image = frame[y:y+h, x:x+w]  # 사람의 bounding box에서 이미지 추출
                clothing_feature = get_clothing_feature(image)
                clothing_features.add(clothing_feature)
                cv2.putText(frame, "Loitering Detected", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            prev_frame = frame # 현재 프레임 -> 다음 프레임 때의 이전 프레임으로 설정

            # 침입 감지 조건2: 사람이 구역 내에 들어가면 감지----------------------
            if x < monitor_x1 and (y+h) > monitor_y:
                new_image = frame[y:y+h, x:x+w]  # 사람의 bounding box에서 이미지 추출

                # 저장된 의상 특징 벡터 (이전에 추출한 벡터)
                if clothing_features: # 비어있지 않다면
                    new_clothing_feature = get_clothing_feature(new_image).flatten()  # 한 번만 추출
                    min_distance = float('inf')  # 최소 거리 추적

                    stored_features = np.array([f.flatten() for f in clothing_features])  # 벡터 리스트 변환
                    if stored_features.shape[0] > 0:  # 저장된 특징이 있을 때만 비교
                        # 벡터 거리 계산 (numpy 연산)
                        dot_products = np.dot(stored_features, new_clothing_feature)  # 내적
                        norms = np.linalg.norm(stored_features, axis=1) * np.linalg.norm(new_clothing_feature)  # 정규화
                        distances = 1 - (dot_products / norms)  # 코사인 거리
                        min_distance = np.min(distances)  # 가장 가까운 거리 선택

                    # 유사도가 일정 기준 이하일 경우, 같은 사람으로 판단
                    if min_distance < 0.5:
                        print("서성이던 사람과 같은 사람입니다.")
                        intrusion_detected = True  # 침입 감지
                        cv2.putText(frame, "Intrusion Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        print("다른 사람입니다.")
                
            # --------------------------------------------------------------------------
            # 두 조건 모두 만족 시, 침입으로 간주하고 경고문구 & 경고음 발생
            if  intrusion_detected and stagnant_detected:
                cv2.putText(frame, "WARNING!", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "(Intrusion who was loitering)", (230, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                winsound.Beep(500, 1000)

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

##
# 머리, 상체 등 신체 일부만 보여도 사람인 것 감지 필요
# mediapipe로 담 넘는 것 감지 제대로 안 됨.