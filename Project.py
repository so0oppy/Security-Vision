# 실시간 보안 감시 및 침입 탐지 시스템
# -----------------------------------

import cv2
import numpy as np

# YOLO 모델 로드
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg") # 가중치, 설정파일
layer_names = net.getLayerNames() # YOLO의 전체 레이어 이름 가져오기
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()] # net.getUnconnectedOutLayers(): 출력 레이어(감지 결과 나오는 레이어) 가져오기

# coco 클래스 로드 (사람 감지)
with open("coco.names", "r") as file: # YOLO가 학습된 80개의 객체 이름 저장
    classes = [line.strip() for line in file.readlines()]

# cap = cv2.VideoCapture(0) # 기본 카메라
cap = cv2.VideoCapture("") # 웹캠 대신 영상파일 사용

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 입력 전처리
    height, width, channels = frame.shape # 프레임의 높이, 너비, 채널(RGB) 값 가져오기
    # cv2.dnn.blobFromImage: YOLO가 사용 가능하도록 프레임 정규화(0~1사이 값으로 변환)
    # (416, 416): YOLO의 입력 크기(고정)
    blob = cv2.dnn.blobFromImages(frame, 0.00392, (416, 416), swapRB=True, crop=False) 
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

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidence[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()