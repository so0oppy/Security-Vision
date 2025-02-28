# Security-Vision
### [실시간 보안 감시 및 침입 탐지 시스템] OpenCV/AI 프로젝트

시연영상: https://youtu.be/FemXG-6iyXY

<img width="389" alt="project_result" src="https://github.com/user-attachments/assets/a986d8be-bf94-483d-91db-1f5ed0666071" />

#### 🪐프로그래밍 언어
Python
- OpenCV, NumPy, SciPy 등의 라이브러리를 활용하여 영상 처리 및 침입 감지 구현

#### 🕶️영상 처리 및 객체 탐지
OpenCV (cv2)
- cv2.VideoCapture(): 실시간 영상 스트리밍
- cv2.putText(): 화면에 침입 감지 텍스트 출력
- cv2.boundingRect(): 객체 탐지 후 Bounding Box 설정
  
#### 👕의상 특징 벡터 추출 및 비교
특징 추출 함수 (get_clothing_feature())
- 사람이 입은 옷의 특징 벡터를 추출
벡터 유사도 비교 (Cosine Similarity)
- SciPy의 cosine() 함수를 사용하여 의상 특징 벡터 간 유사도 비교
- 일정 거리(distance < 0.5) 이하이면 같은 사람으로 판단

#### 👁️‍🗨️침입 감지 로직
- 특정 영역 (monitor_x1, monitor_x2, monitor_y)을 설정하여 감시 구역 내에서 사람의 침입 감지
- 조건을 만족하면 intrusion_detected = True 설정 후 경고 메시지 출력
- 만약, 어슬렁거리던 사람이 (loitering_detected = True) 침입했을 경우 (intrusion_detected = True)
  <br>⚠️"Warning!" 메시지 출력 및 🚨경고음 발생
