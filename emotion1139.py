import cv2
from deepface import DeepFace
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import base64
from flask_cors import CORS  # CORS 추가

# Flask 앱 생성
app = Flask(__name__)
CORS(app)  # Flask 앱에서 CORS 활성화

# 메인 페이지 라우트
@app.route('/')
def index():
    return send_from_directory('', 'EmotionWeb.html')  # index.html이 현재 디렉토리에 있다고 가정

# 얼굴 감지 및 감정 분석 함수
def analyze_emotion(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    emotions = []

    for (x, y, w, h) in faces:
        face_roi = image[y:y + h, x:x + w]
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if result and 'dominant_emotion' in result[0]:  # 첫 번째 얼굴의 결과 확인
                emotions.append(result[0]['dominant_emotion'])
            else:
                emotions.append('감정 감지 실패')  # 감정 감지 실패 시 메시지 추가
        except Exception as e:
            print(f"감정 분석 중 오류 발생: {e}")
            emotions.append('분석 오류')

    return emotions

# 클라이언트로부터 이미지를 받아 분석하고 결과를 반환하는 엔드포인트
@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json['image']
    image_data = base64.b64decode(data)
    np_image = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': '이미지 디코딩 오류'}), 400

    emotions = analyze_emotion(image)
    return jsonify({'emotions': emotions})

if __name__ == '__main__':
    app.run(debug=True)
