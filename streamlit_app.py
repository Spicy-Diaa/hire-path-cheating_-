import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import mediapipe as mp
import os
from datetime import datetime

CAP_DIR = "cheating_captures"
os.makedirs(CAP_DIR, exist_ok=True)

st.title("🛡️ نظام مراقبة المقابلات - منع الغش")
st.write("اضغط على زر **Start** بالأسفل لتشغيل الكاميرا وبدء جلسة المراقبة.")

mp_face_mesh = mp.solutions.face_mesh

class CheatingDetector(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_tip = face_landmarks.landmark[1]
                x_pos = nose_tip.x
                
                if x_pos < 0.35 or x_pos > 0.65:
                    cv2.putText(img, "WARNING: Looking away!", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_name = os.path.join(CAP_DIR, f"cheat_{timestamp}.jpg")
                    cv2.imwrite(img_name, img)
                else:
                    cv2.putText(img, "Status: Focused", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

webrtc_streamer(key="monitoring", video_transformer_factory=CheatingDetector)
