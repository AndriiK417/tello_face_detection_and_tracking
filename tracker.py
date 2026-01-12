import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import math
import config

# --- НАЛАШТУВАННЯ БІОМЕТРІЇ ---
BIOMETRIC_THRESHOLD = 0.1
CONFIDENCE_FRAMES = 30      

class FaceTracker:
    def __init__(self):
        self.detector = FaceMeshDetector(maxFaces=5)
        
        self.pid = config.PID_COEFFICIENTS
        self.pError = 0
        self.w = config.FRAME_WIDTH
        self.h = config.FRAME_HEIGHT
        self.fbRange = config.FACE_AREA_RANGE
        
        # Лічильники
        self.consecutive_match_frames = 0    
        self.consecutive_mismatch_frames = 0 

    def reset_counters(self):
        self.consecutive_match_frames = 0
        self.consecutive_mismatch_frames = 0

    def find_faces(self, img):
        img, faces = self.detector.findFaceMesh(img, draw=False) 
        processed_faces = []

        if faces:
            for face in faces:
                left = face[234]
                right = face[454]
                top = face[10]
                bottom = face[152]
                cx, cy = face[1]
                
                w_face = right[0] - left[0]
                h_face = bottom[1] - top[1]
                area = w_face * h_face
                
                signature = self.calculate_extended_signature(face)

                face_data = {
                    "id": face, 
                    "center": (cx, cy),
                    "bbox": (left[0], top[1], right[0], bottom[1]),
                    "area": area,
                    "signature": signature
                }
                processed_faces.append(face_data)

        return img, processed_faces

    def calculate_extended_signature(self, face):
        face_width = math.hypot(face[234][0]-face[454][0], face[234][1]-face[454][1])
        face_height = math.hypot(face[10][0]-face[152][0], face[10][1]-face[152][1])
        
        if face_width == 0 or face_height == 0: return [0,0,0,0,0]

        # R1
        dist_eyes_inner = math.hypot(face[133][0]-face[362][0], face[133][1]-face[362][1])
        r1 = dist_eyes_inner / face_width
        # R2
        dist_mouth = math.hypot(face[61][0]-face[291][0], face[61][1]-face[291][1])
        r2 = dist_mouth / face_width
        # R3
        eye_y = (face[133][1] + face[362][1]) / 2
        nose_y = face[1][1]
        r3 = abs(eye_y - nose_y) / face_height
        # R4
        mouth_y = (face[61][1] + face[291][1]) / 2
        r4 = abs(nose_y - mouth_y) / face_height
        # R5
        chin_y = face[152][1]
        r5 = abs(mouth_y - chin_y) / face_height

        return [r1, r2, r3, r4, r5]

    def compare_signatures(self, sig1, sig2):
        if not sig1 or not sig2: return 999.0
        
        diff_1 = abs(sig1[0] - sig2[0]) * 2.0
        diff_2 = abs(sig1[1] - sig2[1]) * 0.5
        diff_3 = abs(sig1[2] - sig2[2]) * 1.5
        diff_4 = abs(sig1[3] - sig2[3]) * 1.0
        diff_5 = abs(sig1[4] - sig2[4]) * 1.0
        
        return diff_1 + diff_2 + diff_3 + diff_4 + diff_5

    def validate_match(self, current_sig, target_sig, is_already_locked):
        score = self.compare_signatures(current_sig, target_sig)
        
        if is_already_locked:
            if score > BIOMETRIC_THRESHOLD:
                self.consecutive_mismatch_frames += 1
                msg = f"Сумніваюся... ({self.consecutive_mismatch_frames}/{CONFIDENCE_FRAMES})"
                if self.consecutive_mismatch_frames > CONFIDENCE_FRAMES:
                    self.reset_counters()
                    return False, "Втрачено довіру (чуже обличчя)"
            else:
                self.consecutive_mismatch_frames = 0
                msg = "Стежу за ціллю."
            return True, msg 

        else:
            if score < BIOMETRIC_THRESHOLD:
                self.consecutive_match_frames += 1
                msg = f"Перевірка... ({self.consecutive_match_frames}/{CONFIDENCE_FRAMES})"
                if self.consecutive_match_frames > CONFIDENCE_FRAMES:
                    self.reset_counters()
                    return True, "Ціль підтверджено!"
            else:
                if self.consecutive_match_frames > 0:
                    self.consecutive_match_frames -= 1
                msg = "Шукаю..."
            return False, msg

    def draw_face(self, img, face_data, is_locked=False, draw_mesh=True, score=None):
        cx, cy = face_data["center"]
        x1, y1, x2, y2 = face_data["bbox"]
        
        color = (0, 0, 255) if is_locked else (0, 255, 0)
        
        if draw_mesh:
            for point in face_data["id"]:
                cv2.circle(img, point, 1, color, cv2.FILLED)
        else:
            pad = 10
            cv2.rectangle(img, (x1-pad, y1-pad), (x2+pad, y2+pad), color, 2)
        
        # --- МАЛЮВАННЯ ТЕКСТУ ---
        
        # 1. Статус (LOCKED / Scanning) - Над самою рамкою
        label = "LOCKED" if is_locked else ""
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 2. SCORE - Ще вище над рамкою (Жовтим)
        if score is not None:
            score_text = f"Diff: {score:.3f}"
            # Колір Score: Зелений якщо < 0.15, Червоний якщо більше
            score_color = (0, 255, 0) if score < BIOMETRIC_THRESHOLD else (0, 0, 255)
            cv2.putText(img, score_text, (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
        
        if is_locked:
            cv2.line(img, (cx-10, cy), (cx+10, cy), (0, 0, 255), 2)
            cv2.line(img, (cx, cy-10), (cx, cy+10), (0, 0, 255), 2)

        return img

    def calculate_pid(self, face_data):
        if face_data is None: return 0, 0, 0
        area = face_data["area"]
        cx, cy = face_data["center"]
        fb, yaw, ud = 0, 0, 0
        
        error_yaw = cx - self.w // 2
        yaw = self.pid[0] * error_yaw + self.pid[1] * (error_yaw - self.pError)
        yaw = int(np.clip(yaw, -100, 100))
        if -45 < error_yaw < 45: yaw = 0; error_yaw = 0
        self.pError = error_yaw

        if area != 0: 
            if area < self.fbRange[0]: fb = 25
            elif area > self.fbRange[1]: fb = -25

        error_ud = (self.h // 2) - cy
        ud = int(error_ud * 0.25)
        ud = int(np.clip(ud, -50, 50))
        if -45 < error_ud < 45: ud = 0
            
        return yaw, fb, ud
