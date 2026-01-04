import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np # Не забудь імпортувати numpy
import config

class FaceTracker:
    def __init__(self):
        # cvzone сам налаштовує MediaPipe "під капотом"
        self.detector = FaceMeshDetector(maxFaces=1)
        
        self.pid = config.PID_COEFFICIENTS
        self.pError = 0
        self.w = config.FRAME_WIDTH
        self.h = config.FRAME_HEIGHT
        self.fbRange = config.FACE_AREA_RANGE

    def find_face(self, img):
        # cvzone робить всю магію одним рядком
        img, faces = self.detector.findFaceMesh(img, draw=True)
        
        info = [[0, 0], 0]
        
        if faces:
            face = faces[0]
            # Точка 10 (лоб) і 152 (підборіддя) для висоти
            # Точка 234 (ліво) і 454 (право) для ширини
            left = face[234]
            right = face[454]
            top = face[10]
            bottom = face[152]

            # Центр
            cx, cy = face[1] # Точка 1 - кінчик носа (центр)
            
            # Площа (приблизно як ширина * висоту)
            w_face = right[0] - left[0]
            h_face = bottom[1] - top[1]
            area = w_face * h_face
            
            cv2.putText(img, f"Area: {area}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            info = [[cx, cy], area]

        return img, info

    def calculate_pid(self, info):
        area = info[1]
        x, y = info[0]
        fb = 0 
        yaw = 0
        ud = 0 # <-- НОВА ЗМІННА ВИСОТИ
        
        # --- 1. ПОВОРОТ (Yaw) ---
        error_yaw = x - self.w // 2
        yaw = self.pid[0] * error_yaw + self.pid[1] * (error_yaw - self.pError)
        
        # Обмежуємо (clamping)
        if yaw > 100: yaw = 100
        elif yaw < -100: yaw = -100
        else: yaw = int(yaw)
        
        if -15 < error_yaw < 15: 
            yaw = 0
            error_yaw = 0 # Скидаємо помилку в мертвій зоні
            
        self.pError = error_yaw

        # --- 2. ДИСТАНЦІЯ (Forward/Back) ---
        if area != 0: 
            if area < self.fbRange[0]: fb = 25
            elif area > self.fbRange[1]: fb = -25

        # --- 3. ВИСОТА (Up/Down) ---
        # Центр екрану по висоті
        error_ud = (self.h // 2) - y
        
        # Простий пропорційний регулятор (Kp = 0.5)
        ud = int(error_ud * 0.5)
        
        # Обмеження швидкості
        if ud > 50: ud = 50
        elif ud < -50: ud = -50
        
        # Мертва зона для висоти
        if -15 < error_ud < 15: ud = 0
        
        if x == 0:
            yaw = 0
            fb = 0
            ud = 0
            
        return yaw, fb, ud # <-- ПОВЕРТАЄМО 3 ЗНАЧЕННЯ
