import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import config

class FaceTracker:
    def __init__(self):
        # Ініціалізація детектора (максимум 1 обличчя)
        self.detector = FaceMeshDetector(maxFaces=1)
        
        # Налаштування PID
        self.pid = config.PID_COEFFICIENTS
        self.pError = 0
        
        # Розміри кадру
        self.w = config.FRAME_WIDTH
        self.h = config.FRAME_HEIGHT
        
        # Діапазон площі (для дистанції)
        self.fbRange = config.FACE_AREA_RANGE

    def find_face(self, img, draw_mesh=True):
        """
        Знаходить обличчя.
        draw_mesh=True -> малює сітку (Mesh).
        draw_mesh=False -> малює рамку (Box).
        """
        # cvzone шукає обличчя. 
        # Якщо draw_mesh=True, він сам одразу малює сітку на img.
        img, faces = self.detector.findFaceMesh(img, draw=draw_mesh)
        
        info = [[0, 0], 0]
        
        if faces:
            # Беремо перше знайдене обличчя
            face = faces[0]
            
            # Ключові точки для визначення розмірів
            # 234 - ліве вухо, 454 - праве вухо
            # 10 - лоб, 152 - підборіддя
            left = face[234]
            right = face[454]
            top = face[10]
            bottom = face[152]

            # Центр обличчя (точка на носі #1)
            cx, cy = face[1]
            
            # Розрахунок площі
            w_face = right[0] - left[0]
            h_face = bottom[1] - top[1]
            area = w_face * h_face
            
            # --- ВІЗУАЛІЗАЦІЯ ---
            if draw_mesh:
                # Якщо сітка вже намальована cvzone, додаємо тільки текст площі
                cv2.putText(img, f"Area: {area}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Якщо сітка вимкнена, малюємо рамку вручну
                pad = 10 # Відступ рамки від обличчя
                cv2.rectangle(img, 
                              (left[0]-pad, top[1]-pad), 
                              (right[0]+pad, bottom[1]+pad), 
                              (0, 255, 0), 3)
                
                # Малюємо точку центру
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                # Пишемо площу біля рамки
                cv2.putText(img, f"Area: {area}", (left[0], top[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Оновлюємо info для PID регулятора
            info = [[cx, cy], area]

        return img, info

    def calculate_pid(self, info):
        """Розраховує команди для дрона (Yaw, Forward/Back, Up/Down)"""
        area = info[1]
        x, y = info[0]
        fb = 0 
        yaw = 0
        ud = 0 # Up/Down
        
        # --- 1. ПОВОРОТ (YAW) ---
        error_yaw = x - self.w // 2
        
        # PID формула
        yaw = self.pid[0] * error_yaw + self.pid[1] * (error_yaw - self.pError)
        
        # Обмеження значень -100...100
        yaw = int(np.clip(yaw, -100, 100))
        
        # Мертва зона (щоб не сіпався)
        if -15 < error_yaw < 15: 
            yaw = 0
            error_yaw = 0
            
        self.pError = error_yaw # Запам'ятовуємо помилку для наступного кадру

        # --- 2. ДИСТАНЦІЯ (FORWARD/BACK) ---
        if area != 0: 
            if area < self.fbRange[0]:
                fb = 25  # Летіти вперед
            elif area > self.fbRange[1]:
                fb = -25 # Летіти назад

        # --- 3. ВИСОТА (UP/DOWN) ---
        # Центр екрану по висоті
        error_ud = (self.h // 2) - y
        
        # Простий коефіцієнт 0.5 для висоти
        ud = int(error_ud * 0.5)
        
        # Обмеження швидкості висоти (-50...50)
        ud = int(np.clip(ud, -50, 50))
        
        # Мертва зона висоти
        if -15 < error_ud < 15: 
            ud = 0
        
        # Якщо обличчя не знайдено (x=0), зупиняємось
        if x == 0:
            yaw = 0
            fb = 0
            ud = 0
            
        return yaw, fb, ud
