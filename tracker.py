import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import config

class FaceTracker:
    def __init__(self):
        # ЗМІНА: Дозволяємо бачити до 5 облич одночасно
        self.detector = FaceMeshDetector(maxFaces=5)
        
        self.pid = config.PID_COEFFICIENTS
        self.pError = 0
        self.w = config.FRAME_WIDTH
        self.h = config.FRAME_HEIGHT
        self.fbRange = config.FACE_AREA_RANGE

    def find_faces(self, img, draw_mesh=True):
        """
        Знаходить ВСІ обличчя в кадрі.
        Повертає зображення та список знайдених облич.
        """
        # cvzone малює сітку автоматично, якщо draw=True.
        # Але ми хочемо керувати кольорами самі (червоний/зелений), 
        # тому ставимо draw=False тут, і малюємо вручну в GUI або нижче.
        img, faces = self.detector.findFaceMesh(img, draw=False) 
        
        processed_faces = []

        if faces:
            for face in faces:
                # Витягуємо дані для кожного обличчя
                left = face[234]
                right = face[454]
                top = face[10]
                bottom = face[152]
                cx, cy = face[1] # Центр
                
                w_face = right[0] - left[0]
                h_face = bottom[1] - top[1]
                area = w_face * h_face
                
                # Формуємо зручний словник для GUI
                face_data = {
                    "id": face, # Сирі дані від cvzone
                    "center": (cx, cy),
                    "bbox": (left[0], top[1], right[0], bottom[1]), # x1, y1, x2, y2
                    "area": area
                }
                processed_faces.append(face_data)

        return img, processed_faces

    def draw_face(self, img, face_data, is_locked=False, draw_mesh=True):
        """Малює рамку або сітку навколо конкретного обличчя"""
        cx, cy = face_data["center"]
        x1, y1, x2, y2 = face_data["bbox"]
        area = face_data["area"]
        
        # Колір: ЗЕЛЕНИЙ (якщо просто перехожий), ЧЕРВОНИЙ (якщо ЦІЛЬ)
        color = (0, 0, 255) if is_locked else (0, 255, 0)
        thickness = 3 if is_locked else 1

        if draw_mesh:
            # Малюємо всі точки (mesh)
            # face_data["id"] містить всі 468 точок
            for point in face_data["id"]:
                cv2.circle(img, point, 1, color, cv2.FILLED)
        else:
            # Малюємо рамку (Box)
            pad = 10
            cv2.rectangle(img, (x1-pad, y1-pad), (x2+pad, y2+pad), color, thickness)
        
        # Текст
        label = "LOCKED" if is_locked else f"Area: {area}"
        cv2.putText(img, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if is_locked:
            # Приціл у центрі для обраного
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (cx-10, cy), (cx+10, cy), (0, 0, 255), 2)
            cv2.line(img, (cx, cy-10), (cx, cy+10), (0, 0, 255), 2)

        return img

    def calculate_pid(self, face_data):
        """PID розрахунок для одного конкретного обличчя"""
        if face_data is None:
            return 0, 0, 0

        area = face_data["area"]
        cx, cy = face_data["center"]
        
        fb = 0 
        yaw = 0
        ud = 0
        
        # --- 1. ПОВОРОТ (YAW) ---
        error_yaw = cx - self.w // 2
        yaw = self.pid[0] * error_yaw + self.pid[1] * (error_yaw - self.pError)
        yaw = int(np.clip(yaw, -100, 100))
        if -15 < error_yaw < 15: yaw = 0; error_yaw = 0
        self.pError = error_yaw

        # --- 2. ДИСТАНЦІЯ (FB) ---
        if area != 0: 
            if area < self.fbRange[0]: fb = 25
            elif area > self.fbRange[1]: fb = -25

        # --- 3. ВИСОТА (UD) ---
        error_ud = (self.h // 2) - cy
        ud = int(error_ud * 0.5)
        ud = int(np.clip(ud, -50, 50))
        if -15 < error_ud < 15: ud = 0
            
        return yaw, fb, ud
