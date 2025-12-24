import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
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
        # Ця частина не змінюється
        area = info[1]
        x, y = info[0]
        fb = 0 
        yaw = 0 
        
        error = x - self.w // 2
        yaw = self.pid[0] * error + self.pid[1] * (error - self.pError)
        
        # Обмежуємо (clamping)
        if yaw > 100: yaw = 100
        elif yaw < -100: yaw = -100
        else: yaw = int(yaw)
        
        if -15 < error < 15: 
            yaw = 0
            error = 0
            
        self.pError = error

        if area != 0: 
            if area < self.fbRange[0]: fb = 25
            elif area > self.fbRange[1]: fb = -25
        
        if x == 0: yaw = 0; fb = 0
            
        return yaw, fb
