import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
import numpy as np
import math
import config

class FaceTracker:
    def __init__(self):
        # maxFaces=5, щоб бачити всіх кандидатів
        self.detector = FaceMeshDetector(maxFaces=5)
        
        self.pid = config.PID_COEFFICIENTS
        self.pError = 0
        self.w = config.FRAME_WIDTH
        self.h = config.FRAME_HEIGHT
        self.fbRange = config.FACE_AREA_RANGE

    def find_faces(self, img):
        # Отримуємо обличчя
        img, faces = self.detector.findFaceMesh(img, draw=False) 
        processed_faces = []

        if faces:
            for face in faces:
                # Базові координати рамки
                left = face[234]
                right = face[454]
                top = face[10]
                bottom = face[152]
                cx, cy = face[1]
                
                w_face = right[0] - left[0]
                h_face = bottom[1] - top[1]
                area = w_face * h_face
                
                # --- РОЗРАХУНОК ВЕКТОРА БІОМЕТРІЇ ---
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
        """
        Рахує 5 ключових коефіцієнтів (Вектор ознак).
        ВИПРАВЛЕНО: Використовуємо лише точки 0-467 (безпечні).
        """
        # --- БЕЗПЕЧНІ ТОЧКИ (0-467) ---
        # Внутрішні кутики очей: 133 (ліве), 362 (праве)
        # Зовнішні кутики очей: 33 (ліве), 263 (праве)
        # Ніс (кінчик): 1
        # Рот (кутики): 61, 291
        # Підборіддя: 152
        # Лоб: 10
        # Вуха (ширина): 234, 454

        # 1. Розміри обличчя (База)
        face_width = math.hypot(face[234][0]-face[454][0], face[234][1]-face[454][1])
        face_height = math.hypot(face[10][0]-face[152][0], face[10][1]-face[152][1])
        
        if face_width == 0 or face_height == 0: return [0,0,0,0,0]

        # --- ГОРИЗОНТАЛЬНІ (ділимо на Width) ---
        
        # R1: Відстань між внутрішніми кутиками очей (Замість зіниць)
        # Це навіть стабільніше, бо зіниці рухаються, а кутики - ні
        dist_eyes_inner = math.hypot(face[133][0]-face[362][0], face[133][1]-face[362][1])
        r1 = dist_eyes_inner / face_width

        # R2: Ширина рота (кутики)
        dist_mouth = math.hypot(face[61][0]-face[291][0], face[61][1]-face[291][1])
        r2 = dist_mouth / face_width

        # --- ВЕРТИКАЛЬНІ (ділимо на Height) ---
        
        # R3: Очі -> Ніс
        # Беремо середню висоту між кутиками очей
        eye_y = (face[133][1] + face[362][1]) / 2
        nose_y = face[1][1]
        dist_eye_nose = abs(eye_y - nose_y)
        r3 = dist_eye_nose / face_height

        # R4: Ніс -> Рот
        mouth_y = (face[61][1] + face[291][1]) / 2
        dist_nose_mouth = abs(nose_y - mouth_y)
        r4 = dist_nose_mouth / face_height

        # R5: Рот -> Підборіддя (Масивність щелепи)
        chin_y = face[152][1]
        dist_mouth_chin = abs(mouth_y - chin_y)
        r5 = dist_mouth_chin / face_height

        # Повертаємо вектор з 5 чисел
        return [r1, r2, r3, r4, r5]

    def compare_signatures(self, sig1, sig2):
        """
        Векторне порівняння (Manhattan Distance).
        Повертає загальну оцінку схожості (Score).
        """
        if not sig1 or not sig2: return False, 999.0
        
        # Рахуємо різницю
        diff_1 = abs(sig1[0] - sig2[0]) * 2.0  # Очі
        diff_2 = abs(sig1[1] - sig2[1]) * 0.5  # Рот
        diff_3 = abs(sig1[2] - sig2[2]) * 1.5  # Очі-Ніс
        diff_4 = abs(sig1[3] - sig2[3]) * 1.0  # Ніс-Рот
        diff_5 = abs(sig1[4] - sig2[4]) * 1.0  # Щелепа
        
        total_score = diff_1 + diff_2 + diff_3 + diff_4 + diff_5
        
        # ПОРІГ СХОЖОСТІ
        # Якщо Score < limit, то це та сама людина
        limit = 0.1 
        
        is_match = (total_score < limit)
        
        return is_match, total_score

    def draw_face(self, img, face_data, is_locked=False, draw_mesh=True):
        cx, cy = face_data["center"]
        x1, y1, x2, y2 = face_data["bbox"]
        
        color = (0, 0, 255) if is_locked else (0, 255, 0)
        
        if draw_mesh:
            for point in face_data["id"]:
                cv2.circle(img, point, 1, color, cv2.FILLED)
        else:
            pad = 10
            cv2.rectangle(img, (x1-pad, y1-pad), (x2+pad, y2+pad), color, 2)
        
        label = "LOCKED" if is_locked else "Scanning..."
        cv2.putText(img, label, (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if is_locked:
            cv2.line(img, (cx-10, cy), (cx+10, cy), (0, 0, 255), 2)
            cv2.line(img, (cx, cy-10), (cx, cy+10), (0, 0, 255), 2)

        return img

    def calculate_pid(self, face_data):
        if face_data is None: return 0, 0, 0
        area = face_data["area"]
        cx, cy = face_data["center"]
        fb, yaw, ud = 0, 0, 0
        
        # --- YAW (Поворот) ---
        error_yaw = cx - self.w // 2
        
        # Формула PD-регулятора
        yaw = self.pid[0] * error_yaw + self.pid[1] * (error_yaw - self.pError)
        yaw = int(np.clip(yaw, -100, 100))
        
        # --- ВАЖЛИВО: Збільшуємо мертву зону ---
        # Було 15, для HD якості ставимо 40-50
        if -45 < error_yaw < 45: 
            yaw = 0
            error_yaw = 0
            
        self.pError = error_yaw

        # --- Distance (Вперед/Назад) ---
        if area != 0: 
            if area < self.fbRange[0]: fb = 25
            elif area > self.fbRange[1]: fb = -25

        # --- UP/DOWN (Висота) ---
        error_ud = (self.h // 2) - cy
        ud = int(error_ud * 0.5) # Тут коефіцієнт 0.5 теж може бути завеликим для HD
        
        # Можна зменшити коефіцієнт висоти до 0.2
        ud = int(error_ud * 0.25) 
        
        ud = int(np.clip(ud, -50, 50))
        
        # Мертва зона для висоти теж більша
        if -45 < error_ud < 45: 
            ud = 0
            
        return yaw, fb, ud
