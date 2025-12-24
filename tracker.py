import cv2
import numpy as np
import config

class FaceTracker:
    def __init__(self):
        # Завантажуємо каскад Хаара
        # Використовуємо шлях з конфігу або системний шлях OpenCV
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + config.CASCADE_PATH)
        if self.faceCascade.empty():
             # Якщо файл не знайдено у папці, пробуємо стандартний з бібліотеки
             self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        self.pid = config.PID_COEFFICIENTS
        self.pError = 0
        self.w = config.FRAME_WIDTH
        self.h = config.FRAME_HEIGHT
        self.fbRange = config.FACE_AREA_RANGE

    def find_face(self, img):
        """Знаходить обличчя на зображенні"""
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(imgGray, 1.1, 4)
        
        myFaceListC = []
        myFaceListArea = []

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = x + w // 2
            cy = y + h // 2
            area = w * h
            myFaceListC.append([cx, cy])
            myFaceListArea.append(area)
            cv2.putText(img, f"Area: {area}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if len(myFaceListArea) != 0:
            i = myFaceListArea.index(max(myFaceListArea))
            return img, [myFaceListC[i], myFaceListArea[i]]
        else:
            return img, [[0, 0], 0]

    def calculate_pid(self, info):
        """Розраховує керуючі команди на основі PID"""
        area = info[1]
        x, y = info[0]
        fb = 0 
        yaw = 0 
        
        # Розрахунок помилки для повороту
        error = x - self.w // 2
        
        # PID формула
        yaw = self.pid[0] * error + self.pid[1] * (error - self.pError)
        yaw = int(np.clip(yaw, -100, 100))
        
        # Мертва зона (щоб не сіпався при дрібних відхиленнях)
        if -15 < error < 15: 
            yaw = 0
            error = 0
            
        self.pError = error

        # Розрахунок дистанції (Forward/Back)
        if area != 0: 
            if area < self.fbRange[0]:
                fb = 25
            elif area > self.fbRange[1]:
                fb = -25
        
        # Якщо обличчя немає
        if x == 0:
            yaw = 0
            fb = 0
            
        return yaw, fb
