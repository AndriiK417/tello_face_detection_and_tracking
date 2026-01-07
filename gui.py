import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
import math
from djitellopy import Tello

import config
from videograbber import VideoGrabber
from tracker import FaceTracker

class MainWindow:
    def __init__(self, root):
        self.window = root
        self.window.title("DJI Tello: Smart Tracking")
        self.window.geometry("900x650")

        self.tracker = FaceTracker()
        self.grabber = None
        self.tello = Tello(host=config.DRONE_IP)
        
        # --- СТАН СИСТЕМИ ---
        self.is_distance_active = False
        self.is_flying = False
        self.show_mesh = True 
        
        # --- ЛОГІКА ЗАХОПЛЕННЯ ЦІЛІ ---
        self.locked_face_center = None 
        self.locked_face_area = 0      # <-- НОВЕ: Запам'ятовуємо розмір обличчя
        self.is_tracking_locked = False
        
        # Лічильник кадрів втрати цілі
        self.lost_target_frames = 0    # <-- НОВЕ: Скільки кадрів ми не бачимо ціль

        self.pressed_keys = set()

        self.setup_ui()
        self.setup_input()
        self.connect_drone()
        self.update_loop()

    def setup_ui(self):
        self.video_label = tk.Label(self.window, bg="black")
        self.video_label.pack(pady=10)
        self.video_label.focus_set()
        
        self.video_label.bind("<Button-1>", self.on_mouse_click)
        self.video_label.bind("<Button-3>", self.cancel_tracking)

        self.info_label = tk.Label(self.window, text="ЛКМ - Захопити ціль | ПКМ - Скинути | Space - Посадка", font=("Arial", 11))
        self.info_label.pack(pady=5)

        controls_frame = tk.Frame(self.window)
        controls_frame.pack(side=tk.BOTTOM, pady=20)
        btn_opts = {"width": 15, "padx": 10, "pady": 5}

        self.btn_takeoff = tk.Button(controls_frame, text="ЗЛІТ", bg="green", fg="white", command=self.takeoff, **btn_opts)
        self.btn_takeoff.pack(side=tk.LEFT, padx=5)

        self.btn_land = tk.Button(controls_frame, text="ПОСАДКА", bg="red", fg="white", command=self.land, **btn_opts)
        self.btn_land.pack(side=tk.LEFT, padx=5)

        self.btn_dist = tk.Button(controls_frame, text="Увімк. Наближення", bg="gray", command=self.toggle_dist, **btn_opts)
        self.btn_dist.pack(side=tk.LEFT, padx=5)

        self.btn_view = tk.Button(controls_frame, text="Вигляд: Сітка", bg="purple", fg="white", command=self.toggle_view, **btn_opts)
        self.btn_view.pack(side=tk.LEFT, padx=5)

        self.btn_info = tk.Button(controls_frame, text="Інфо", bg="#007acc", fg="white", command=self.show_help, **btn_opts)
        self.btn_info.pack(side=tk.LEFT, padx=5)

    def on_mouse_click(self, event):
        """Обробка кліку по відео"""
        gui_x, gui_y = event.x, event.y
        scale_x = config.FRAME_WIDTH / 720
        scale_y = config.FRAME_HEIGHT / 540
        click_x = int(gui_x * scale_x)
        click_y = int(gui_y * scale_y)
        
        print(f"Клік по: {click_x}, {click_y}")
        
        # При кліку ми поки не знаємо площу, вона оновиться в update_loop
        self.locked_face_center = (click_x, click_y)
        self.locked_face_area = 0 
        self.is_tracking_locked = True
        self.lost_target_frames = 0
        self.info_label.config(text="ЦІЛЬ ЗАХОПЛЕНО! Шукаю обличчя...", fg="blue")

    def cancel_tracking(self, event=None):
        """Скидання цілі"""
        self.is_tracking_locked = False
        self.locked_face_center = None
        self.locked_face_area = 0
        self.info_label.config(text="Ціль скинуто. Режим очікування.", fg="black")

    def find_closest_face(self, faces, target_pos, target_area):
        """
        Розумний пошук цілі:
        1. Перевіряє дистанцію (щоб не стрибало далеко).
        2. Перевіряє площу (щоб не перемикалось на когось іншого розміру).
        """
        if not faces or target_pos is None:
            return None
        
        closest_face = None
        min_dist = float('inf')
        
        tx, ty = target_pos
        
        for face in faces:
            fx, fy = face["center"]
            f_area = face["area"]
            
            # 1. Перевірка відстані
            dist = math.hypot(fx - tx, fy - ty)
            
            # 2. Перевірка площі (якщо ми вже знаємо площу цілі)
            is_area_ok = True
            if target_area > 0:
                # Дозволяємо зміну розміру не більше ніж на 50%
                ratio = f_area / target_area
                if ratio < 0.5 or ratio > 1.5:
                    is_area_ok = False
            
            # Якщо обличчя близько І підходить за розміром
            if dist < min_dist and is_area_ok:
                min_dist = dist
                closest_face = face
        
        # Жорсткий поріг дистанції: 100 пікселів
        # Якщо найближче обличчя далі 100 пікселів від останньої точки - це не наша людина
        if min_dist > 100:
            return None
            
        return closest_face

    def update_loop(self):
        if self.grabber is not None:
            frame = self.grabber.read()
            
            if frame is not None:
                frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                frame_processed, faces = self.tracker.find_faces(frame_resized)
                
                target_face = None
                
                # --- ЛОГІКА ТРЕКІНГУ ---
                if self.is_tracking_locked and self.locked_face_center is not None:
                    # Шукаємо наше обличчя серед усіх знайдених
                    target_face = self.find_closest_face(faces, self.locked_face_center, self.locked_face_area)
                    
                    if target_face:
                        # ЦІЛЬ ЗНАЙДЕНО
                        self.locked_face_center = target_face["center"]
                        self.locked_face_area = target_face["area"] # Оновлюємо поточний розмір
                        self.lost_target_frames = 0 # Скидаємо лічильник втрати
                        
                        self.info_label.config(text=f"Стеження... Area: {target_face['area']}", fg="green")
                    else:
                        # ЦІЛЬ НЕ ЗНАЙДЕНО (в цьому кадрі)
                        self.lost_target_frames += 1
                        self.info_label.config(text=f"Втрата цілі! ({self.lost_target_frames}/50)", fg="orange")
                        
                        # Якщо цілі немає вже 50 кадрів (~2-3 секунди), скидаємо захоплення повністю
                        if self.lost_target_frames > 50:
                            self.cancel_tracking()
                            messagebox.showwarning("Увага", "Ціль втрачено надовго. Трекінг вимкнено.")

                # --- МАЛЮВАННЯ ТА PID ---
                pid_yaw, pid_fb, pid_ud = 0, 0, 0
                
                if faces:
                    for face in faces:
                        # Чи це те обличчя, яке ми ведемо?
                        is_locked = (face == target_face)
                        frame_processed = self.tracker.draw_face(frame_processed, face, is_locked, self.show_mesh)
                        
                        if is_locked:
                            pid_yaw, pid_fb, pid_ud = self.tracker.calculate_pid(face)

                # --- КЕРУВАННЯ ДРОНОМ ---
                man_lr, man_fb, man_ud, man_yaw = self.get_manual_command()
                
                final_lr = man_lr
                final_fb = man_fb
                final_ud = man_ud
                final_yaw = man_yaw

                if self.is_tracking_locked and target_face:
                    # Автопілот працює тільки якщо ми БАЧИМО ціль прямо зараз
                    if final_yaw == 0: final_yaw = pid_yaw
                    if final_ud == 0: final_ud = pid_ud
                    
                    if self.is_distance_active and final_fb == 0:
                        final_fb = pid_fb
                else:
                    # Якщо ціль втрачено (але ще не скинуто повністю), дрон має зависнути
                    # Він не повинен виконувати старі команди PID
                    pass

                if self.is_flying:
                    self.tello.send_rc_control(final_lr, final_fb, final_ud, final_yaw)
                
                # Відображення
                img_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil.resize((720, 540), Image.Resampling.BOX))
                
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

        self.window.after(10, self.update_loop)

    # --- РЕШТА МЕТОДІВ (Без змін) ---
    def connect_drone(self):
        def _connect():
            try:
                self.tello.connect()
                self.tello.streamoff()
                self.tello.streamon()
                bat = self.tello.get_battery()
                self.info_label.config(text=f"Батарея: {bat}% | Готовий")
                print("Запуск відео...")
                time.sleep(1)
                self.grabber = VideoGrabber(config.UDP_VIDEO_ADDRESS).start()
            except Exception as e:
                self.info_label.config(text=f"Помилка: {e}", fg="red")
        threading.Thread(target=_connect, daemon=True).start()

    def setup_input(self):
        self.window.bind("<KeyPress>", self.key_down)
        self.window.bind("<KeyRelease>", self.key_up)

    def key_down(self, event):
        key = event.keysym.lower()
        self.pressed_keys.add(key)
        if key == 'space': self.land()

    def key_up(self, event):
        key = event.keysym.lower()
        if key in self.pressed_keys: self.pressed_keys.remove(key)

    def get_manual_command(self):
        lr, fb, ud, yv = 0, 0, 0, 0
        speed = config.MANUAL_SPEED
        pk = self.pressed_keys
        if 'w' in pk: fb = speed
        if 's' in pk: fb = -speed
        if 'a' in pk: lr = -speed
        if 'd' in pk: lr = speed
        if 'q' in pk: yv = -speed
        if 'e' in pk: yv = speed
        if 'shift_l' in pk or 'shift_r' in pk: ud = speed
        if 'control_l' in pk or 'control_r' in pk: ud = -speed
        return lr, fb, ud, yv

    def takeoff(self):
        try:
            self.tello.takeoff()
            self.tello.send_rc_control(0, 0, 25, 0)
            self.is_flying = True
        except: pass

    def land(self):
        self.is_tracking_locked = False
        self.is_distance_active = False
        self.update_btns()
        try:
            self.tello.land()
            self.is_flying = False
        except: pass

    def toggle_dist(self):
        self.is_distance_active = not self.is_distance_active
        self.update_btns()

    def toggle_view(self):
        self.show_mesh = not self.show_mesh
        self.btn_view.config(text="Вигляд: Сітка" if self.show_mesh else "Вигляд: Рамка", bg="purple" if self.show_mesh else "#8e44ad")

    def update_btns(self):
        self.btn_dist.config(bg="orange" if self.is_distance_active else "gray", text="ВИМК. Наближення" if self.is_distance_active else "Увімк. Наближення")

    def show_help(self):
        messagebox.showinfo("Інфо", "ЛКМ - Вибрати ціль\nПКМ - Скинути ціль\nSpace - Посадка")
        self.video_label.focus_set()

    def close(self):
        if self.grabber: self.grabber.stop()
        self.window.destroy()
