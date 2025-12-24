import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
import threading
from djitellopy import Tello

# Імпорт наших модулів
import config
from drone_utils import VideoGrabber
from tracker import FaceTracker

class MainWindow:
    def __init__(self, root):
        self.window = root
        self.window.title("DJI Tello Face Tracking System")
        self.window.geometry("900x650")

        # --- Ініціалізація компонентів ---
        self.tracker = FaceTracker()
        self.grabber = None
        self.tello = Tello(host=config.DRONE_IP)
        
        # Прапорці стану
        self.is_rotation_active = False
        self.is_distance_active = False
        self.is_flying = False

        self.setup_ui()
        self.connect_drone()

        # Запуск циклу оновлення
        self.update_loop()

    def setup_ui(self):
        # Віджет відео
        self.video_label = tk.Label(self.window, bg="black")
        self.video_label.pack(pady=10)

        # Інфо панель
        self.info_label = tk.Label(self.window, text="Ініціалізація...", font=("Arial", 12))
        self.info_label.pack(pady=5)

        # Кнопки
        controls_frame = tk.Frame(self.window)
        controls_frame.pack(side=tk.BOTTOM, pady=20)

        btn_opts = {"width": 15, "padx": 10, "pady": 5}

        self.btn_takeoff = tk.Button(controls_frame, text="ЗЛІТ", bg="green", fg="white", command=self.takeoff, **btn_opts)
        self.btn_takeoff.pack(side=tk.LEFT, padx=5)

        self.btn_land = tk.Button(controls_frame, text="ПОСАДКА", bg="red", fg="white", command=self.land, **btn_opts)
        self.btn_land.pack(side=tk.LEFT, padx=5)

        self.btn_yaw = tk.Button(controls_frame, text="Увімк. Повороти", bg="gray", command=self.toggle_yaw, **btn_opts)
        self.btn_yaw.pack(side=tk.LEFT, padx=5)

        self.btn_dist = tk.Button(controls_frame, text="Увімк. Наближення", bg="gray", command=self.toggle_dist, **btn_opts)
        self.btn_dist.pack(side=tk.LEFT, padx=5)

    def connect_drone(self):
        """Підключення у фоновому потоці, щоб не вішати інтерфейс"""
        def _connect():
            try:
                self.tello.connect()
                self.tello.streamoff()
                self.tello.streamon()
                bat = self.tello.get_battery()
                self.info_label.config(text=f"Підключено. Батарея: {bat}%")
                
                # Запуск граббера після успішного підключення
                print("Запуск відео...")
                time.sleep(1)
                self.grabber = VideoGrabber(config.UDP_VIDEO_ADDRESS).start()
                
            except Exception as e:
                self.info_label.config(text=f"Помилка: {e}", fg="red")
        
        threading.Thread(target=_connect, daemon=True).start()

    def update_loop(self):
        """Головний цикл обробки кадрів"""
        if self.grabber is not None:
            frame = self.grabber.read()
            
            if frame is not None:
                # 1. Зміна розміру
                frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                
                # 2. Пошук обличчя (через модуль tracker)
                frame_processed, info = self.tracker.find_face(frame_resized)
                
                # 3. Розрахунок PID (через модуль tracker)
                yaw, fb = self.tracker.calculate_pid(info)
                
                # 4. Відправка команд на дрон
                real_yaw = yaw if self.is_rotation_active else 0
                real_fb = fb if self.is_distance_active else 0
                
                if self.is_flying:
                    self.tello.send_rc_control(0, real_fb, 0, real_yaw)
                
                # 5. Відображення
                img_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil.resize((720, 540), Image.Resampling.BOX))
                
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

        # Плануємо наступний кадр через 10 мс
        self.window.after(10, self.update_loop)

    # --- Обробники кнопок ---
    def takeoff(self):
        try:
            self.tello.takeoff()
            self.tello.send_rc_control(0, 0, 25, 0)
            self.is_flying = True
        except: pass

    def land(self):
        self.is_rotation_active = False
        self.is_distance_active = False
        self.update_btns()
        try:
            self.tello.land()
            self.is_flying = False
        except: pass

    def toggle_yaw(self):
        self.is_rotation_active = not self.is_rotation_active
        self.update_btns()

    def toggle_dist(self):
        self.is_distance_active = not self.is_distance_active
        self.update_btns()

    def update_btns(self):
        color_yaw = "orange" if self.is_rotation_active else "gray"
        text_yaw = "ВИМК. Повороти" if self.is_rotation_active else "Увімк. Повороти"
        self.btn_yaw.config(bg=color_yaw, text=text_yaw)

        color_dist = "orange" if self.is_distance_active else "gray"
        text_dist = "ВИМК. Наближення" if self.is_distance_active else "Увімк. Наближення"
        self.btn_dist.config(bg=color_dist, text=text_dist)

    def close(self):
        if self.grabber:
            self.grabber.stop()
        self.window.destroy()
