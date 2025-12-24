import tkinter as tk
from tkinter import messagebox  # <-- ДОДАНО: Для спливаючих вікон
from PIL import Image, ImageTk
import cv2
import time
import threading
from djitellopy import Tello

import config
from videograbber import VideoGrabber
from tracker import FaceTracker

class MainWindow:
    def __init__(self, root):
        self.window = root
        self.window.title("DJI Tello: Face Tracking + Manual Control")
        self.window.geometry("900x650")

        self.tracker = FaceTracker()
        self.grabber = None
        self.tello = Tello(host=config.DRONE_IP)
        
        self.is_rotation_active = False
        self.is_distance_active = False
        self.is_flying = False

        self.pressed_keys = set()

        self.setup_ui()
        self.setup_input()
        self.connect_drone()

        self.update_loop()

    def setup_ui(self):
        self.video_label = tk.Label(self.window, bg="black")
        self.video_label.pack(pady=10)
        self.video_label.focus_set()

        self.info_label = tk.Label(self.window, text="Керування: WASD, Shift/Ctrl, Q/E", font=("Arial", 12))
        self.info_label.pack(pady=5)

        controls_frame = tk.Frame(self.window)
        controls_frame.pack(side=tk.BOTTOM, pady=20)

        btn_opts = {"width": 15, "padx": 10, "pady": 5}

        # --- КНОПКИ ---
        self.btn_takeoff = tk.Button(controls_frame, text="ЗЛІТ", bg="green", fg="white", command=self.takeoff, **btn_opts)
        self.btn_takeoff.pack(side=tk.LEFT, padx=5)

        self.btn_land = tk.Button(controls_frame, text="ПОСАДКА", bg="red", fg="white", command=self.land, **btn_opts)
        self.btn_land.pack(side=tk.LEFT, padx=5)

        self.btn_yaw = tk.Button(controls_frame, text="Увімк. Повороти", bg="gray", command=self.toggle_yaw, **btn_opts)
        self.btn_yaw.pack(side=tk.LEFT, padx=5)

        self.btn_dist = tk.Button(controls_frame, text="Увімк. Наближення", bg="gray", command=self.toggle_dist, **btn_opts)
        self.btn_dist.pack(side=tk.LEFT, padx=5)

        # Нова кнопка ІНФО
        self.btn_info = tk.Button(controls_frame, text="Інфо", bg="#007acc", fg="white", command=self.show_help, **btn_opts)
        self.btn_info.pack(side=tk.LEFT, padx=5)

    def show_help(self):
        """Функція для відображення інструкції"""
        help_text = """
        🎮 ІНСТРУКЦІЯ З КЕРУВАННЯ:

        • W / S  — Рух Вперед / Назад
        • A / D  — Рух Вліво / Вправо
        • Q / E  — Поворот навколо осі
        • Shift  — Піднятися Вгору
        • Ctrl   — Опуститися Вниз

        ⚠️ ВАЖЛИВО:
        Натискання клавіш має пріоритет над автопілотом.
        Якщо ви натискаєте кнопки, дрон слухає ВАС, а не камеру.
        """
        messagebox.showinfo("Як керувати дроном", help_text)
        # Повертаємо фокус на відео, щоб клавіатура працювала далі
        self.video_label.focus_set()

    def setup_input(self):
        self.window.bind("<KeyPress>", self.key_down)
        self.window.bind("<KeyRelease>", self.key_up)

    def key_down(self, event):
        key = event.keysym.lower()
        self.pressed_keys.add(key)

    def key_up(self, event):
        key = event.keysym.lower()
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)

    def get_manual_command(self):
        lr, fb, ud, yv = 0, 0, 0, 0
        speed = config.MANUAL_SPEED

        if 'w' in self.pressed_keys: fb = speed
        if 's' in self.pressed_keys: fb = -speed
        if 'a' in self.pressed_keys: lr = -speed
        if 'd' in self.pressed_keys: lr = speed
        if 'q' in self.pressed_keys: yv = -speed
        if 'e' in self.pressed_keys: yv = speed
        if 'shift_l' in self.pressed_keys or 'shift_r' in self.pressed_keys: ud = speed
        if 'control_l' in self.pressed_keys or 'control_r' in self.pressed_keys: ud = -speed

        return lr, fb, ud, yv

    def connect_drone(self):
        def _connect():
            try:
                self.tello.connect()
                self.tello.streamoff()
                self.tello.streamon()
                bat = self.tello.get_battery()
                self.info_label.config(text=f"Батарея: {bat}% | Керування: WASD")
                
                print("Запуск відео...")
                time.sleep(1)
                self.grabber = VideoGrabber(config.UDP_VIDEO_ADDRESS).start()
                
            except Exception as e:
                self.info_label.config(text=f"Помилка: {e}", fg="red")
        
        threading.Thread(target=_connect, daemon=True).start()

    def update_loop(self):
        if self.grabber is not None:
            frame = self.grabber.read()
            
            if frame is not None:
                frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                frame_processed, info = self.tracker.find_face(frame_resized)
                
                pid_yaw, pid_fb = self.tracker.calculate_pid(info)
                man_lr, man_fb, man_ud, man_yaw = self.get_manual_command()

                final_lr = man_lr
                final_ud = man_ud
                final_fb = man_fb
                final_yaw = man_yaw

                if final_fb == 0 and self.is_distance_active:
                    final_fb = pid_fb
                
                if final_yaw == 0 and self.is_rotation_active:
                    final_yaw = pid_yaw

                if self.is_flying:
                    self.tello.send_rc_control(final_lr, final_fb, final_ud, final_yaw)
                
                img_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil.resize((720, 540), Image.Resampling.BOX))
                
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

        self.window.after(10, self.update_loop)

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
        self.btn_yaw.config(bg=color_yaw)

        color_dist = "orange" if self.is_distance_active else "gray"
        self.btn_dist.config(bg=color_dist)

    def close(self):
        if self.grabber:
            self.grabber.stop()
        self.window.destroy()
