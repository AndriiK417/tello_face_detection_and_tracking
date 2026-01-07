import tkinter as tk
from tkinter import messagebox
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
        
        # Прапорці автопілота
        self.is_face_tracking_active = False # Yaw + Up/Down
        self.is_distance_active = False      # Forward/Back
        self.is_flying = False
        
        # Прапорець візуалізації (True = Сітка, False = Рамка)
        self.show_mesh = True 

        # Для клавіатури
        self.pressed_keys = set()

        self.setup_ui()
        self.setup_input()
        self.connect_drone()

        self.update_loop()

    def setup_ui(self):
        # Віджет відео
        self.video_label = tk.Label(self.window, bg="black")
        self.video_label.pack(pady=10)
        self.video_label.focus_set()

        # Інфо панель (Додано про Пробіл)
        self.info_label = tk.Label(self.window, text="Керування: WASD, Shift/Ctrl, Space - Посадка", font=("Arial", 12))
        self.info_label.pack(pady=5)

        # Панель кнопок
        controls_frame = tk.Frame(self.window)
        controls_frame.pack(side=tk.BOTTOM, pady=20)

        btn_opts = {"width": 15, "padx": 10, "pady": 5}

        # 1. Зліт/Посадка
        self.btn_takeoff = tk.Button(controls_frame, text="ЗЛІТ", bg="green", fg="white", command=self.takeoff, **btn_opts)
        self.btn_takeoff.pack(side=tk.LEFT, padx=5)

        self.btn_land = tk.Button(controls_frame, text="ПОСАДКА", bg="red", fg="white", command=self.land, **btn_opts)
        self.btn_land.pack(side=tk.LEFT, padx=5)

        # 2. Автопілот
        self.btn_track = tk.Button(controls_frame, text="Відстеження обличчя", bg="gray", command=self.toggle_tracking, **btn_opts)
        self.btn_track.pack(side=tk.LEFT, padx=5)

        self.btn_dist = tk.Button(controls_frame, text="Увімк. Наближення", bg="gray", command=self.toggle_dist, **btn_opts)
        self.btn_dist.pack(side=tk.LEFT, padx=5)

        # 3. Додаткові
        self.btn_view = tk.Button(controls_frame, text="Вигляд: Сітка", bg="purple", fg="white", command=self.toggle_view, **btn_opts)
        self.btn_view.pack(side=tk.LEFT, padx=5)

        self.btn_info = tk.Button(controls_frame, text="Інфо", bg="#007acc", fg="white", command=self.show_help, **btn_opts)
        self.btn_info.pack(side=tk.LEFT, padx=5)

    def toggle_view(self):
        """Перемикає вигляд між Сіткою та Рамкою"""
        self.show_mesh = not self.show_mesh
        if self.show_mesh:
            self.btn_view.config(text="Вигляд: Сітка", bg="purple")
        else:
            self.btn_view.config(text="Вигляд: Рамка", bg="#8e44ad")

    def update_loop(self):
        if self.grabber is not None:
            frame = self.grabber.read()
            
            if frame is not None:
                frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                
                # --- ПОШУК ОБЛИЧЧЯ ---
                frame_processed, info = self.tracker.find_face(frame_resized, draw_mesh=self.show_mesh)
                
                # PID розрахунки
                pid_yaw, pid_fb, pid_ud = self.tracker.calculate_pid(info)
                
                # Ручне керування
                man_lr, man_fb, man_ud, man_yaw = self.get_manual_command()

                final_lr = man_lr
                final_ud = man_ud
                final_fb = man_fb
                final_yaw = man_yaw

                # Логіка автопілота (якщо немає ручних команд)
                if self.is_face_tracking_active:
                    if final_yaw == 0: final_yaw = pid_yaw
                    if final_ud == 0: final_ud = pid_ud

                if self.is_distance_active and final_fb == 0:
                    final_fb = pid_fb

                # Відправка команд дрону
                if self.is_flying:
                    self.tello.send_rc_control(final_lr, final_fb, final_ud, final_yaw)
                
                # Відображення в GUI
                img_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil.resize((720, 540), Image.Resampling.BOX))
                
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

        self.window.after(10, self.update_loop)

    def connect_drone(self):
        def _connect():
            try:
                self.tello.connect()
                self.tello.streamoff()
                self.tello.streamon()
                bat = self.tello.get_battery()
                self.info_label.config(text=f"Батарея: {bat}% | Керування: WASD, Space")
                print("Запуск відео...")
                time.sleep(1)
                self.grabber = VideoGrabber(config.UDP_VIDEO_ADDRESS).start()
            except Exception as e:
                self.info_label.config(text=f"Помилка: {e}", fg="red")
        threading.Thread(target=_connect, daemon=True).start()

    # --- Керування клавіатурою ---
    def setup_input(self):
        self.window.bind("<KeyPress>", self.key_down)
        self.window.bind("<KeyRelease>", self.key_up)

    def key_down(self, event):
        key = event.keysym.lower()
        self.pressed_keys.add(key)
        
        # --- НОВА ЛОГІКА: Екстрена посадка на ПРОБІЛ ---
        if key == 'space':
            print("Екстрена посадка!")
            self.land()

    def key_up(self, event):
        key = event.keysym.lower()
        if key in self.pressed_keys:
            self.pressed_keys.remove(key)

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

    # --- Кнопки інтерфейсу ---
    def takeoff(self):
        try:
            self.tello.takeoff()
            self.tello.send_rc_control(0, 0, 25, 0)
            self.is_flying = True
        except: pass

    def land(self):
        self.is_face_tracking_active = False
        self.is_distance_active = False
        self.update_btns()
        try:
            self.tello.land()
            self.is_flying = False
        except: pass

    def toggle_tracking(self):
        self.is_face_tracking_active = not self.is_face_tracking_active
        self.update_btns()

    def toggle_dist(self):
        self.is_distance_active = not self.is_distance_active
        self.update_btns()

    def update_btns(self):
        if self.is_face_tracking_active:
            self.btn_track.config(bg="orange", text="ВИМК. Відстеження")
        else:
            self.btn_track.config(bg="gray", text="Відстеження обличчя")

        if self.is_distance_active:
            self.btn_dist.config(bg="orange", text="ВИМК. Наближення")
        else:
            self.btn_dist.config(bg="gray", text="Увімк. Наближення")

    def show_help(self):
        help_text = """
        🎮 ІНСТРУКЦІЯ:
        • W / S  — Вперед / Назад
        • A / D  — Вліво / Вправо
        • Q / E  — Поворот
        • Shift / Ctrl — Висота
        • ПРОБІЛ (Space) — Посадка

        Пріоритет у кнопок. Якщо не тиснеш кнопки — працює автопілот.
        """
        messagebox.showinfo("Інфо", help_text)
        self.video_label.focus_set()

    def close(self):
        if self.grabber:
            self.grabber.stop()
        self.window.destroy()
