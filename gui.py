import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
import math
import numpy as np 
from djitellopy import Tello

import config
from videograbber import VideoGrabber
from tracker import FaceTracker

class MainWindow:
    def __init__(self, root):
        self.window = root
        self.window.title("DJI Tello: Final Biometric System")
        self.window.geometry("900x750")

        self.tracker = FaceTracker()
        self.grabber = None
        self.tello = Tello(host=config.DRONE_IP)
        
        # Стан
        self.is_distance_active = False
        self.is_flying = False
        self.show_mesh = True 
        
        # Пам'ять
        self.locked_face_center = None 
        self.locked_face_area = 0      
        self.locked_signature = None   
        
        self.is_tracking_locked = False
        
        # Калібрування
        self.is_calibrating = False
        self.calibration_data = []
        self.calibration_frames_target = 20
        
        self.last_battery_check = 0
        self.pressed_keys = set()

        self.setup_ui()
        self.setup_input()
        self.connect_drone()
        self.update_loop()

    def setup_ui(self):
        self.battery_label = tk.Label(self.window, text="Заряд батареї: --%", font=("Arial", 16, "bold"), fg="#333")
        self.battery_label.pack(pady=10)

        self.video_label = tk.Label(self.window, bg="black")
        self.video_label.pack(pady=5)
        self.video_label.focus_set()
        
        self.video_label.bind("<Button-1>", self.on_mouse_click)
        self.video_label.bind("<Button-3>", self.cancel_tracking)

        self.status_label = tk.Label(self.window, text="Система готова. Чекаю наказу.", font=("Consolas", 14), fg="blue")
        self.status_label.pack(pady=10)

        self.info_label = tk.Label(self.window, text="ЛКМ - Захопити | ПКМ - Скинути | Space - Посадка", font=("Arial", 10), fg="gray")
        self.info_label.pack(pady=0)

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
        gui_x, gui_y = event.x, event.y
        scale_x = config.FRAME_WIDTH / 720
        scale_y = config.FRAME_HEIGHT / 540
        click_x = int(gui_x * scale_x)
        click_y = int(gui_y * scale_y)
        
        self.locked_face_center = (click_x, click_y)
        self.locked_face_area = 0 
        self.locked_signature = None 
        
        self.is_tracking_locked = True
        self.is_calibrating = True
        self.calibration_data = []
        self.tracker.reset_counters()
        
        self.status_label.config(text="Починаю вивчення обличчя... Не рухайся!", fg="blue")

    def cancel_tracking(self, event=None):
        self.is_tracking_locked = False
        self.is_calibrating = False
        self.locked_face_center = None
        self.locked_signature = None
        self.tracker.reset_counters()
        self.status_label.config(text="Стеження зупинено. Чекаю нову ціль.", fg="black")

    def track_by_position(self, faces, last_pos, last_area):
        if not faces or last_pos is None: return None
        best_face = None
        min_dist = float('inf')
        tx, ty = last_pos
        for face in faces:
            fx, fy = face["center"]
            f_area = face["area"]
            dist = math.hypot(fx - tx, fy - ty)
            
            is_area_ok = True
            if last_area > 0:
                ratio = f_area / last_area
                if ratio < 0.6 or ratio > 1.4: is_area_ok = False
            
            if dist < min_dist and is_area_ok and dist < 200:
                min_dist = dist
                best_face = face
        return best_face

    def update_loop(self):
        if self.grabber is not None:
            frame = self.grabber.read()
            
            if time.time() - self.last_battery_check > 5:
                try:
                    bat = self.tello.get_battery()
                    color = "green" if bat > 50 else ("orange" if bat > 20 else "red")
                    self.battery_label.config(text=f"Заряд батареї: {bat}%", fg=color)
                    self.last_battery_check = time.time()
                except: pass

            if frame is not None:
                frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                frame_processed, faces = self.tracker.find_faces(frame_resized)
                
                target_face = None
                
                if self.is_tracking_locked:
                    
                    # === 1. КАЛІБРУВАННЯ ===
                    if self.is_calibrating:
                        if self.locked_face_center is not None:
                            target_face = self.track_by_position(faces, self.locked_face_center, self.locked_face_area)
                        
                        if target_face:
                            self.calibration_data.append(target_face["signature"])
                            count = len(self.calibration_data)
                            self.status_label.config(text=f"Збираю дані... {count}/{self.calibration_frames_target}", fg="blue")
                            
                            self.locked_face_center = target_face["center"]
                            self.locked_face_area = target_face["area"]

                            if count >= self.calibration_frames_target:
                                data_np = np.array(self.calibration_data)
                                self.locked_signature = np.mean(data_np, axis=0).tolist()
                                self.is_calibrating = False
                                self.tracker.reset_counters()
                                self.status_label.config(text="Обличчя вивчено. Починаю стеження.", fg="green")
                        else:
                            self.status_label.config(text="Не бачу обличчя для калібрування!", fg="red")

                    # === 2. СТЕЖЕННЯ ===
                    else:
                        best_candidate = None
                        best_score = 999.0
                        
                        # Знаходимо найкращого кандидата
                        for face in faces:
                            score = self.tracker.compare_signatures(face["signature"], self.locked_signature)
                            if score < best_score:
                                best_score = score
                                best_candidate = face
                        
                        # Відправляємо найкращого на валідацію
                        cand_sig = best_candidate["signature"] if best_candidate else None
                        
                        is_verified_target, msg = self.tracker.validate_match(
                            cand_sig, 
                            self.locked_signature, 
                            is_already_locked=(self.locked_face_center is not None)
                        )
                        
                        self.status_label.config(text=msg, fg="green" if is_verified_target else "purple")

                        if is_verified_target and best_candidate:
                            target_face = best_candidate
                            self.locked_face_center = target_face["center"]
                            self.locked_face_area = target_face["area"]
                        else:
                            target_face = None
                            if "Втрачено довіру" in msg:
                                self.locked_face_center = None 

                # --- МАЛЮВАННЯ ТА PID ---
                pid_yaw, pid_fb, pid_ud = 0, 0, 0
                if faces:
                    for face in faces:
                        # 1. Рахуємо SCORE для кожного обличчя (якщо є еталон)
                        face_score = None
                        if self.locked_signature is not None:
                             face_score = self.tracker.compare_signatures(face["signature"], self.locked_signature)
                        
                        # 2. Малюємо обличчя + Score
                        is_target = (face == target_face)
                        frame_processed = self.tracker.draw_face(frame_processed, face, is_target, self.show_mesh, score=face_score)
                        
                        # 3. PID
                        if is_target:
                            pid_yaw, pid_fb, pid_ud = self.tracker.calculate_pid(face)

                # --- Control ---
                man_lr, man_fb, man_ud, man_yaw = self.get_manual_command()
                final_lr, final_fb, final_ud, final_yaw = man_lr, man_fb, man_ud, man_yaw

                if self.is_tracking_locked and target_face and not self.is_calibrating:
                    if final_yaw == 0: final_yaw = pid_yaw
                    if final_ud == 0: final_ud = pid_ud
                    if self.is_distance_active and final_fb == 0: final_fb = pid_fb
                
                if self.is_flying:
                    self.tello.send_rc_control(final_lr, final_fb, final_ud, final_yaw)
                
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
                self.battery_label.config(text=f"Заряд батареї: {bat}%")
                time.sleep(1)
                self.grabber = VideoGrabber(config.UDP_VIDEO_ADDRESS).start()
            except Exception as e:
                self.status_label.config(text=f"Помилка підключення: {e}", fg="red")
        threading.Thread(target=_connect, daemon=True).start()

    def setup_input(self):
        self.window.bind("<KeyPress>", self.key_down)
        self.window.bind("<KeyRelease>", self.key_up)

    def key_down(self, event):
        self.pressed_keys.add(event.keysym.lower())
        if event.keysym.lower() == 'space': self.land()

    def key_up(self, event):
        k = event.keysym.lower()
        if k in self.pressed_keys: self.pressed_keys.remove(k)

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
        if 'shift_l' in pk: ud = speed
        if 'control_l' in pk: ud = -speed
        return lr, fb, ud, yv

    def takeoff(self):
        try:
            self.tello.takeoff()
            self.tello.send_rc_control(0,0,25,0)
            self.is_flying = True
            self.status_label.config(text="Зліт виконано!", fg="green")
        except: pass

    def land(self):
        self.cancel_tracking()
        try:
            self.tello.land()
            self.is_flying = False
            self.status_label.config(text="Посадка...", fg="red")
        except: pass

    def toggle_dist(self):
        self.is_distance_active = not self.is_distance_active
        self.btn_dist.config(bg="orange" if self.is_distance_active else "gray")

    def toggle_view(self):
        self.show_mesh = not self.show_mesh
        self.btn_view.config(text="Вигляд: Сітка" if self.show_mesh else "Вигляд: Рамка", bg="purple" if self.show_mesh else "#8e44ad")

    def show_help(self):
        messagebox.showinfo("Інфо", "ЛКМ - Захопити\nПКМ - Скинути\nSpace - Посадка")
        self.video_label.focus_set()

    def close(self):
        if self.grabber: self.grabber.stop()
        self.window.destroy()
