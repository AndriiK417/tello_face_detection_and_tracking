import customtkinter as ctk
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
import cv2
import time
import threading
import math
import numpy as np 
from djitellopy import Tello

import config
from videograbber import VideoGrabber
from tracker import FaceTracker, BIOMETRIC_THRESHOLD

# Налаштування вигляду CustomTkinter
ctk.set_appearance_mode("dark")  # Темна тема
ctk.set_default_color_theme("blue")  # Синій акцент

class MainWindow:
    def __init__(self, root):
        self.window = root
        self.window.title("Супровід Доповідача | Автономна система")
        self.window.geometry("950x800")
        
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
        # Головний шрифт для статусів
        font_main = ctk.CTkFont(family="Roboto", size=18, weight="bold")
        font_status = ctk.CTkFont(family="Consolas", size=16)

        # 1. БАТАРЕЯ (Верхня панель)
        self.battery_label = ctk.CTkLabel(
            self.window, 
            text="Заряд батареї: --%", 
            font=font_main, 
            text_color="#a8a8a8"
        )
        self.battery_label.pack(pady=(15, 5))

        # 2. ВІДЕО
        # У CTkLabel обов'язково текст="", щоб він не перекривав відео
        self.video_label = ctk.CTkLabel(self.window, text="", fg_color="black", corner_radius=10)
        self.video_label.pack(pady=10)
        self.video_label.focus_set()
        
        self.video_label.bind("<Button-1>", self.on_mouse_click)
        self.video_label.bind("<Button-3>", self.cancel_tracking)

        # 3. СТАТУС
        self.status_label = ctk.CTkLabel(
            self.window, 
            text="Система готова. Чекаю наказу.", 
            font=font_status, 
            text_color="#1f6aa5"
        )
        self.status_label.pack(pady=5)

        # 4. ІНФО
        self.info_label = ctk.CTkLabel(
            self.window, 
            text="ЛКМ: Захопити | ПКМ: Скинути ціль | Space: Екстрена посадка | WASD+Shift/Ctrl: Ручне керування", 
            font=ctk.CTkFont(size=12), 
            text_color="gray"
        )
        self.info_label.pack(pady=5)

        # 5. ПАНЕЛЬ КНОПОК
        controls_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        controls_frame.pack(side="bottom", fill="x", pady=20, padx=20)
        
        # Центрування кнопок
        controls_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        btn_opts = {"height": 40, "corner_radius": 8, "font": ctk.CTkFont(weight="bold")}

        self.btn_takeoff = ctk.CTkButton(
            controls_frame, text="ЗЛІТ", fg_color="#28a745", hover_color="#218838", 
            command=self.takeoff, **btn_opts
        )
        self.btn_takeoff.grid(row=0, column=0, padx=10)

        self.btn_land = ctk.CTkButton(
            controls_frame, text="ПОСАДКА", fg_color="#dc3545", hover_color="#c82333", 
            command=self.land, **btn_opts
        )
        self.btn_land.grid(row=0, column=1, padx=10)

        self.btn_dist = ctk.CTkButton(
            controls_frame, text="Дистанція: ВИМК", fg_color="#6c757d", hover_color="#5a6268", 
            command=self.toggle_dist, **btn_opts
        )
        self.btn_dist.grid(row=0, column=2, padx=10)

        self.btn_view = ctk.CTkButton(
            controls_frame, text="Вигляд: Сітка", fg_color="#6f42c1", hover_color="#59339d", 
            command=self.toggle_view, **btn_opts
        )
        self.btn_view.grid(row=0, column=3, padx=10)

        self.btn_info = ctk.CTkButton(
            controls_frame, text="Довідка", fg_color="#17a2b8", hover_color="#138496", 
            command=self.show_help, **btn_opts
        )
        self.btn_info.grid(row=0, column=4, padx=10)


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
        
        self.status_label.configure(text="Починаю вивчення обличчя... Не рухайся!", text_color="#1f6aa5")

    def cancel_tracking(self, event=None):
        self.is_tracking_locked = False
        self.is_calibrating = False
        self.locked_face_center = None
        self.locked_signature = None
        self.tracker.reset_counters()
        self.status_label.configure(text="Стеження зупинено. Чекаю нову ціль.", text_color="gray")

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

    def draw_face_on_image(self, img, face_data, is_locked=False, draw_mesh=True, score=None):
        cx, cy = face_data["center"]
        x1, y1, x2, y2 = face_data["bbox"]
        
        color = (0, 0, 255) if is_locked else (0, 255, 0)
        
        if draw_mesh:
            for point in face_data["id"]:
                cv2.circle(img, point, 1, color, cv2.FILLED)
        else:
            pad = 10
            cv2.rectangle(img, (x1-pad, y1-pad), (x2+pad, y2+pad), color, 2)
        
        label = "LOCKED" if is_locked else ""
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if score is not None:
            score_text = f"Diff: {score:.3f}"
            score_color = (0, 255, 0) if score < BIOMETRIC_THRESHOLD else (0, 0, 255)
            cv2.putText(img, score_text, (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
        
        if is_locked:
            cv2.line(img, (cx-10, cy), (cx+10, cy), (0, 0, 255), 2)
            cv2.line(img, (cx, cy-10), (cx, cy+10), (0, 0, 255), 2)

        return img

    def update_loop(self):
        if self.grabber is not None:
            frame = self.grabber.read()
            
            if time.time() - self.last_battery_check > 5:
                try:
                    bat = self.tello.get_battery()
                    b_color = "#28a745" if bat > 50 else ("#ffc107" if bat > 20 else "#dc3545")
                    self.battery_label.configure(text=f"Заряд батареї: {bat}%", text_color=b_color)
                    self.last_battery_check = time.time()
                except: pass

            if frame is not None:
                frame_resized = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                frame_processed, faces = self.tracker.find_faces(frame_resized)
                
                target_face = None
                
                if self.is_tracking_locked:
                    
                    if self.is_calibrating:
                        if self.locked_face_center is not None:
                            target_face = self.track_by_position(faces, self.locked_face_center, self.locked_face_area)
                        
                        if target_face:
                            self.calibration_data.append(target_face["signature"])
                            count = len(self.calibration_data)
                            self.status_label.configure(text=f"Збираю дані... {count}/{self.calibration_frames_target}", text_color="#1f6aa5")
                            
                            self.locked_face_center = target_face["center"]
                            self.locked_face_area = target_face["area"]

                            if count >= self.calibration_frames_target:
                                data_np = np.array(self.calibration_data)
                                self.locked_signature = np.mean(data_np, axis=0).tolist()
                                self.is_calibrating = False
                                self.tracker.reset_counters()
                                self.status_label.configure(text="Обличчя вивчено. Починаю стеження.", text_color="#28a745")
                        else:
                            self.status_label.configure(text="Не бачу обличчя для калібрування!", text_color="#dc3545")

                    else:
                        best_candidate = None
                        best_score = 999.0
                        
                        for face in faces:
                            score = self.tracker.compare_signatures(face["signature"], self.locked_signature)
                            if score < best_score:
                                best_score = score
                                best_candidate = face
                        
                        cand_sig = best_candidate["signature"] if best_candidate else None
                        
                        is_verified_target, msg = self.tracker.validate_match(
                            cand_sig, 
                            self.locked_signature, 
                            is_already_locked=(self.locked_face_center is not None)
                        )
                        
                        status_color = "#28a745" if is_verified_target else "#fd7e14"
                        self.status_label.configure(text=msg, text_color=status_color)

                        if is_verified_target and best_candidate:
                            target_face = best_candidate
                            self.locked_face_center = target_face["center"]
                            self.locked_face_area = target_face["area"]
                        else:
                            target_face = None
                            if "Втрачено довіру" in msg:
                                self.locked_face_center = None 

                pid_yaw, pid_fb, pid_ud = 0, 0, 0
                if faces:
                    for face in faces:
                        face_score = None
                        if self.locked_signature is not None:
                             face_score = self.tracker.compare_signatures(face["signature"], self.locked_signature)
                        
                        is_target = (face == target_face)
                        
                        frame_processed = self.draw_face_on_image(
                            frame_processed, face, is_target, self.show_mesh, score=face_score
                        )
                        
                        if is_target:
                            pid_yaw, pid_fb, pid_ud = self.tracker.calculate_pid(face)

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
                img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(720, 540))
                
                self.video_label.configure(image=img_tk)
                self.video_label.image = img_tk

        self.window.after(10, self.update_loop)

    def connect_drone(self):
        def _connect():
            try:
                self.tello.connect()
                self.tello.streamoff()
                self.tello.streamon()
                bat = self.tello.get_battery()
                self.battery_label.configure(text=f"Заряд батареї: {bat}%")
                time.sleep(1)
                self.grabber = VideoGrabber(config.UDP_VIDEO_ADDRESS).start()
            except Exception as e:
                self.status_label.configure(text=f"Помилка підключення: {e}", text_color="#dc3545")
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
            self.status_label.configure(text="Зліт виконано!", text_color="#28a745")
        except: pass

    def land(self):
        self.cancel_tracking()
        try:
            self.tello.land()
            self.is_flying = False
            self.status_label.configure(text="Посадка...", text_color="#dc3545")
        except: pass

    def toggle_dist(self):
        self.is_distance_active = not self.is_distance_active
        if self.is_distance_active:
            self.btn_dist.configure(text="Дистанція: УВІМК", fg_color="#fd7e14", hover_color="#e36209")
        else:
            self.btn_dist.configure(text="Дистанція: ВИМК", fg_color="#6c757d", hover_color="#5a6268")

    def toggle_view(self):
        self.show_mesh = not self.show_mesh
        if self.show_mesh:
            self.btn_view.configure(text="Вигляд: Сітка", fg_color="#6f42c1", hover_color="#59339d")
        else:
            self.btn_view.configure(text="Вигляд: Рамка", fg_color="#007bff", hover_color="#0056b3")

    def show_help(self):
        messagebox.showinfo("Інфо", "ЛКМ: Захопити ціль\nПКМ: Скинути ціль\nSpace: Посадка")
        self.video_label.focus_set()

    def close(self):
        if self.grabber: self.grabber.stop()
        self.window.destroy()
