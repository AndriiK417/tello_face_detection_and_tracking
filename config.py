# --- config.py ---

# Налаштування мережі
DRONE_IP = '192.168.10.1'
UDP_VIDEO_ADDRESS = 'udp://192.168.10.1:11111'

# Налаштування зображення
FRAME_WIDTH = 360
FRAME_HEIGHT = 240

# Налаштування PID контролера [Kp, Ki, Kd]
PID_COEFFICIENTS = [0.4, 0.4, 0]

# Діапазон площі обличчя для дистанції [min, max]
FACE_AREA_RANGE = [6200, 6800]

# Швидкість ручного керування (0-100)
MANUAL_SPEED = 50  # <-- НОВЕ

# Шляхи до файлів
CASCADE_PATH = "haarcascade_frontalface_default.xml"
