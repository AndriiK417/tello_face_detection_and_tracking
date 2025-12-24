import cv2
import threading
import time

class VideoGrabber:
    """
    Клас для асинхронного захоплення кадрів з UDP потоку.
    Допомагає уникнути затримок (лагів) відео.
    """
    def __init__(self, addr):
        self.cap = cv2.VideoCapture(addr)
        # Спроба зменшити буфер (працює залежно від бекенду OpenCV)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue
            
            grabbed, frame = self.cap.read()
            
            if grabbed:
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.cap.release()
