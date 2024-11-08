# camera_stream.py
import cv2

class CameraStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise Exception('Failed to read frame from camera.')
        return frame

    def display_frame(self, frame):
        cv2.imshow('Camera Feed', frame)

    def exit_requested(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()