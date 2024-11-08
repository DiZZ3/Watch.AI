# motion_detector.py
import cv2

class MotionDetector:
    def __init__(self):
        self.previous_frame = None

    def detect(self, frame):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise and detail
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return False

        # Compute the absolute difference between the current frame and previous frame
        frame_diff = cv2.absdiff(self.previous_frame, gray_frame)
        self.previous_frame = gray_frame

        # Apply threshold to get binary image
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        # Dilate the thresholded image to fill in holes
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours on the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if any contour area is above a certain threshold
        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Adjust the threshold as needed
                return True

        return False
