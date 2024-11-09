# main.py
from camera_stream import CameraStream
from object_detection import ObjectDetector
from utils import process_detections, draw_detections
import cv2
import time

def main():
    # Initialize components
    stream = CameraStream(source=0)  # Adjust source if needed
    detector = ObjectDetector('yolo11s.pt', device='cpu')  # Use appropriate model and device

    # Target objects to look for
    target_objects = ['person', 'cat', 'dog', 'bird', 'squirrel']

    # Frame skipping parameters
    frame_count = 0
    process_every_n_frames = 5  # Adjust as needed

    try:
        while True:
            frame = stream.get_frame()
            frame_count += 1

            if frame_count % process_every_n_frames == 0:
                # Run object detection on this frame
                detections = detector.detect(frame)
                detected_objects = process_detections(
                    detections, confidence_threshold=0.5)

                # Filter detections for target objects
                targets_in_frame = [
                    obj for obj in detected_objects if obj['label'] in target_objects]

                if targets_in_frame:
                    print("Target object(s) detected:")
                    for obj in targets_in_frame:
                        print(f" - {obj['label']} with confidence {obj['score']:.2f}")
                    # Draw detections on frame
                    frame = draw_detections(frame, targets_in_frame)
                    # Save or process the frame as needed
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f'captures/detection_{timestamp}.jpg', frame)
                    # Optional: Add notification logic here

            # Display the frame (optional)
            cv2.imshow('Camera Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        stream.exit_requested()
        cv2.destroyAllWindows()
        print("Exiting...")

if __name__ == '__main__':
    main()
