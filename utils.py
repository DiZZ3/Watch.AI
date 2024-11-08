# utils.py
import cv2

def process_detections(detections, confidence_threshold=0.5):
    detected_objects = []
    boxes = detections.boxes  # Bounding boxes
    for box in boxes:
        score = box.conf.item()
        if score >= confidence_threshold:
            label_index = int(box.cls.item())
            label = detections.names[label_index]
            coords = box.xyxy[0].cpu().numpy().astype(int)
            detected_objects.append({
                'label': label,
                'box': coords,
                'score': score
            })
    return detected_objects

def draw_detections(frame, detections):
    for obj in detections:
        label = obj['label']
        box = obj['box']
        score = obj['score']
        # Draw bounding box
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # Put label text above the bounding box
        text = f'{label}: {score:.2f}'
        cv2.putText(frame, text, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame
