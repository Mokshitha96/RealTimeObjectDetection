import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("../data/yolov3.weights", "../data/yolov3.cfg")

# Load class names
with open("../data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names (universal fix)
layer_names = net.getLayerNames()
output_layers = [layer_names[int(i) - 1] for i in net.getUnconnectedOutLayers()]

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

print("[INFO] Starting real-time object detection...")

while True:
    ret, img = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    print("[DEBUG] Frame size:", img.shape)

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    print("[DEBUG] Got output from YOLO")

    class_ids = []
    confidences = []
    boxes = []

    print("[DEBUG] Checking detections...")
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            print(f"[DEBUG] Confidence: {confidence:.2f} — Class: {classes[class_id]}")
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    print(f"[DEBUG] Indexes returned: {indexes}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in indexes:
        i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]} ({confidences[i]:.2f})"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), font, 0.5, color, 2)
        print(f"[INFO] Detected: {label}")

    cv2.imshow("YOLO Object Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
