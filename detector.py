import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

model = YOLO("yolov8n.pt")
model.overrides['imgsz'] = 320

zone_points = []
zone_complete = False
video_writer = None
recording = False
record_frames = 0
RECORD_SECONDS = 10
FPS = 20
ALERTS_FOLDER = "alerts"
alerted_ids = set()

def mouse_click(event, x, y, flags, param):
    global zone_points, zone_complete
    if event == cv2.EVENT_LBUTTONDOWN:
        if not zone_complete:
            zone_points.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(zone_points) >= 3:
            zone_complete = True

def is_inside_zone(cx, cy, zone):
    if len(zone) < 3:
        return False
    zone_array = np.array(zone, dtype=np.int32)
    return cv2.pointPolygonTest(zone_array, (cx, cy), False) >= 0

def start_recording(frame, track_id):
    global video_writer, recording, record_frames
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(ALERTS_FOLDER, f"alert_person{track_id}_{timestamp}.mp4")
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(filename, fourcc, FPS, (w, h))
    recording = True
    record_frames = 0
    print(f"Recording started: {filename}")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Zone Detector")
cv2.setMouseCallback("Zone Detector", mouse_click)

print("LEFT CLICK to place zone points")
print("RIGHT CLICK to complete the zone")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if zone_complete:
        results = model.track(frame, classes=[0], verbose=False, persist=True)[0]
        alert = False
        alert_id = None

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            track_id = int(box.id[0]) if box.id is not None else 0
            inside = is_inside_zone(cx, cy, zone_points)
            color = (0, 0, 255) if inside else (0, 255, 0)
            label = f"INTRUDER #{track_id}" if inside else f"person #{track_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if inside and track_id not in alerted_ids:
                alert = True
                alert_id = track_id

        if alert and alert_id is not None:
            alerted_ids.add(alert_id)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(frame, f"ALERT: Intruder #{alert_id} detected!", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if not recording:
                start_recording(frame, alert_id)

    if recording:
        video_writer.write(frame)
        record_frames += 1
        remaining = RECORD_SECONDS - (record_frames // FPS)
        cv2.putText(frame, f"Recording... {remaining}s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if record_frames >= RECORD_SECONDS * FPS:
            video_writer.release()
            recording = False
            print("Recording saved.")

    zone_array = np.array(zone_points, dtype=np.int32)
    if len(zone_points) >= 2:
        cv2.polylines(frame, [zone_array], zone_complete, (0, 0, 255), 2)
    for pt in zone_points:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

    if not zone_complete:
        cv2.putText(frame, "Left click to draw zone. Right click to finish.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Zone Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if recording:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()