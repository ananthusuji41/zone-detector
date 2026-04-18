import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import secrets
import sqlite3
import threading
import subprocess

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")

USERNAME = "admin"
PASSWORD = "zonedetector123"
sessions = {}

model = YOLO("yolov8n.pt")
model.overrides['imgsz'] = 320

zone_points = []
zone_complete = False
video_writer = None
recording = False
record_frames = 0
RECORD_SECONDS = 10
FPS = 10
alerted_ids = set()
frame_count = 0

def init_db():
    conn = sqlite3.connect("database/alerts.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  person_id INTEGER,
                  timestamp TEXT,
                  clip_filename TEXT)''')
    conn.commit()
    conn.close()

def log_alert(person_id, timestamp, clip_filename):
    conn = sqlite3.connect("database/alerts.db")
    c = conn.cursor()
    c.execute("INSERT INTO alerts (person_id, timestamp, clip_filename) VALUES (?, ?, ?)",
              (person_id, timestamp, clip_filename))
    conn.commit()
    conn.close()

def get_alerts():
    conn = sqlite3.connect("database/alerts.db")
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()
    return rows

def is_inside_zone(cx, cy, zone):
    if len(zone) < 3:
        return False
    zone_array = np.array(zone, dtype=np.int32)
    return cv2.pointPolygonTest(zone_array, (cx, cy), False) >= 0

def convert_to_mp4(avi_path, mp4_path):
    try:
        subprocess.run([
            "ffmpeg", "-i", avi_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-y", mp4_path
        ], capture_output=True)
        os.remove(avi_path)
        print(f"Converted to mp4: {mp4_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")

def start_recording(frame, track_id, timestamp):
    global video_writer, recording, record_frames
    avi_filename = f"alert_person{track_id}_{timestamp}.avi"
    avi_filepath = os.path.join("alerts", avi_filename)
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    video_writer = cv2.VideoWriter(avi_filepath, fourcc, FPS, (w, h))
    recording = True
    record_frames = 0
    print(f"Recording started: {avi_filepath}")
    return avi_filename

def finish_recording(avi_filename):
    avi_path = os.path.join("alerts", avi_filename)
    mp4_filename = avi_filename.replace(".avi", ".mp4")
    mp4_path = os.path.join("alerts", mp4_filename)
    t = threading.Thread(target=convert_to_mp4, args=(avi_path, mp4_path))
    t.start()
    return mp4_filename

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

current_avi = None

def generate_frames():
    global zone_complete, video_writer, recording, record_frames
    global alerted_ids, frame_count, current_avi
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if zone_complete:
            if frame_count % 2 == 0:
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
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
                    cv2.putText(frame, f"ALERT: Intruder #{alert_id} detected!", (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if not recording:
                        avi_filename = start_recording(frame, alert_id, timestamp)
                        current_avi = avi_filename
                        mp4_filename = avi_filename.replace(".avi", ".mp4")
                        log_alert(alert_id, timestamp, mp4_filename)

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
                if current_avi:
                    finish_recording(current_avi)
                    current_avi = None

        zone_array = np.array(zone_points, dtype=np.int32)
        if len(zone_points) >= 2:
            cv2.polylines(frame, [zone_array], zone_complete, (0, 0, 255), 2)
        for pt in zone_points:
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        if not zone_complete:
            cv2.putText(frame, "Zone not set - use dashboard to draw zone",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        ret2, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/login")
async def login(request: Request, username: str = Form(), password: str = Form()):
    if username == USERNAME and password == PASSWORD:
        token = secrets.token_hex(16)
        sessions[token] = True
        response = RedirectResponse(url="/dashboard", status_code=302)
        response.set_cookie("session", token, httponly=True, max_age=1800)
        return response
    return RedirectResponse(url="/login?error=1", status_code=302)

@app.get("/logout")
async def logout(request: Request):
    token = request.cookies.get("session")
    if token in sessions:
        del sessions[token]
    return RedirectResponse(url="/login", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    token = request.cookies.get("session")
    if token not in sessions:
        return RedirectResponse(url="/login")
    alerts = get_alerts()
    return templates.TemplateResponse("dashboard.html", {"request": request, "alerts": alerts})

@app.get("/video_feed")
async def video_feed(request: Request):
    token = request.cookies.get("session")
    if token not in sessions:
        return RedirectResponse(url="/login")
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace;boundary=frame")

@app.post("/set_zone")
async def set_zone(request: Request):
    token = request.cookies.get("session")
    if token not in sessions:
        return RedirectResponse(url="/login")
    global zone_points, zone_complete, alerted_ids
    data = await request.json()
    zone_points = [(int(p['x']), int(p['y'])) for p in data['points']]
    zone_complete = len(zone_points) >= 3
    alerted_ids = set()
    return {"status": "ok", "points": len(zone_points)}

@app.get("/alerts_data")
async def alerts_data(request: Request):
    token = request.cookies.get("session")
    if token not in sessions:
        return JSONResponse({"alerts": []})
    alerts = get_alerts()
    return JSONResponse({"alerts": [list(a) for a in alerts]})

@app.post("/clear_alerts")
async def clear_alerts(request: Request):
    token = request.cookies.get("session")
    if token not in sessions:
        return RedirectResponse(url="/login")
    conn = sqlite3.connect("database/alerts.db")
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()
    return {"status": "cleared"}

@app.get("/")
async def root():
    return RedirectResponse(url="/login")

init_db()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)