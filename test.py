#!/usr/bin/env python3
"""
YOLOv8 Detection - Burst Capture + InsightFace Training & Identification
UI: Redesigned — industrial terminal aesthetic
"""

import os
import re
import csv
import cv2
import time
import pickle
import zipfile
import signal
import sys
import logging
import threading
import subprocess
import numpy as np
from datetime import datetime
from io import BytesIO, StringIO
from flask import Flask, Response, jsonify, send_file, request

import insightface
from insightface.app import FaceAnalysis

class NoStatusFilter(logging.Filter):
    def filter(self, record):
        return "/status" not in record.getMessage()
logging.getLogger("werkzeug").addFilter(NoStatusFilter())

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath("/app/detections" if os.path.exists("/app") else "./detections")
TRAIN_DIR = os.path.join(BASE_DIR, "training_images")
EMB_FILE  = os.path.join(BASE_DIR, "face_embeddings.pkl")
CSV_FILE  = os.path.join(BASE_DIR, "detections.csv")

for d in [BASE_DIR, TRAIN_DIR]:
    os.makedirs(d, exist_ok=True)

CSV_LOCK = threading.Lock()
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp","Frame_Number","Objects_Detected","Image_File"])

print(f"[INIT] BASE_DIR  : {BASE_DIR}")
print(f"[INIT] TRAIN_DIR : {TRAIN_DIR}")

# ──────────────────────────────────────────────────────────────────────────────
# InsightFace
# ──────────────────────────────────────────────────────────────────────────────
print("[INIT] Loading InsightFace model…", flush=True)
face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))
print("[INIT] InsightFace ready ✅", flush=True)

face_db: dict = {}

def load_face_db():
    global face_db
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE, "rb") as f:
            face_db = pickle.load(f)
        print(f"[INIT] Loaded {len(face_db)} trained person(s): {list(face_db.keys())}", flush=True)
    else:
        face_db = {}

def save_face_db():
    with open(EMB_FILE, "wb") as f:
        pickle.dump(face_db, f)

load_face_db()

# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────
state = {
    "mode"                    : "idle",
    "proc"                    : None,
    "lock"                    : threading.Lock(),
    "latest_frame"            : None,
    "frame_ready"             : False,
    "frame_count"             : 0,
    "fps"                     : 0.0,
    "current_block"           : [],
    "last_block"              : [],
    "detecting"               : False,
    "duration"                : 5,
    "num_images"              : 10,
    "capture_interval"        : 0.5,
    "frames_captured_in_burst": 0,
    "last_capture_time"       : 0.0,
    "burst_image_paths"       : [],
    "burst_csv_rows"          : [],
    "total_captures"          : 0,
    "last_error"              : "",
    "training"                : False,
    "train_name"              : "",
    "train_target"            : 10,
    "train_collected"         : 0,
    "train_last_capture"      : 0.0,
    "train_interval"          : 0.5,
    "train_status"            : "",
    "model_ready"             : len(face_db) > 0,
    "behaviour_monitoring" : False,
    "behaviour_events"     : [],
}

FACE_THRESHOLD = 0.45

# ──────────────────────────────────────────────────────────────────────────────
# Core logic (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
def stop_stream():
    p = state["proc"]
    if p:
        p.terminate()
        try: p.wait(timeout=3)
        except Exception: p.kill()
    state.update({
        "proc": None, "latest_frame": None, "frame_ready": False,
        "mode": "idle", "detecting": False, "training": False,
        "frames_captured_in_burst": 0, "current_block": [], "last_block": [],
    })
    print("\n[STOP] Stream stopped", flush=True)
    print("=" * 60, flush=True)

def collect_training_image(frame: bytes, name: str, index: int):
    person_dir = os.path.join(TRAIN_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    arr = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return None
    path = os.path.join(person_dir, f"{name}_{index:03d}.jpg")
    cv2.imwrite(path, img)
    return path

def train_person(name: str) -> str:
    person_dir = os.path.join(TRAIN_DIR, name)
    if not os.path.exists(person_dir):
        return f"❌ No training images found for '{name}'"
    images = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    if not images:
        return f"❌ No images in training folder for '{name}'"
    embeddings, failed = [], 0
    for img_file in images:
        img = cv2.imread(os.path.join(person_dir, img_file))
        if img is None: failed += 1; continue
        faces = face_app.get(img)
        if not faces: failed += 1; continue
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb  = face.embedding / np.linalg.norm(face.embedding)
        embeddings.append(emb)
    if not embeddings:
        return f"❌ No faces detected in any training image ({failed} failed)"
    avg_emb = np.mean(embeddings, axis=0)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    face_db[name] = avg_emb
    state["model_ready"] = True
    save_face_db()
    msg = f"✅ '{name}' trained — {len(embeddings)} faces used, {failed} skipped"
    print(f"[TRAIN] {msg}", flush=True)
    return msg

HAILO_RE = re.compile(r"Object:\s+([a-zA-Z][a-zA-Z0-9 _-]*?)\s*\[\d+\]\s+\(([\d.]+)\)")

def parse_last_block(block_lines: list) -> dict:
    best = {}
    for line in block_lines:
        m = HAILO_RE.search(line)
        if m:
            label, conf = m.group(1).strip().lower(), float(m.group(2))
            if conf >= 0.1 and (label not in best or conf > best[label]):
                best[label] = conf
    return best

def identify_persons(jpeg_bytes: bytes, hailo_objects: dict) -> dict:
    if "person" not in hailo_objects or not face_db: return hailo_objects
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None: return hailo_objects
    faces = face_app.get(img)
    if not faces: return hailo_objects
    result = {k: v for k, v in hailo_objects.items() if k != "person"}
    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        best_name, best_score = "Unknown", -1.0
        for name, avg_emb in face_db.items():
            score = float(np.dot(emb, avg_emb))
            if score > best_score: best_score, best_name = score, name
        if best_score < FACE_THRESHOLD: best_name = "Unknown"
        print(f"   👤 Identified: {best_name} ({best_score:.2f})", flush=True)
        result[best_name] = round(best_score, 2)
    return result

def save_frame(frame: bytes, frame_num: int, last_block: list):
    if not frame or len(frame) < 1000:
        print(f"[WARN] Frame #{frame_num} too small, retrying", flush=True); return False
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(BASE_DIR, date_str)
    os.makedirs(date_dir, exist_ok=True)
    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_name = f"IMG_{ts_str}_frame{frame_num:02d}.jpg"
    img_path = os.path.join(date_dir, img_name)
    img_rel  = os.path.join(date_str, img_name)
    try:
        with open(img_path, "wb") as f: f.write(frame)
        print(f"✅ [SAVE] Frame {frame_num}/{state['num_images']} → {img_name}", flush=True)
    except Exception as e:
        print(f"❌ [ERROR] Save failed: {e}", flush=True); return False
    state["burst_image_paths"].append(img_path)
    detected = parse_last_block(last_block)
    if "person" in detected and state["model_ready"]:
        print(f"   🔎 Person found — running InsightFace…", flush=True)
        detected = identify_persons(frame, detected)
    timestamp   = datetime.now().isoformat()
    objects_str = ", ".join(f"{l}({c:.2f})" for l, c in sorted(detected.items(), key=lambda x:-x[1])) if detected else "none"
    print(f"   🔍 Detections: {objects_str}", flush=True)
    row = [timestamp, frame_num, objects_str, img_rel]
    with CSV_LOCK:
        try:
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f: csv.writer(f).writerow(row)
            state["burst_csv_rows"].append(row)
        except Exception as e: print(f"❌ [ERROR] CSV write: {e}", flush=True)
    state["total_captures"] += 1
    return True

def read_frames(proc):
    buffer = b""
    t0 = time.time()
    print("[INFO] Frame reader started", flush=True)
    while state["mode"] != "idle":
        try: chunk = proc.stdout.read(65536)
        except Exception as e: print(f"[ERROR] stdout read: {e}", flush=True); break
        if not chunk: break
        buffer += chunk
        while True:
            s = buffer.find(b"\xff\xd8")
            if s == -1: buffer = b""; break
            e = buffer.find(b"\xff\xd9", s + 2)
            if e == -1: buffer = buffer[s:]; break
            frame  = buffer[s:e + 2]
            buffer = buffer[e + 2:]
            if len(frame) > 1000:
                with state["lock"]:
                    state["latest_frame"] = frame
                    state["frame_ready"]  = True
                state["frame_count"] += 1
                if state["training"] and state["train_name"]:
                    now = time.time()
                    collected, target = state["train_collected"], state["train_target"]
                    if collected < target and now - state["train_last_capture"] >= state["train_interval"]:
                        idx  = collected + 1
                        path = collect_training_image(frame, state["train_name"], idx)
                        if path:
                            state["train_collected"]   += 1
                            state["train_last_capture"] = now
                            state["train_status"] = f"📸 Collecting {state['train_collected']}/{target} for '{state['train_name']}'…"
                            print(f"[TRAIN] {state['train_status']}", flush=True)
                    if state["train_collected"] >= target and state["training"]:
                        state["training"]     = False
                        state["train_status"] = f"⚙️ Training '{state['train_name']}'…"
                        def do_train(name):
                            result = train_person(name)
                            state["train_status"] = result
                            state["mode"] = "preview"
                        threading.Thread(target=do_train, args=(state["train_name"],), daemon=True).start()
                if state["frame_count"] % 30 == 0:
                    el = time.time() - t0
                    if el > 0: state["fps"] = round(state["frame_count"] / el, 1)
    print("[INFO] Frame reader exited", flush=True)

def monitor_stderr(proc):
    in_block = False
    print("[INFO] Monitor started", flush=True)
    for raw in proc.stderr:
        line = raw.decode(errors="ignore").strip()
        if not line: continue
        state["last_error"] = line
        if line.startswith("Object:") or "------" in line or "Camera started" in line or "ERROR" in line or "Warning" in line:
            print(f"[RPICAM] {line}", flush=True)
        m = re.search(r"FPS[:\s]*([\d.]+)", line, re.IGNORECASE)
        if m: state["fps"] = float(m.group(1))
        if "------" in line:
            if in_block:
                state["last_block"] = list(state["current_block"])
                state["current_block"] = []
                in_block = False
            else:
                state["current_block"] = []
                in_block = True
        elif in_block and line.startswith("Object:"):
            state["current_block"].append(line)
        if not state["detecting"]: continue
        now = time.time()
        if not state["frame_ready"]: continue
        if state["last_capture_time"] == 0.0:
            state["last_capture_time"] = now
            print(f"[INFO] Burst started — {state['num_images']} images every {state['capture_interval']:.3f}s", flush=True)
            continue
        if now - state["last_capture_time"] >= state["capture_interval"]:
            with state["lock"]: frame = state["latest_frame"]
            frame_num = state["frames_captured_in_burst"] + 1
            print(f"\n⏱️  [CAPTURE] Image {frame_num}/{state['num_images']}", flush=True)
            success = save_frame(frame, frame_num, list(state["last_block"]))
            if success: state["frames_captured_in_burst"] += 1
            state["last_capture_time"] = now
    print("[INFO] Monitor exited", flush=True)

def start_preview():
    stop_stream()
    state["mode"] = "preview"
    cmd = ["rpicam-vid","--codec","mjpeg","--output","-","--timeout","0",
           "--width","1280","--height","720","--rotation","180",
           "--framerate","30","--nopreview","-n"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    state["proc"] = proc
    threading.Thread(target=read_frames,    args=(proc,), daemon=True).start()
    threading.Thread(target=monitor_stderr, args=(proc,), daemon=True).start()

def start_training_capture(name: str, num_images: int):
    if state["mode"] not in ("preview", "training"):
        start_preview(); time.sleep(1.0)
    state.update({
        "mode": "training", "training": True, "train_name": name,
        "train_target": num_images, "train_collected": 0,
        "train_last_capture": 0.0,
        "train_status": f"📸 Starting capture for '{name}'…",
    })
    print(f"\n[TRAIN] Collecting {num_images} images for '{name}'", flush=True)

def start_detection(duration: float, num_images: int):
    stop_stream()
    capture_interval = duration / num_images
    state.update({
        "mode": "detecting", "detecting": True, "duration": duration,
        "num_images": num_images, "capture_interval": capture_interval,
        "frames_captured_in_burst": 0, "last_capture_time": 0.0,
        "frame_ready": False, "latest_frame": None, "frame_count": 0,
        "fps": 0.0, "current_block": [], "last_block": [],
        "burst_image_paths": [], "burst_csv_rows": [],
    })
    cmd = ["rpicam-vid",
           "--post-process-file", "/usr/share/rpi-camera-assets/hailo_yolov8_inference.json",
           "--codec","mjpeg","--output","-","--timeout","0",
           "--width","1280","--height","720","--rotation","180",
           "--framerate","30","--nopreview","-n","--verbose","2"]
    print("\n" + "=" * 60, flush=True)
    print(f"🚀 [DETECT] {num_images} images in {duration}s → every {capture_interval:.3f}s", flush=True)
    print(f"👤 Trained: {list(face_db.keys()) or 'none'} | Model: {state['model_ready']}", flush=True)
    print("=" * 60 + "\n", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    state["proc"] = proc
    threading.Thread(target=read_frames,    args=(proc,), daemon=True).start()
    threading.Thread(target=monitor_stderr, args=(proc,), daemon=True).start()

def generate_stream():
    while True:
        with state["lock"]: frame = state["latest_frame"]
        if frame: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        else: time.sleep(0.05)


# ──────────────────────────────────────────────────────────────────────────────
# Behaviour Monitoring (YOLOv8l Pose - Hailo)
# ──────────────────────────────────────────────────────────────────────────────

def analyze_pose_block(block_lines: list):
    behaviours = []

    for line in block_lines:
        if "pose" in line.lower():
            behaviours.append("movement_detected")

    if len(block_lines) > 5:
        behaviours.append("crowd_activity")

    return behaviours


def start_behaviour_monitoring():
    stop_stream()

    state.update({
        "mode": "behaviour",
        "behaviour_monitoring": True,
        "frame_ready": False,
        "latest_frame": None,
        "frame_count": 0,
        "fps": 0.0,
        "current_block": [],
        "last_block": [],
    })

    cmd = [
    "rpicam-vid",
    "--post-process-file", "/usr/share/rpi-camera-assets/hailo_yolov8_pose.json",

    "--width", "1280",
    "--height", "720",

    "--viewfinder-width", "0",
    "--viewfinder-height", "0",

    "--codec", "mjpeg",
    "--output", "-",
    "--timeout", "0",

    "--rotation", "180",   # keep rotation fix
    "--framerate", "20",

    "--nopreview",
    "-n",
    "--verbose", "2"
]

    print("\n🎯 [BEHAVIOUR] YOLOv8l Pose monitoring started", flush=True)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    state["proc"] = proc

    def behaviour_monitor():
        while state["mode"] == "behaviour":
            time.sleep(0.5)
            block = list(state["last_block"])
            if not block:
                continue

            behaviours = analyze_pose_block(block)
            if behaviours:
                event = {
                    "timestamp": datetime.now().isoformat(),
                    "behaviours": behaviours
                }
                state["behaviour_events"].append(event)
                print(f"🧠 Behaviour detected: {behaviours}", flush=True)

    threading.Thread(target=read_frames, args=(proc,), daemon=True).start()
    threading.Thread(target=monitor_stderr, args=(proc,), daemon=True).start()
    threading.Thread(target=behaviour_monitor, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# UI — Industrial Terminal Redesign
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Classroom Monitor</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #080b0f;
  --surface:  #0d1117;
  --border:   #1e2a38;
  --border2:  #243040;
  --accent:   #00e5ff;
  --accent2:  #ff3d71;
  --green:    #00ff88;
  --amber:    #ffb300;
  --purple:   #b040ff;
  --text:     #c8d8e8;
  --muted:    #4a6070;
  --mono:     'Space Mono', monospace;
  --cond:     'Barlow Condensed', sans-serif;
}

* { margin:0; padding:0; box-sizing:border-box; }

html, body {
  height: 100%;
  overflow: hidden;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  min-height: 100%;
}

.app-shell {
  height: 100vh;
  overflow: hidden;
}

/* Scanline overlay */
body::before {
  content:'';
  position:fixed; inset:0; z-index:999; pointer-events:none;
  background: repeating-linear-gradient(
    0deg,
    transparent, transparent 2px,
    rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px
  );
}

/* Grid bg */
body::after {
  content:'';
  position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(0,229,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
}

.wrap {
  position:relative; z-index:1;
  width: 100%;
  max-width: none;
  margin: 0;
  padding: 0 16px;
  height: calc(100vh - 74px);
  overflow: hidden;
}

/* Layout — main + event feed */
.layout {
  display: grid;
  grid-template-columns: 1fr 360px;
  gap: 16px;
  align-items: start;
  height: 100%;
}

/* Event feed sidebar */
.event-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.event-top {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
}
.event-title {
  font-family: var(--cond);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #fff;
}
.event-actions {
  display:flex;
  align-items:center;
  gap:10px;
  font-size:11px;
  color: var(--muted);
}
.event-tabs {
  display:flex;
  gap:8px;
  margin-top:10px;
}
.chip {
  font-family: var(--cond);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  padding: 4px 10px;
  border: 1px solid var(--border2);
  border-radius: 999px;
  cursor: pointer;
  color: var(--muted);
  background: rgba(255,255,255,0.02);
}
.chip.active {
  color: var(--accent);
  border-color: rgba(0,229,255,0.35);
  background: rgba(0,229,255,0.06);
}
.event-list {
  flex: 1;
  overflow: auto;
  margin-top: 12px;
  padding-right: 4px;
}
.event-empty {
  color: var(--muted);
  font-size: 11px;
  line-height: 1.8;
  text-align: center;
  margin-top: 16px;
}
.event-item {
  border: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
  border-radius: 3px;
  padding: 10px 12px;
  margin-bottom: 10px;
}
.event-item .k {
  color: var(--muted);
  font-size: 10px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  font-family: var(--cond);
}
.event-item .v {
  color: #fff;
  font-family: var(--mono);
  font-size: 11px;
  margin-top: 4px;
  word-break: break-word;
}

/* Bottom control bar */
.bottom-bar {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 1000;
  border-top: 1px solid var(--border);
  background: rgba(8,11,15,0.92);
  backdrop-filter: blur(8px);
}
.bottom-inner {
  max-width: 1280px;
  margin: 0 auto;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
}

/* ── HEADER ── */
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 0 18px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
}

.logo {
  display: flex;
  align-items: baseline;
  gap: 12px;
}

.logo-main {
  font-family: var(--cond);
  font-size: 28px;
  font-weight: 900;
  letter-spacing: 4px;
  color: #fff;
  text-transform: uppercase;
}

.logo-sub {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--accent);
  letter-spacing: 2px;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

.sys-time {
  font-size: 11px;
  color: var(--muted);
  letter-spacing: 1px;
}

.status-pill {
  font-family: var(--cond);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 2px;
  padding: 5px 14px;
  border-radius: 2px;
  border: 1px solid;
  text-transform: uppercase;
}
.pill-idle     { color: var(--muted);  border-color: var(--muted); }
.pill-completed{ color: var(--green);  border-color: var(--green);
                 box-shadow: 0 0 12px rgba(0,255,136,0.2); }
.pill-preview  { color: var(--green);  border-color: var(--green);
                 box-shadow: 0 0 12px rgba(0,255,136,0.2); }
.pill-training { color: var(--purple); border-color: var(--purple);
                 box-shadow: 0 0 12px rgba(176,64,255,0.2); }
.pill-detecting{ color: var(--accent); border-color: var(--accent);
                 box-shadow: 0 0 12px rgba(0,229,255,0.2); animation: blink 1s infinite; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.5} }

/* ── STEP NAV ── */
.step-nav {
  display: flex;
  gap: 0;
  margin-bottom: 24px;
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}

.step-tab {
  flex: 1;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  border-right: 1px solid var(--border);
  transition: background .15s;
  background: var(--surface);
}
.step-tab:last-child { border-right: none; }

.step-num {
  font-family: var(--cond);
  font-size: 22px;
  font-weight: 900;
  color: var(--muted);
  line-height: 1;
}

.step-label {
  font-family: var(--cond);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted);
}

.step-tab.active {
  background: rgba(0,229,255,0.06);
  border-bottom: 2px solid var(--accent);
}
.step-tab.active .step-num  { color: var(--accent); }
.step-tab.active .step-label{ color: #fff; }

.step-tab.done {
  background: rgba(0,255,136,0.04);
}
.step-tab.done .step-num   { color: var(--green); }
.step-tab.done .step-label { color: var(--green); }

.step-check { font-size: 14px; margin-left: auto; }

/* ── MAIN GRID ── */
.main-grid {
  display: grid;
  grid-template-columns: 1fr 420px;
  gap: 16px;
  align-items: start;
}

/* ── PANELS ── */
.panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
}

.panel.camera-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  background: rgba(255,255,255,0.02);
}

.panel-title {
  font-family: var(--cond);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--muted);
}

.panel-body { padding: 16px; }

/* ── CAMERA FEED ── */
.cam-wrap {
  position: relative;
  background: #000;
  overflow: hidden;
  flex: 1;
  min-height: 0;
}

#cam {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: none;
}

.cam-placeholder {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  gap: 12px;
}

.cam-icon {
  font-size: 48px;
  opacity: .15;
}

.cam-placeholder p {
  font-family: var(--cond);
  font-size: 13px;
  letter-spacing: 2px;
  color: var(--muted);
  text-transform: uppercase;
}

/* Corner decorations */
.cam-corner {
  position: absolute;
  width: 20px; height: 20px;
  border-color: var(--accent);
  border-style: solid;
  opacity: .6;
}
.cam-corner.tl { top:8px; left:8px;  border-width:2px 0 0 2px; }
.cam-corner.tr { top:8px; right:8px; border-width:2px 2px 0 0; }
.cam-corner.bl { bottom:8px; left:8px;  border-width:0 0 2px 2px; }
.cam-corner.br { bottom:8px; right:8px; border-width:0 2px 2px 0; }

/* Rec indicator */
.cam-rec {
  position: absolute;
  top: 12px; left: 50%;
  transform: translateX(-50%);
  display: none;
  align-items: center;
  gap: 6px;
  background: rgba(0,0,0,.6);
  padding: 4px 10px;
  border-radius: 2px;
  font-family: var(--cond);
  font-size: 12px;
  letter-spacing: 2px;
  color: var(--accent2);
}
.cam-rec.show { display:flex; }
.rec-dot {
  width:7px; height:7px; border-radius:50%;
  background: var(--accent2);
  animation: blink .8s infinite;
}

/* ── STATS ROW ── */
.stats-row {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 1px;
  background: var(--border);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
  margin-top: 16px;
}

.stat-cell {
  background: var(--surface);
  padding: 12px 14px;
}

.stat-lbl {
  font-family: var(--cond);
  font-size: 10px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
}

.stat-val {
  font-family: var(--cond);
  font-size: 26px;
  font-weight: 700;
  line-height: 1;
  color: #fff;
}

/* ── RIGHT SIDEBAR ── */
.sidebar { display: flex; flex-direction: column; gap: 16px; }

/* ── CONTROL GROUPS ── */
.ctrl-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: flex-end;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.field label {
  font-family: var(--cond);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--muted);
}

.field input {
  background: var(--bg);
  border: 1px solid var(--border2);
  color: #fff;
  font-family: var(--mono);
  font-size: 14px;
  padding: 8px 10px;
  border-radius: 2px;
  width: 90px;
  outline: none;
  transition: border-color .15s, box-shadow .15s;
}
.field input[type=text] { width: 150px; }
.field input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(0,229,255,.1);
}

/* ── BUTTONS ── */
.btn {
  font-family: var(--cond);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  padding: 9px 18px;
  border: 1px solid;
  border-radius: 2px;
  cursor: pointer;
  transition: all .15s;
  background: transparent;
  white-space: nowrap;
}
.btn:disabled { opacity: .3; cursor: not-allowed; }
.btn:hover:not(:disabled) { filter: brightness(1.2); }
.btn:active:not(:disabled) { transform: scale(.97); }

.btn-preview  { color: var(--green);  border-color: var(--green);  }
.btn-preview:hover:not(:disabled)  { background: rgba(0,255,136,.1); }

.btn-stop     { color: var(--accent2); border-color: var(--accent2); }
.btn-stop:hover:not(:disabled)     { background: rgba(255,61,113,.1); }

.btn-train    { color: var(--purple); border-color: var(--purple); }
.btn-train:hover:not(:disabled)    { background: rgba(176,64,255,.1); }

.btn-detect   { color: var(--accent); border-color: var(--accent); }
.btn-detect:hover:not(:disabled)   { background: rgba(0,229,255,.1); }

.btn-export   { color: var(--amber);  border-color: var(--amber);  }
.btn-export:hover:not(:disabled)   { background: rgba(255,179,0,.1); }

.btn-sm {
  font-size: 11px;
  padding: 6px 12px;
  color: var(--muted);
  border-color: var(--border2);
}
.btn-sm:hover:not(:disabled) { color: var(--accent2); border-color: var(--accent2); background:transparent; }

/* ── HINT LINE ── */
.hint-line {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--accent);
  margin-top: 6px;
  padding-left: 2px;
}

/* ── PROGRESS ── */
.progress-block { display: none; margin-top: 12px; }

.progress-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 6px;
}
.progress-label span {
  font-family: var(--cond);
  font-size: 11px;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.progress-label .pl { color: var(--muted); }
.progress-label .pr { color: var(--purple); }

.progress-track {
  height: 4px;
  background: var(--border2);
  border-radius: 2px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--purple), var(--accent));
  border-radius: 2px;
  transition: width .3s ease;
  width: 0%;
}

#train-status {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--purple);
  margin-top: 8px;
  min-height: 16px;
  word-break: break-word;
}

/* ── PERSON CHIPS ── */
#train-persons {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
  min-height: 24px;
}

.p-chip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px 4px 8px;
  background: rgba(176,64,255,0.12);
  border: 1px solid rgba(176,64,255,0.3);
  border-radius: 2px;
  font-family: var(--cond);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: .5px;
  color: #ddd;
}

.p-chip-del {
  background: none; border: none; cursor: pointer;
  color: var(--muted); font-size: 13px; line-height: 1;
  padding: 0; transition: color .15s;
}
.p-chip-del:hover { color: var(--accent2); }

/* ── MODEL BADGE ── */
.model-badge {
  font-family: var(--cond);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  padding: 3px 10px;
  border-radius: 2px;
}
.badge-ready     { background: rgba(0,255,136,.15); color: var(--green); border:1px solid rgba(0,255,136,.3); }
.badge-notready  { background: rgba(176,64,255,.1);  color: var(--purple); border:1px solid rgba(176,64,255,.3); }

/* ── DETECTION ALERT ── */
#det-alert {
  display: none;
  background: rgba(0,229,255,.06);
  border: 1px solid rgba(0,229,255,.2);
  border-radius: 2px;
  padding: 10px 14px;
  margin-bottom: 12px;
}
.det-alert-top {
  font-family: var(--cond);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 1px;
  color: var(--accent);
}
.det-alert-sub {
  font-size: 11px;
  color: var(--muted);
  margin-top: 3px;
  font-style: italic;
}

/* ── LOG / TERMINAL ── */
.log-box {
  background: #050810;
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 10px 12px;
  font-family: var(--mono);
  font-size: 11px;
  line-height: 1.7;
  color: var(--muted);
  height: 90px;
  overflow-y: auto;
  margin-top: 16px;
}
.log-box .log-ok   { color: var(--green); }
.log-box .log-warn { color: var(--amber); }
.log-box .log-err  { color: var(--accent2); }

/* ── DIVIDER ── */
.hdivider {
  height: 1px;
  background: var(--border);
  margin: 14px 0;
}

/* ── BURST PROGRESS ── */
.burst-bar-wrap { margin-top: 10px; }
.burst-track {
  height: 3px;
  background: var(--border2);
  border-radius: 2px;
  overflow: hidden;
}
.burst-fill {
  height: 100%;
  background: var(--accent);
  border-radius: 2px;
  transition: width .3s;
  width: 0%;
}

</style>
</head>
<body>
<div class="app-shell">
<div class="wrap">

  <!-- HEADER -->
  <header>
    <div class="logo">
      <span class="logo-main">Classroom Monitor</span>
    </div>
    <div class="header-right">
      <span class="sys-time" id="sys-time">--:--:--</span>
      <span class="status-pill pill-completed" id="status-pill">Completed</span>
    </div>
  </header>

  <div class="layout">

    <div>
      <div class="panel camera-panel">
        <div class="panel-header">
          <span class="panel-title">Camera Feed</span>
        </div>
        <div class="cam-wrap">
          <img id="cam" alt="feed">
          <div class="cam-placeholder" id="cam-ph">
            <div class="cam-icon">⬛</div>
            <p>No Signal — Start Preview</p>
          </div>
          <div class="cam-corner tl"></div>
          <div class="cam-corner tr"></div>
          <div class="cam-corner bl"></div>
          <div class="cam-corner br"></div>
          <div class="cam-rec" id="cam-rec">
            <div class="rec-dot"></div>
            CAPTURING
          </div>
        </div>
      </div>
    </div>

    <div class="panel event-panel">
      <div class="panel-header">
        <div class="event-top" style="width:100%">
          <span class="event-title">Event Feed</span>
          <span class="event-actions">
            <span id="evt-count">0</span>
          </span>
        </div>
      </div>
      <div class="panel-body">
        <div class="event-tabs" id="evt-tabs">
          <span class="chip active" data-filter="all">All</span>
          <span class="chip" data-filter="people">People</span>
          <span class="chip" data-filter="classroom">Classroom</span>
          <span class="chip" data-filter="devices">Devices</span>
          <span class="chip" data-filter="movement">Movement</span>
        </div>
        <div class="event-list" id="evt-list">
          <div class="event-empty" id="evt-empty">No events yet. Start monitoring to see detections.</div>
        </div>
      </div>
    </div>
  </div><!-- /layout -->

</div><!-- /wrap -->
</div><!-- /app-shell -->

<div class="bottom-bar">
  <div class="bottom-inner">
    <button class="btn btn-preview" id="btn-start-preview" onclick="startPreview()">Start Preview</button>
    <button class="btn btn-stop" id="btn-end-preview" onclick="endPreview()">End Preview</button>
    <button class="btn btn-detect" id="btn-start-mon" onclick="startMonitoring()">Start Monitoring</button>
    <button class="btn btn-detect" onclick="startBehaviour()">Behaviour Monitoring</button>
    <button class="btn btn-stop" id="btn-end-mon" onclick="endMonitoring()">End Monitoring</button>
    <button class="btn btn-export" onclick="exportData()">Export Report</button>
  </div>
</div>

<script>
let prevBurst   = 0;
let burstTarget = 0;
const DET_DURATION_DEFAULT = 5;
const DET_NUM_IMAGES_DEFAULT = 10;
let evtFilter = 'all';
let evtCount = 0;

// Clock
setInterval(() => {
  document.getElementById('sys-time').textContent =
    new Date().toTimeString().slice(0,8);
}, 1000);

// Status pill
function setPill(mode) {
  const el = document.getElementById('status-pill');
  const map = {
    idle:      ['COMPLETED', 'pill-completed'],
    preview:   ['PREVIEW',   'pill-preview'],
    training:  ['TRAINING',  'pill-training'],
    detecting: ['DETECTING', 'pill-detecting'],
  };
  const [txt, cls] = map[mode] || ['COMPLETED','pill-completed'];
  el.textContent = txt;
  el.className   = 'status-pill ' + cls;
}

// Camera
function showStream() {
  document.getElementById('cam').src          = '/stream?' + Date.now();
  document.getElementById('cam').style.display = 'block';
  document.getElementById('cam-ph').style.display = 'none';
}
function hideStream() {
  document.getElementById('cam').style.display     = 'none';
  document.getElementById('cam-ph').style.display  = 'flex';
}

// Preview
function startPreview() {
  fetch('/start_preview', {method:'POST'});
  showStream();
}

function endPreview() {
  fetch('/stop', {method:'POST'});
  hideStream();
  document.getElementById('cam-rec').classList.remove('show');
}

function stopStream() {
  fetch('/stop',{method:'POST'});
  hideStream();
  document.getElementById('cam-rec').classList.remove('show');
}

function startMonitoring() {
  const d = DET_DURATION_DEFAULT;
  const n = DET_NUM_IMAGES_DEFAULT;
  document.getElementById('cam-rec').classList.add('show');
  prevBurst = 0;
  fetch('/start_detection',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({duration:d,num_images:n})});
  showStream();
}

function startBehaviour() {
  document.getElementById('cam-rec').classList.add('show');
  fetch('/start_behaviour', {method:'POST'});
  showStream();
}

function endMonitoring() {
  stopStream();
}

function addEvent(kind, text) {
  const list = document.getElementById('evt-list');
  const empty = document.getElementById('evt-empty');
  if (empty) empty.style.display = 'none';
  const ts = new Date().toTimeString().slice(0,8);
  const el = document.createElement('div');
  el.className = 'event-item';
  el.dataset.kind = kind;
  el.innerHTML = `<div class="k">${ts} // ${kind}</div><div class="v">${text}</div>`;
  list.prepend(el);
  evtCount += 1;
  document.getElementById('evt-count').textContent = String(evtCount);
  applyEventFilter();
}

function applyEventFilter() {
  const items = document.querySelectorAll('.event-item');
  items.forEach(it => {
    const k = it.dataset.kind || 'all';
    const show = (evtFilter === 'all') || (k === evtFilter);
    it.style.display = show ? 'block' : 'none';
  });
}

document.getElementById('evt-tabs').addEventListener('click', (e) => {
  const t = e.target;
  if (!t.classList.contains('chip')) return;
  evtFilter = t.dataset.filter || 'all';
  document.querySelectorAll('#evt-tabs .chip').forEach(c => c.classList.remove('active'));
  t.classList.add('active');
  applyEventFilter();
});

function exportData() { window.open('/export','_blank'); }

// Poll
setInterval(() => {
  fetch('/status').then(r=>r.json()).then(d => {

    setPill(d.mode);

    if (d.burst > 0 && prevBurst === 0)
      addEvent('classroom', 'Monitoring started');

    prevBurst = d.burst;

    if (d.mode === 'idle')
      document.getElementById('cam-rec').classList.remove('show');

    if (d.mode === 'behaviour') {
      if (d.behaviour_events && d.behaviour_events.length > lastBehaviourCount) {

        for (let i = lastBehaviourCount; i < d.behaviour_events.length; i++) {
          const evt = d.behaviour_events[i];
          addEvent('movement', evt.behaviours.join(', '));
        }

        lastBehaviourCount = d.behaviour_events.length;
      }
    } else {
      lastBehaviourCount = 0;
    }

  }).catch(()=>{});
}, 500);
</script>
</body>
</html>"""
    return html, 200, {"Content-Type": "text/html; charset=utf-8"}


# ──────────────────────────────────────────────────────────────────────────────
# Routes (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/stream")
def stream():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start_preview", methods=["POST"])
def api_preview():
    threading.Thread(target=start_preview, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/start_training", methods=["POST"])
def api_start_training():
    data       = request.get_json() or {}
    name       = data.get("name", "").strip()
    num_images = max(5, min(50, int(data.get("num_images", 15))))
    if not name:
        return jsonify({"ok": False, "error": "Name required"})
    if state["mode"] == "idle":
        threading.Thread(target=start_preview, daemon=True).start()
        time.sleep(1.5)
    threading.Thread(target=start_training_capture, args=(name, num_images), daemon=True).start()
    return jsonify({"ok": True, "name": name, "num_images": num_images})

@app.route("/start_detection", methods=["POST"])
def api_detect():
    data       = request.get_json() or {}
    duration   = max(1, min(120, float(data.get("duration",  5))))
    num_images = max(1, min(60,  int(data.get("num_images", 10))))
    threading.Thread(target=start_detection, args=(duration, num_images), daemon=True).start()
    return jsonify({"ok": True})
    
@app.route("/start_behaviour", methods=["POST"])
def api_behaviour():
    threading.Thread(target=start_behaviour_monitoring, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def api_stop():
    stop_stream()
    return jsonify({"ok": True})

@app.route("/faces")
def api_faces():
    return jsonify({"faces": [{"name": n} for n in face_db]})

@app.route("/delete_person", methods=["POST"])
def api_delete_person():
    name = (request.get_json() or {}).get("name", "").strip()
    if name in face_db:
        del face_db[name]
        save_face_db()
        state["model_ready"] = len(face_db) > 0
    return jsonify({"ok": True})

@app.route("/clear_faces", methods=["POST"])
def api_clear_faces():
    global face_db
    face_db = {}
    save_face_db()
    state["model_ready"] = False
    return jsonify({"ok": True})

@app.route("/status")
def api_status():
    return jsonify({
        "mode"           : state["mode"],
        "fps"            : state["fps"],
        "total"          : state["total_captures"],
        "burst"          : state["frames_captured_in_burst"],
        "num_images"     : state["num_images"],
        "ready"          : state["frame_ready"],
        "training"       : state["training"],
        "train_collected": state["train_collected"],
        "train_target"   : state["train_target"],
        "train_status"   : state["train_status"],
        "model_ready"    : state["model_ready"],
        "trained_persons": len(face_db),
        "behaviour_events": state.get("behaviour_events", []),
    })

@app.route("/export")
def api_export():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with CSV_LOCK:
        burst_images = list(state.get("burst_image_paths", []))
        burst_rows   = list(state.get("burst_csv_rows",   []))
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        csv_io = StringIO()
        w = csv.writer(csv_io)
        w.writerow(["Timestamp","Frame_Number","Objects_Detected","Image_File"])
        for row in burst_rows: w.writerow(row)
        z.writestr("detections.csv", csv_io.getvalue().encode("utf-8"))
        for img_path in sorted(burst_images):
            if os.path.exists(img_path):
                z.write(img_path, os.path.relpath(img_path, BASE_DIR))
        if os.path.exists(EMB_FILE):
            z.write(EMB_FILE, "face_embeddings.pkl")
    buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"detections_{ts}.zip",
                     mimetype="application/zip")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def shutdown(sig, frame):
    stop_stream(); sys.exit(0)

signal.signal(signal.SIGINT,  shutdown)
signal.signal(signal.SIGTERM, shutdown)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 YOLOv8 Hailo + InsightFace  //  VISION SYS")
    print("① Preview  — warm up camera")
    print("② Training — capture images → train → model locked")
    print("③ Detect   — YOLO + face ID, persons named in CSV")
    print("=" * 60)
    print(f"📁 {BASE_DIR}")
    print(f"🎓 {TRAIN_DIR}")
    print(f"👤 {EMB_FILE}")
    print("🌐 http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, threaded=True)
