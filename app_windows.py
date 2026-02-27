#!/usr/bin/env python3
"""
YOLOv8 Detection + InsightFace Training + YOLOv8-Pose Behavior Monitor
Windows GPU Edition  —  NVIDIA CUDA (RTX PRO 1000 / Blackwell)
UI : Professional Light Theme — Full-Screen 3-Column Layout 
Entry Point: Launches Camera acquisition + AI Inference threads.
"""

import os, re, csv, cv2, time, pickle, zipfile, signal, sys
import logging, threading, numpy as np
from io import BytesIO, StringIO
from datetime import datetime
import torch
import insightface
from insightface.app import FaceAnalysis
from ultralytics import YOLO
from flask import Flask, Response, jsonify, send_file, request

# Import Core Logic
import core_logic as core

# ── Windows specific alerts ──────────────────────────────────────────────────
import winsound, tkinter as tk
from tkinter import messagebox

def beep(freq: int = 1000, ms: int = 200):
    try: winsound.Beep(freq, ms)
    except Exception: pass

def show_popup(person_id: str, alert_type: str, count: int):
    def _run():
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showwarning(
                "Suspicious Activity Alert",
                f"Person {person_id}: {count} {alert_type} detected!\n"
                "Possible cheating behaviour.")
            root.destroy()
        except Exception: pass
    threading.Thread(target=_run, daemon=True).start()

# ── GPU / Model Initialization ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Device      : {DEVICE}")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DET_MODEL_PATH  = os.path.join(_SCRIPT_DIR, "yolov8l.pt") # YOLOv8 Large
# DET_MODEL_PATH  = os.path.join(_SCRIPT_DIR, "yolo26s.pt")  # YOLO26 Small
#DET_MODEL_PATH  = os.path.join(_SCRIPT_DIR, "yolo26l.pt")  # YOLO26 Large

POSE_MODEL_PATH = os.path.join(_SCRIPT_DIR, "yolov8l-pose.pt") # YOLOv8 Large Pose
#POSE_MODEL_PATH = os.path.join(_SCRIPT_DIR, "yolo26l-pose.pt") # YOLO26 Large Pose

# Fallback for Pose if L is missing
if not os.path.exists(POSE_MODEL_PATH):
    POSE_MODEL_PATH = os.path.join(_SCRIPT_DIR, "yolov8l-pose.pt")

try:
    det_model  = YOLO(DET_MODEL_PATH).to(DEVICE)
    DET_AVAILABLE = True
except Exception as e:
    det_model = None; DET_AVAILABLE = False
    print(f"[WARN] Det model missing: {e}")

try:
    pose_model = YOLO(POSE_MODEL_PATH).to(DEVICE)
    POSE_AVAILABLE = True
    _pose_name = os.path.basename(POSE_MODEL_PATH)
except Exception as e:
    pose_model = None; POSE_AVAILABLE = False
    _pose_name = "none"
    print(f"[WARN] Pose model missing: {e}")

# High performance settings
DET_EVERY_N  = 2 if "yolov8l" in os.path.basename(DET_MODEL_PATH or "") else 1
POSE_EVERY_N = 2 if "yolov8l" in _pose_name else 1

# InsightFace
try:
    face_app = FaceAnalysis(name="buffalo_sc", providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(320,320))
    print("[INIT] InsightFace : CUDA")
except:
    face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(320,320))
    print("[INIT] InsightFace : CPU fallback")

# ── Shared Intelligence ─────────────────────────────────────────────────────
core.init_csv()
face_db = core.load_face_db()

# Application State (Merged with Core)
state = {
    "mode":"idle", "lock":threading.Lock(), "running":False,
    "latest_frame":None, "latest_raw":None, "frame_ready":False,
    "frame_count":0, "fps":0.0, "inference_fps":0.0,
    "detected_objects":{}, "current_frame_behaviors":[], "detecting":False,
    "duration":5, "num_images":10, "capture_interval":0.5,
    "frames_captured_in_burst":0, "last_capture_time":0.0,
    "burst_image_paths":[], "burst_csv_rows":[],
    "total_captures":0, "last_error":"",
    "training":False, "train_name":"", "train_target":10,
    "train_collected":0, "train_last_capture":0.0, "train_interval":0.5, "train_status":"",
    "model_ready":len(face_db)>0, "pose_enabled":POSE_AVAILABLE,
    "persistent_draw_data": {"objs": [], "kps": []}
}

# ── AI Processing ───────────────────────────────────────────────────────────
_det_frame_counter  = -1
_pose_frame_counter = -1
TRACKED_OBJECTS = core.TRACKED_OBJECTS

def process_frame(img: np.ndarray) -> np.ndarray:
    global _det_frame_counter, _pose_frame_counter
    
    if state["mode"] not in ["detecting", "training"]:
        with state["lock"]:
            state["detected_objects"] = {}
            state["current_frame_behaviors"] = []
            state["persistent_draw_data"] = {"objs": [], "kps": []}
        return img

    _det_frame_counter += 1
    _pose_frame_counter += 1
    face_map = []
    
    # 1. Detection (Periodic)
    if state["mode"] == "detecting" and DET_AVAILABLE and _det_frame_counter % DET_EVERY_N == 0:
        try:
            results = det_model(img, verbose=False, conf=0.3)
            objs = {}; all_det_boxes = []
            for box in results[0].boxes:
                label = det_model.names[int(box.cls[0])]
                conf  = float(box.conf[0])
                bbox_ltrb = box.xyxy[0].cpu().numpy().astype(int)
                all_det_boxes.append((label, bbox_ltrb))
                if label not in objs or conf > objs[label]: objs[label] = conf
            
            # Global Zone Rules (Aisle, Door, etc.)
            core.check_global_rules(all_det_boxes)
            
            # Person Identification
            if "person" in objs and state["model_ready"]:
                faces = face_app.get(img)
                person_conf = objs.pop("person")
                found_face = False
                for face in faces:
                    emb = face.embedding/np.linalg.norm(face.embedding)
                    best_name, best_score = "Unknown", -1.0
                    for nm, avg in face_db.items():
                        sc = float(np.dot(emb, avg))
                        if sc > best_score: best_score, best_name = sc, nm
                    if best_score < core.FACE_THRESHOLD: best_name = "Unknown Person"
                    
                    bbox = face.bbox.astype(int)
                    face_map.append((best_name, bbox))
                    objs[best_name] = best_score if best_name != "Unknown Person" else person_conf
                    found_face = True
                    
                    # Log Identification (Throttled)
                    if best_name != "Unknown Person":
                        now_id = time.time()
                        if now_id - state.get(f"last_id_t_{best_name}", 0) > 60.0:
                            core._log_alert("System", f"Confirmed: {best_name}", int(best_score*100))
                            state[f"last_id_t_{best_name}"] = now_id

                if not found_face: objs["person"] = person_conf

            # Object Identification Alerts (Throttled)
            for obj_name in core.TRACKED_OBJECTS:
                if obj_name in objs:
                    conf = objs[obj_name]
                    now_obj = time.time()
                    if now_obj - state.get(f"last_obj_alert_{obj_name}", 0) > 60.0:
                        core._log_alert("System", f"Object: {obj_name}", int(conf*100))
                        state[f"last_obj_alert_{obj_name}"] = now_obj

            with state["lock"]:
                state["detected_objects"] = objs
                state["persistent_draw_data"]["objs"] = face_map.copy()
            
            # Periodic Verbose Logging
            now_v = time.time()
            if now_v - state.get("last_verbose_t", 0) > 4.0:
                summary = [f"{k} ({int(v*100) if v <= 1.0 else int(v)}%)" for k, v in objs.items()]
                if summary: core.log_system_event("DETECT", ", ".join(summary))
                state["last_verbose_t"] = now_v
        except Exception as e: print(f"[DET] {e}")

    # 2. Pose & Behavior (Periodic)
    if POSE_AVAILABLE and _pose_frame_counter % POSE_EVERY_N == 0:
        try:
            p_results = pose_model(img, verbose=False, conf=0.25)
            global_alert = False; summary = []
            new_kps = []
            with state["lock"]: state["current_frame_behaviors"] = []
            
            if p_results and p_results[0].keypoints is not None:
                for j, kp_tensor in enumerate(p_results[0].keypoints.data):
                    kp = kp_tensor.cpu().numpy(); pid = f"P{j}"
                    nose = kp[0]
                    name = "Unknown"
                    bbox_passed = None
                    with state["lock"]:
                        for nm, bb in state["persistent_draw_data"]["objs"]:
                            if bb[0]<=nose[0]<=bb[2] and bb[1]<=nose[1]<=bb[3]:
                                name=nm; bbox_passed=bb; break
                    
                    with core.BEHAVIOR_LOCK:
                        core.person_behavior[pid]["name"] = name
                        log_name = name if name != "Unknown" else pid
                        action, alerts = core.detect_behavior(log_name, kp, bbox=bbox_passed, all_objs_with_boxes=all_det_boxes)
                        beh_desc = ", ".join(dict.fromkeys(alerts)) if alerts else action
                        
                        with state["lock"]:
                            state["current_frame_behaviors"].append(f"{log_name} = {beh_desc}")
                            new_kps.append((kp, len(alerts)>0, f"{log_name}: {beh_desc}"))
                        
                        if alerts:
                            global_alert = True; summary.extend(alerts)
                            if (time.time() - core.person_behavior[pid].get("last_beep_time",0)) > 2.0:
                                beep(core.BEEP_FREQ.get("turn",1000))
                                core.person_behavior[pid]["last_beep_time"] = time.time()
                
                # Multi-Person / Unauthorized Entry (Global Check)
                person_count = len(p_results[0].keypoints.data)
                if person_count > 1:
                    now_multi = time.time()
                    if now_multi - state.get("last_multi_entry_t", 0) > 10.0:
                        core._log_alert("System", f"Unauthorized Entry: {person_count} people in frame", 1)
                        state["last_multi_entry_t"] = now_multi
                        global_alert = True
            
            with state["lock"]: state["persistent_draw_data"]["kps"] = new_kps
            if global_alert: core.draw_warning_banner(img, f"ALERT: {', '.join(list(dict.fromkeys(summary))[:3])}")
        except Exception as e: print(f"[POSE] {e}")

    # 3. Persistent Overlay Rendering (Every Frame)
    with state["lock"]:
        draw_data = state["persistent_draw_data"]
        for name, bbox in draw_data["objs"]:
            cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(200,0,200),2)
            cv2.putText(img, name, (bbox[0],max(bbox[1]-5,10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,0,200), 1)
        for kp, is_alert, label in draw_data["kps"]:
            core.draw_skeleton(img, kp, alert=is_alert, label=label)

    return img

# ── Threads ─────────────────────────────────────────────────────────────────
def camera_loop():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    t0 = time.time(); fc = 0
    while state["running"]:
        ret, frame = cap.read()
        if not ret: time.sleep(0.01); continue
        with state["lock"]: state["latest_raw"] = frame
        if state["mode"] not in ["detecting", "training"]:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with state["lock"]: state["latest_frame"] = buf.tobytes(); state["frame_ready"] = True
        fc += 1; state["frame_count"] += 1
        if fc % 30 == 0:
            el = time.time() - t0
            if el > 0: state["fps"] = round(fc/el, 1)
    cap.release()

def inference_loop():
    t0 = time.time(); ifc = 0
    while state["running"]:
        with state["lock"]: raw = state["latest_raw"].copy() if state["latest_raw"] is not None else None
        if raw is None or state["mode"] not in ["detecting", "training"]:
            time.sleep(0.02); continue
        
        annotated = process_frame(raw)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            with state["lock"]: state["latest_frame"] = buf.tobytes(); state["frame_ready"] = True
        
        # Training
        if state["training"] and state["train_name"]:
            now = time.time()
            if state["train_collected"] < state["train_target"] and (now - state["train_last_capture"]) >= state["train_interval"]:
                ok2, rb = cv2.imencode(".jpg", raw)
                if core.collect_training_image(rb.tobytes() if ok2 else b"", state["train_name"], state["train_collected"]+1):
                    state["train_collected"] += 1; state["train_last_capture"] = now
            if state["train_collected"] >= state["train_target"]:
                state["training"] = False; state["train_status"] = "Training..."
                def _do_train():
                    state["train_status"] = core.train_person(face_app, state["train_name"])
                    state["mode"] = "preview"; state["model_ready"] = True
                threading.Thread(target=_do_train, daemon=True).start()

        if state["detecting"]:
            now = time.time()
            if (now - state["last_capture_time"]) >= state["capture_interval"]:
                ok3, rb3 = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 93])
                next_frame = state["frames_captured_in_burst"] + 1
                if core.save_frame(state, rb3.tobytes(), next_frame):
                    state["frames_captured_in_burst"] = next_frame
                    state["total_captures"] += 1
                    state["last_capture_time"] = now

        ifc += 1
        if ifc % 20 == 0:
            el = time.time() - t0
            if el > 0: state["inference_fps"] = round(ifc/el, 1)

# ── Flask App ───────────────────────────────────────────────────────────────
app = Flask(__name__)

# Suppress Werkzeug logs for polling routes
import logging
log = logging.getLogger('werkzeug')
class PollingFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not any(p in msg for p in ["/status", "/behavior_status", "/faces"])
log.addFilter(PollingFilter())

@app.before_request
def log_request():
    # Only log meaningful hits
    if request.path not in ["/stream", "/status", "/behavior_status", "/faces"]:
        core.log_system_event("API", f"{request.method} {request.path}")

@app.route("/")
def index():
    return core.get_ui_html(POSE_AVAILABLE, _pose_name, state["model_ready"])

@app.route("/stream")
def stream():
    def gen():
        while True:
            with state["lock"]: frame = state["latest_frame"]
            if frame: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+frame+b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def api_status():
    return jsonify({
        "mode": state["mode"], "fps": state["fps"], "inference_fps": state["inference_fps"],
        "burst": state["frames_captured_in_burst"], "num_images": state["num_images"],
        "total": state["total_captures"], "train_status": state["train_status"], 
        "model_ready": state["model_ready"], "trained_persons": len(core.face_db)
    })

@app.route("/behavior_status")
def api_behavior_status():
    with core.BEHAVIOR_LOCK:
        max_turns = 0; max_peeks = 0; max_away = 0; max_hands = 0; person_count = 0
        all_active = []
        for pid, data in core.person_behavior.items():
            person_count += 1
            max_turns = max(max_turns, data.get("head_turn_count", 0))
            max_peeks = max(max_peeks, data.get("peeking_count", 0))
            max_away  = max(max_away, data.get("away_count", 0))
            max_hands = max(max_hands, data.get("hand_count", 0))
            all_active.extend(data.get("active_alerts", []))
            
    with core.ALERT_LOG_LOCK:
        logs = list(core.alert_log)
        
    return jsonify({
        "person_count": person_count,
        "max_turns": max_turns,
        "max_peeks": max_peeks,
        "max_away": max_away,
        "max_hands": max_hands,
        "active_alerts": list(dict.fromkeys(all_active)),
        "alert_log": logs
    })

@app.route("/start_preview", methods=["POST"])
def api_start_preview():
    if not state["running"]:
        state["running"] = True; state["mode"] = "preview"
        threading.Thread(target=camera_loop, daemon=True).start()
        threading.Thread(target=inference_loop, daemon=True).start()
    return jsonify({"ok":True})

@app.route("/start_training", methods=["POST"])
def api_train():
    data = request.get_json() or {}
    state.update({"mode":"training","training":True,"train_name":data.get("name","Admin"),
                  "train_target":int(data.get("num_images",15)),"train_collected":0,"train_last_capture":0})
    return jsonify({"ok":True})

@app.route("/start_detection", methods=["POST"])
def api_detect():
    # Create a new session folder for this detection run
    core.start_session()
    state.update({"mode":"detecting","detecting":True,"num_images":0,
                  "capture_interval":0.5,
                  "frames_captured_in_burst":0,"last_capture_time":0,
                  "total_captures":0,
                  "burst_image_paths":[],"burst_csv_rows":[]})
    return jsonify({"ok":True})

@app.route("/stop_detection", methods=["POST"])
def api_stop_detection():
    state["detecting"] = False
    state["mode"] = "preview"
    return jsonify({"ok": True, "total_saved": state.get("total_captures", 0)})

@app.route("/stop", methods=["POST"])
def api_stop():
    state["running"] = False; state["mode"] = "idle"; return jsonify({"ok":True})

@app.route("/faces")
def api_faces(): return jsonify({"faces":[{"name":n} for n in core.face_db]})

@app.route("/delete_person", methods=["POST"])
def api_delete():
    name = (request.get_json() or {}).get("name")
    if name in core.face_db: del core.face_db[name]; core.save_face_db()
    return jsonify({"ok":True})

@app.route("/clear_behavior", methods=["POST"])
def api_clear_behavior():
    core.clear_behavior_state()
    return jsonify({"ok":True})

@app.route("/export")
def api_export():
    if not state["burst_csv_rows"]: return "No data", 404
    bio = BytesIO()
    session_name = os.path.basename(core.SESSION_DIR) if core.SESSION_DIR else "results"
    with zipfile.ZipFile(bio, 'w') as zf:
        # CSV
        out = StringIO()
        cw = csv.writer(out)
        cw.writerow(["Timestamp","Frame","Objects","Behaviors","File"])
        cw.writerows(state["burst_csv_rows"])
        zf.writestr(f"{session_name}/report.csv", out.getvalue())
        # Images
        for p in state["burst_image_paths"]:
            if os.path.exists(p): zf.write(p, f"{session_name}/images/{os.path.basename(p)}")
    bio.seek(0)
    return send_file(bio, mimetype="application/zip", as_attachment=True, download_name=f"{session_name}.zip")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)
