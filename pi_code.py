#!/usr/bin/env python3
"""
Vision System - Raspberry Pi Edition
Hailo-10H Acceleration - Detection + Pose via dual_yolo_h10.json
"""

import os, re, csv, cv2, time, pickle, zipfile, signal, sys
import logging, threading, subprocess, numpy as np
from datetime import datetime
from io import BytesIO, StringIO
from collections import defaultdict
from flask import Flask, Response, jsonify, send_file, request

import insightface
from insightface.app import FaceAnalysis
import core_logic as core

# ── RPi specific alerts ───────────────────────────────────────────────────────
import tkinter as tk
from tkinter import messagebox

def show_popup(person_id: str, alert_type: str, count: int):
    def _run():
        try:
            root = tk.Tk(); root.withdraw()
            messagebox.showwarning("Suspicious Activity Alert",
                f"Person {person_id}: {count} {alert_type} detected!\nPossible cheating behaviour.")
            root.destroy()
        except Exception: pass
    threading.Thread(target=_run, daemon=True).start()

def beep(freq: int = 1000, ms: int = 200):
    try:
        subprocess.Popen(["beep", "-f", str(freq), "-l", str(ms)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception: pass

# ── Hailo Config ──────────────────────────────────────────────────────────────
# dual_yolo_h10.json runs BOTH hailo_yolo_inference + hailo_yolo_pose
# on Hailo-10H in a single rpicam-vid process. No ultralytics needed.
HAILO_DUAL_JSON   = "/home/sarasai/dual_yolo_h10.json"
HAILO_DETECT_JSON = "/usr/share/rpi-camera-assets/hailo_yolov8_inference.json"

POSE_AVAILABLE = os.path.exists(HAILO_DUAL_JSON)
_pose_name     = "Hailo-10H Detect+Pose" if POSE_AVAILABLE else "none"
POSE_EVERY_N   = 2

print(f"[INIT] Hailo Dual JSON : {'OK' if POSE_AVAILABLE else 'MISSING'} -> {HAILO_DUAL_JSON}")

TRACKED_OBJECTS = ["cell phone", "laptop", "book", "backpack", "tablet", "electronic"]

# Hailo stderr regexes
HAILO_OBJ_RE = re.compile(
    r"Object:\s+([a-zA-Z][a-zA-Z0-9 _-]*?)\s*\[\d+\]\s+\(([\d.]+)\)"
    r"(?:\s+\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\))?"
)
HAILO_KP_RE = re.compile(r"[Kk]eypoints?[:\s]+([\d.,\s]{30,})")

# InsightFace CPU
face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

core.init_csv()
face_db = core.load_face_db()

state = {
    "mode": "idle", "proc": None, "lock": threading.Lock(), "running": False,
    "latest_frame": None, "frame_ready": False, "frame_count": 0, "fps": 0.0,
    "current_block": [], "last_block": [],
    "detected_objects": {}, "all_det_boxes": [],
    "current_frame_behaviors": [],
    # FIX 1: detecting is now a simple bool, no num_images limit
    "detecting": False,
    "burst_image_paths": [], "burst_csv_rows": [],
    # FIX 1: total_saved tracks ALL images saved this session, shown in UI
    "total_saved": 0,
    "last_capture_time": 0.0, "capture_interval": 0.5,
    "last_error": "",
    "training": False, "train_name": "", "train_target": 10,
    "train_collected": 0, "train_last_capture": 0.0,
    "train_interval": 0.5, "train_status": "",
    "model_ready": len(face_db) > 0, "pose_enabled": POSE_AVAILABLE,
    "latest_raw": None,
    "persistent_draw_data": {"objs": [], "kps": []},
    "last_multi_entry_t": 0.0, "last_verbose_t": 0.0,
    "pose_frame_counter": 0,
}

# ── Detection block parser ────────────────────────────────────────────────────
def parse_last_block(block_lines: list) -> tuple:
    best: dict = {}; all_boxes: list = []
    for line in block_lines:
        m = HAILO_OBJ_RE.search(line)
        if m:
            label = m.group(1).strip().lower(); conf = float(m.group(2))
            if conf >= 0.1:
                if label not in best or conf > best[label]: best[label] = conf
                if m.group(3):
                    all_boxes.append((label, (int(m.group(3)), int(m.group(4)),
                                              int(m.group(5)), int(m.group(6)))))
    return best, all_boxes

# ── Pose block parser ─────────────────────────────────────────────────────────
def parse_pose_block(block_lines: list) -> list:
    persons = []; current_bbox = None; current_kps = None
    for line in block_lines:
        m_obj = HAILO_OBJ_RE.search(line)
        if m_obj and m_obj.group(1).strip().lower() == "person":
            if current_kps is not None: persons.append((current_bbox, current_kps))
            current_bbox = None; current_kps = None
            if m_obj.group(3):
                current_bbox = (int(m_obj.group(3)), int(m_obj.group(4)),
                                int(m_obj.group(5)), int(m_obj.group(6)))
        m_kp = HAILO_KP_RE.search(line)
        if m_kp:
            raw_str = m_kp.group(1).replace(",", " ")
            try: vals = [float(v) for v in raw_str.split() if v]
            except ValueError: continue
            if len(vals) >= 51:
                current_kps = np.array(vals[:51], dtype=np.float32).reshape(17, 3)
    if current_kps is not None: persons.append((current_bbox, current_kps))
    return persons

# ── Behavior analysis ─────────────────────────────────────────────────────────
def _run_behavior(persons: list):
    with state["lock"]:
        all_det_boxes = list(state.get("all_det_boxes", []))
        face_map = list(state["persistent_draw_data"]["objs"])

    core.check_global_rules(all_det_boxes)

    if len(persons) > 1:
        now_m = time.time()
        if now_m - state["last_multi_entry_t"] > 10.0:
            core._log_alert("System", f"Unauthorized Entry: {len(persons)} people in frame", 1)
            state["last_multi_entry_t"] = now_m

    new_kps = []; behaviors = []
    for j, (person_bbox, kp) in enumerate(persons):
        pid = f"P{j}"; name = "Unknown"; bbox_passed = person_bbox
        nose_x, nose_y = float(kp[0][0]), float(kp[0][1])
        for nm, bb in face_map:
            if bb[0] <= nose_x <= bb[2] and bb[1] <= nose_y <= bb[3]:
                name = nm; bbox_passed = bb; break
        with core.BEHAVIOR_LOCK:
            core.person_behavior[pid]["name"] = name
            log_name = name if name != "Unknown" else pid
            action, alerts = core.detect_behavior(log_name, kp,
                                                   bbox=bbox_passed,
                                                   all_objs_with_boxes=all_det_boxes)
            beh_desc = ", ".join(dict.fromkeys(alerts)) if alerts else action
            last_beep = core.person_behavior[pid].get("last_beep_time", 0)
        behaviors.append(f"{log_name} = {beh_desc}")
        new_kps.append((kp, len(alerts) > 0, f"{log_name}: {beh_desc}"))
        if alerts and (time.time() - last_beep > 2.0):
            beep(core.BEEP_FREQ.get("turn", 1000))
            with core.BEHAVIOR_LOCK:
                core.person_behavior[pid]["last_beep_time"] = time.time()

    with state["lock"]:
        state["current_frame_behaviors"] = behaviors
        state["persistent_draw_data"]["kps"] = new_kps

# ── Stderr dispatcher ─────────────────────────────────────────────────────────
def _dispatch_block(block_lines: list):
    has_keypoints = any("eypoints" in ln for ln in block_lines)
    if has_keypoints:
        state["pose_frame_counter"] += 1
        if (state["pose_frame_counter"] % POSE_EVERY_N == 0
                and state["mode"] in ("detecting", "preview")):
            persons = parse_pose_block(block_lines)
            if persons: _run_behavior(persons)
    else:
        hailo_objs, all_boxes = parse_last_block(block_lines)
        with state["lock"]:
            state["last_block"] = list(block_lines)
            state["detected_objects"] = hailo_objs
            state["all_det_boxes"] = all_boxes
        if state["detecting"]:
            for label, conf in hailo_objs.items():
                if any(p in label.lower() for p in TRACKED_OBJECTS):
                    key = f"last_alert_{label}"; now_a = time.time()
                    if now_a - state.get(key, 0) > 5.0:
                        core._log_alert("System", f"Detected: {label}", int(conf * 100))
                        state[key] = now_a
        now_v = time.time()
        if now_v - state["last_verbose_t"] > 4.0 and hailo_objs:
            core.log_system_event("DETECT",
                ", ".join(f"{k} ({int(v*100)}%)" for k, v in hailo_objs.items()))
            state["last_verbose_t"] = now_v

# ── Overlay renderer ──────────────────────────────────────────────────────────
def process_combined(frame: np.ndarray) -> np.ndarray:
    with state["lock"]:
        dd = state["persistent_draw_data"]
        for name, bbox in dd["objs"]:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (200, 0, 200), 2)
            cv2.putText(frame, name, (bbox[0], max(bbox[1]-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 1)
        for kp, is_alert, label in dd["kps"]:
            core.draw_skeleton(frame, kp, alert=is_alert, label=label)
    return frame

# ── Stderr monitor ────────────────────────────────────────────────────────────
def monitor_stderr(proc):
    in_block = False; current_block: list = []
    debug_start = time.time(); debug_done = False
    for raw in proc.stderr:
        if not state["running"]: break
        line = raw.decode(errors="ignore").strip()
        if not line: continue
        if not debug_done:
            if time.time() - debug_start < 60: print(f"[HAILO-RAW] {line}")
            else: debug_done = True; print("[HAILO] Debug window closed.")
        if "------" in line:
            if in_block and current_block: _dispatch_block(current_block)
            current_block = []; in_block = True
        elif in_block:
            current_block.append(line)

# ── Frame reader ──────────────────────────────────────────────────────────────
def read_frames(proc):
    buffer = b""; state["frame_count"] = 0; t0 = time.time()
    while state["running"]:
        try: chunk = proc.stdout.read(65536)
        except: break
        if not chunk: break
        buffer += chunk
        while True:
            s = buffer.find(b"\xff\xd8")
            if s == -1: buffer = b""; break
            e = buffer.find(b"\xff\xd9", s + 2)
            if e == -1: buffer = buffer[s:]; break
            raw = buffer[s:e+2]; buffer = buffer[e+2:]
            if len(raw) < 1000: continue

            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with state["lock"]:
                    hailo_objs = dict(state["detected_objects"])
                    all_boxes = list(state["all_det_boxes"])

                face_map = []
                if "person" in hailo_objs and state["model_ready"]:
                    hailo_objs, face_map = core.identify_persons(
                        face_app, frame, hailo_objs, return_map=True)

                with state["lock"]:
                    state["detected_objects"] = hailo_objs
                    state["all_det_boxes"] = all_boxes
                    state["persistent_draw_data"]["objs"] = face_map
                    state["latest_raw"] = frame

                for obj_name in core.TRACKED_OBJECTS:
                    if obj_name in hailo_objs:
                        now_o = time.time()
                        if now_o - state.get(f"last_obj_alert_{obj_name}", 0) > 60.0:
                            core._log_alert("System", f"Object: {obj_name}",
                                            int(hailo_objs[obj_name]*100))
                            state[f"last_obj_alert_{obj_name}"] = now_o

                annotated = process_combined(frame)
                ok, stream_buf = cv2.imencode(".jpg", annotated,
                                              [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    with state["lock"]:
                        state["latest_frame"] = stream_buf.tobytes()
                        state["frame_ready"] = True

                # FIX 1 & 2: Continuous capture — no num_images limit.
                # Captures every `capture_interval` seconds while detecting is True.
                # Stops only when user clicks Stop Detection.
                if state["detecting"]:
                    now = time.time()
                    if now - state["last_capture_time"] >= state["capture_interval"]:
                        frame_num = state["total_saved"] + 1
                        if core.save_frame(state, stream_buf.tobytes() if ok else raw,
                                           frame_num):
                            # FIX 1: Increment total_saved so UI counter updates
                            state["total_saved"] += 1
                            state["last_capture_time"] = now

                # Training
                if state["training"] and state["train_name"]:
                    now = time.time()
                    if (state["train_collected"] < state["train_target"] and
                            (now - state["train_last_capture"]) >= state["train_interval"]):
                        if core.collect_training_image(raw, state["train_name"],
                                                       state["train_collected"] + 1):
                            state["train_collected"] += 1
                            state["train_last_capture"] = now
                    if state["train_collected"] >= state["train_target"]:
                        state["training"] = False; state["train_status"] = "Training..."
                        def _do_train():
                            state["train_status"] = core.train_person(
                                face_app, state["train_name"])
                            state["mode"] = "preview"
                        threading.Thread(target=_do_train, daemon=True).start()

            state["frame_count"] += 1
            if state["frame_count"] % 30 == 0:
                el = time.time() - t0
                if el > 0: state["fps"] = round(state["frame_count"] / el, 1)

# ── Camera management ─────────────────────────────────────────────────────────
def stop_stream():
    state["running"] = False
    if state["proc"]:
        state["proc"].terminate()
        try: state["proc"].wait(timeout=2)
        except: state["proc"].kill()
    state.update({"proc": None, "mode": "idle",
                  "detecting": False, "training": False})

def _launch(mode: str):
    stop_stream()
    state["running"] = True; state["mode"] = mode
    post_json = HAILO_DUAL_JSON if POSE_AVAILABLE else HAILO_DETECT_JSON
    cmd = [
        "rpicam-vid", "--post-process-file", post_json,
        "--codec", "mjpeg", "--output", "-", "--timeout", "0",
        "--width", "1280", "--height", "720",
        "--rotation", "180", "--framerate", "30",
        "--nopreview", "-n", "--verbose", "2",
    ]
    state["proc"] = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    threading.Thread(target=read_frames,    args=(state["proc"],), daemon=True).start()
    threading.Thread(target=monitor_stderr, args=(state["proc"],), daemon=True).start()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)

import logging as _logging
_wlog = _logging.getLogger("werkzeug")
class PollingFilter(_logging.Filter):
    def filter(self, record):
        return not any(p in record.getMessage()
                       for p in ["/status", "/behavior_status", "/faces"])
_wlog.addFilter(PollingFilter())

@app.before_request
def log_request():
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
            if frame: yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.05)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def api_status():
    # FIX 1: Return total_saved so UI counter increments correctly
    return jsonify({
        "mode":            state["mode"],
        "fps":             state["fps"],
        "inference_fps":   0.0,
        "burst":           state["total_saved"],
        "num_images":      0,
        "total":           state["total_saved"],
        "total_saved":     state["total_saved"],     # ← live capture count
        "detecting":       state["detecting"],        # ← for UI button toggle
        "train_status":    state["train_status"],
        "train_collected": state["train_collected"],
        "model_ready":     state["model_ready"],
        "trained_persons": len(core.face_db),
        "objects":         state["detected_objects"],
    })

@app.route("/behavior_status")
def api_behavior_status():
    with core.BEHAVIOR_LOCK:
        max_turns = max_peeks = max_away = max_hands = person_count = 0
        all_active = []
        for pid, data in core.person_behavior.items():
            person_count += 1
            max_turns = max(max_turns, data.get("head_turn_count", 0))
            max_peeks = max(max_peeks, data.get("peeking_count", 0))
            max_away  = max(max_away,  data.get("away_count", 0))
            max_hands = max(max_hands, data.get("hand_count", 0))
            all_active.extend(data.get("active_alerts", []))
    with core.ALERT_LOG_LOCK:
        logs = list(core.alert_log)
    return jsonify({
        "person_count": person_count,
        "max_turns": max_turns, "max_peeks": max_peeks,
        "max_away": max_away,   "max_hands": max_hands,
        "active_alerts": list(dict.fromkeys(all_active)),
        "alert_log": logs,
    })

@app.route("/clear_behavior", methods=["POST"])
def api_clear_behavior():
    core.clear_behavior_state()
    return jsonify({"ok": True})

@app.route("/start_preview", methods=["POST"])
def api_preview():
    _launch("preview")
    return jsonify({"ok": True})

@app.route("/start_training", methods=["POST"])
def api_train():
    data = request.get_json() or {}
    state.update({
        "training": True, "train_name": data.get("name", "Admin"),
        "train_target": int(data.get("num_images", 15)),
        "train_collected": 0, "train_last_capture": 0.0, "train_status": "",
    })
    if state["mode"] == "idle": _launch("preview")
    return jsonify({"ok": True})

# FIX 2: Start Detection — no num_images param, captures continuously
@app.route("/start_detection", methods=["POST"])
def api_detect():
    core.start_session()
    state.update({
        "detecting": True,
        "total_saved": 0,          # reset counter for new session
        "last_capture_time": 0.0,
        "capture_interval": 0.5,   # capture every 0.5s continuously
        "burst_image_paths": [],
        "burst_csv_rows": [],
    })
    _launch("detecting")
    return jsonify({"ok": True})

# FIX 2: Stop Detection — stops capturing but keeps camera in preview
@app.route("/stop_detection", methods=["POST"])
def api_stop_detection():
    state["detecting"] = False
    state["mode"] = "preview"
    core.log_system_event("DETECT", f"Detection stopped. {state['total_saved']} images saved.")
    return jsonify({"ok": True, "total_saved": state["total_saved"], "total": state["total_saved"]})

@app.route("/stop", methods=["POST"])
def api_stop():
    stop_stream(); return jsonify({"ok": True})

@app.route("/faces")
def api_faces():
    return jsonify({"faces": [{"name": n} for n in core.face_db]})

@app.route("/delete_person", methods=["POST"])
def api_delete_person():
    name = (request.get_json() or {}).get("name")
    if name in core.face_db: del core.face_db[name]; core.save_face_db()
    return jsonify({"ok": True})

@app.route("/clear_faces", methods=["POST"])
def api_clear():
    core.face_db = {}; core.save_face_db(); return jsonify({"ok": True})

@app.route("/export")
def api_export():
    if not state.get("burst_csv_rows"): return "No data", 404
    bio = BytesIO()
    session_name = os.path.basename(core.SESSION_DIR) if core.SESSION_DIR else "results"
    with zipfile.ZipFile(bio, 'w') as zf:
        out = StringIO(); cw = csv.writer(out)
        cw.writerow(["Timestamp", "Frame", "Objects", "Behaviors", "File"])
        cw.writerows(state["burst_csv_rows"])
        zf.writestr(f"{session_name}/report.csv", out.getvalue())
        for p in state.get("burst_image_paths", []):
            if os.path.exists(p): zf.write(p, f"{session_name}/images/{os.path.basename(p)}")
    bio.seek(0)
    return send_file(bio, mimetype="application/zip",
                     as_attachment=True, download_name=f"{session_name}.zip")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=True)
