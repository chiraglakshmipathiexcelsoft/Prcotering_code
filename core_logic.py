import re
import os
import cv2
import numpy as np
import threading
import pickle
import time
import csv
import zipfile
from io import BytesIO, StringIO
from datetime import datetime
from collections import defaultdict

# ── Paths ───────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.join(_SCRIPT_DIR, "detections")
TRAIN_DIR   = os.path.join(BASE_DIR, "training_images")
EMB_FILE    = os.path.join(BASE_DIR, "face_embeddings.pkl")

# ── Session State (set dynamically by start_session) ────────────────────────
SESSION_DIR = None   # e.g. detections/2026-02-22_17-15-00
CSV_FILE    = None   # e.g. detections/2026-02-22_17-15-00/detections.csv

os.makedirs(TRAIN_DIR, exist_ok=True)

# ── Visual Constants ────────────────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    (0,1, (220,180,100)), (0,2, (220,180,100)),
    (1,3, (200,160,80)), (2,4, (200,160,80)),
    (5,6, (60,180,255)), (5,11,(60,180,255)), (6,12,(60,180,255)),
    (11,12,(60,180,255)), (5,7, (80,230,80)), (7,9, (40,200,40)),
    (6,8, (80,230,80)), (8,10, (40,200,40)), (11,13,(200,80,200)),
    (13,15,(180,50,180)), (12,14,(200,80,200)), (14,16,(180,50,180))
]
KP_COLOURS = [
    (220,220,220),(220,180,100),(220,180,100),(200,160,80),(200,160,80),
    (60,180,255),(60,180,255),(80,230,80),(80,230,80),(40,200,40),(40,200,40),
    (60,180,255),(60,180,255),(200,80,200),(200,80,200),(180,50,180),(180,50,180),
]

# ── Shared State ────────────────────────────────────────────────────────────
face_db: dict = {}
person_behavior = defaultdict(lambda: {
    "prev_kp":None, "name": "Unknown",
    "head_turn_count":0, "head_turn_history":[],
    "peeking_count":0, "peek_history":[],
    "away_count":0, "away_start_time":None,
    "hand_count":0, "last_beep_time":0.0,
    "active_alerts":[],
    "last_logged_alerts": set(),
    "seat_coord": None,        # Store (x,y) of box center for seat swap tracking
    "last_phone_move_t": 0,    # Track time of hand moves near phones
    "usage_timer": 0,           # Accumulate "sustained" phone usage
    "suspicious_score": 0,     # Cumulative score for movement patterns
    "last_seen_t": 0,          # Last time seen in seat for abandonment tracking
    "hand_below_desk_count": 0, # Track under-desk activity
    "swap_flicker_count": 0,    # Prevent alert on ID flickering
})

# ── Global Monitor State ────────────────────────────────────────────────────
GLOBAL_MONITOR = {
    "invigilator_last_seen_t": time.time(),
    "door_activity_t": 0,
    "active_session": True
}
ZONES = {
    "AISLE": (200, 0, 1000, 720), # (x1, y1, x2, y2) Example region
    "DOOR":  (1100, 0, 1280, 400),
}
SUSPICIOUS_WEIGHTS = {
    "peek": 5, "hand": 10, "bag_reach": 20, "under_desk": 15, "turn": 5
}

# ── Global Synchronization State ──────────────────────────────────────────
COLLUSION_LOG = []        # List of (timestamp, person_id, action) for synchronization check
SEAT_REGISTRY = {}        # Map of "SeatID" (grid string) to "First Identified Name"
GLOBAL_SYNC_LOCK = threading.Lock()

alert_log      = []
BEHAVIOR_LOCK  = threading.Lock()
ALERT_LOG_LOCK = threading.Lock()
CSV_LOCK       = threading.Lock()

# ── Terminal Deduplication State ─────────────────────────────────────────────
_last_msg: str = ""
_msg_count: int = 0
_last_msg_t: float = 0
_term_lock = threading.Lock()

# ── Behavior Constants ──────────────────────────────────────────────────────
THRESH    = {"turn":5, "peek":3, "away":5, "hand":5}
BEEP_FREQ = {"turn":1000, "peek":800, "away":900, "hand":1100, "gesture":1200}
TRACKED_OBJECTS = [
    "cell phone", "laptop", "book", "backpack", "tablet", "electronic", "remote",
    "calculator", "smartwatch", "earphone", "headphone", "notebook", "paper"
]
FACE_THRESHOLD = 0.45

# ── Hailo Regex ─────────────────────────────────────────────────────────────
# Object: person [42] (0.85) (10, 20, 100, 200)
HAILO_RE = re.compile(r"Object:\s+([a-zA-Z][a-zA-Z0-9 _-]*?)\s*\[\d+\]\s+\(([\d.]+)\)(?:\s+\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\))?")

def log_system_event(tag: str, msg: str):
    """ Unified terminal logging format """
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] [{tag.upper():<7}] {msg}", flush=True)

# ── Initialization Functions ────────────────────────────────────────────────
def load_face_db():
    global face_db
    if os.path.exists(EMB_FILE):
        with open(EMB_FILE,"rb") as f: face_db = pickle.load(f)
        print(f"[INIT] Loaded {len(face_db)} person(s): {list(face_db.keys())}", flush=True)
    else: face_db = {}
    return face_db

def save_face_db():
    with open(EMB_FILE,"wb") as f: pickle.dump(face_db,f)

def init_csv():
    """ Initialize CSV inside the current session folder. """
    if CSV_FILE and not os.path.exists(CSV_FILE):
        os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
        with open(CSV_FILE, "w", newline="") as _f:
            csv.writer(_f).writerow(["Timestamp","Frame_Number","Objects_Detected","Behaviors","Image_File"])

def start_session():
    """
    Called when user clicks "Start Detection".
    Creates a timestamped session folder and initializes its CSV.
    Returns the session directory path.
    """
    global SESSION_DIR, CSV_FILE
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    SESSION_DIR = os.path.join(BASE_DIR, ts)
    os.makedirs(SESSION_DIR, exist_ok=True)
    CSV_FILE = os.path.join(SESSION_DIR, "detections.csv")
    init_csv()
    log_system_event("SESSION", f"New session started → {ts}")
    return SESSION_DIR

# ── Behavior Logic ──────────────────────────────────────────────────────────
def _log_alert(person_id: str, kind: str, count: int):
    global _last_msg, _msg_count, _last_msg_t
    with ALERT_LOG_LOCK:
        now_dt = datetime.now()
        now_str = now_dt.strftime("%H:%M:%S")
        
        # Human-Centric Sentence Mapping
        sub = person_id if person_id != "System" else "Someone"
        msg_map = {
            "Head Turns": f"{sub} is turning their head.",
            "Peeking Down": f"{sub} is peeking down.",
            "Looking Away": f"{sub} is looking away.",
            "Hand Signs": f"{sub} has raised their hand.",
            "Signaling": f"{sub} is signaling.",
        }
        
        # Handle side-specific hand signs
        kind_clean = kind.split(" (")[0]
        sentence = msg_map.get(kind_clean, f"Alert for {sub}: {kind} ({count})")
        
        tag = "BEHAVIOR"
        if "Confirmed" in kind:
            sentence = f"User identified as {kind.split(': ')[1].split(' (')[0]}"
            tag = "IDENTIFY"
        elif kind.startswith("Object: "):
            obj_name = kind.split(": ")[1]
            sentence = f"Prohibited object identified: {obj_name}"
            tag = "IDENTIFY"
            
        # 1. UI Aggregation (alert_log)
        if alert_log and alert_log[-1]["person"] == person_id and alert_log[-1]["type"] == kind:
            alert_log[-1]["repeat"] = alert_log[-1].get("repeat", 1) + 1
            alert_log[-1]["time"] = now_str
        else:
            alert_log.append({"time":now_str, "person":person_id, "type":kind, "count":count, "repeat":1})
            if len(alert_log)>50: alert_log.pop(0)

        # 2. Terminal Deduplication
        with _term_lock:
            now_t = time.time()
            if sentence == _last_msg and (now_t - _last_msg_t) < 5.0:
                _msg_count += 1
                # Only print every 10 repeats or if time gap is large to keep terminal moving 
                # but avoid the wall of text.
                if _msg_count % 10 == 0:
                    log_system_event(tag, f"{sentence} (x{_msg_count})")
            else:
                if _msg_count > 1 and _last_msg:
                    # Final count for the previous message if it repeated and we haven't printed the final one
                    if _msg_count % 10 != 0:
                        log_system_event(tag if "identified" not in _last_msg else "IDENTIFY", f"{_last_msg} (x{_msg_count})")
                
                log_system_event(tag, sentence)
                _last_msg = sentence
                _msg_count = 1
            _last_msg_t = now_t

def clear_behavior_state():
    """ Resets all behavior counters, alert logs, and terminal deduplication state """
    global _last_msg, _msg_count, _last_msg_t
    with BEHAVIOR_LOCK:
        person_behavior.clear()
    with GLOBAL_SYNC_LOCK:
        COLLUSION_LOG.clear()
        SEAT_REGISTRY.clear()
    with ALERT_LOG_LOCK:
        alert_log.clear()
    with _term_lock:
        _last_msg = ""
        _msg_count = 0
        _last_msg_t = 0
    log_system_event("SYSTEM", "Behavior counters and logs cleared.")

def check_global_rules(all_objs_with_boxes: list):
    """
    all_objs_with_boxes: list of (label, bbox_ltrb)
    """
    now = time.time()
    
    # 1. Invigilator Presence in Aisle
    invigilator_seen = False
    ax1, ay1, ax2, ay2 = ZONES["AISLE"]
    
    for label, box in all_objs_with_boxes:
        # Check if a person is in the Aisle
        if label.lower() in ["person", "invigilator"]:
            bx1, by1, bx2, by2 = box
            cx, cy = (bx1+bx2)/2, (by1+by2)/2
            if ax1 <= cx <= ax2 and ay1 <= cy <= ay2:
                invigilator_seen = True; break
    
    if invigilator_seen:
        GLOBAL_MONITOR["invigilator_last_seen_t"] = now
    else:
        absent_dur = now - GLOBAL_MONITOR["invigilator_last_seen_t"]
        if absent_dur > 120: # 2 minutes
            _log_alert("System", f"Invigilator Absent from Aisle for {int(absent_dur)}s", 1)
            GLOBAL_MONITOR["invigilator_last_seen_t"] = now # Log once per window

    # 2. Door Activity
    dx1, dy1, dx2, dy2 = ZONES["DOOR"]
    for label, box in all_objs_with_boxes:
        bx1, by1, bx2, by2 = box
        cx, cy = (bx1+bx2)/2, (by1+by2)/2
        if dx1 <= cx <= dx2 and dy1 <= cy <= dy2:
            if now - GLOBAL_MONITOR["door_activity_t"] > 30:
                _log_alert("System", f"Restricted Door Activity Detected", 1)
                GLOBAL_MONITOR["door_activity_t"] = now

    # 3. Candidate Left Seat
    with BEHAVIOR_LOCK:
        for person_id, data in person_behavior.items():
            if person_id == "System" or data["name"] == "Unknown": continue
            if data["last_seen_t"] > 0 and now - data["last_seen_t"] > 45: # 45 seconds
                if "Left Seat" not in data["last_logged_alerts"]:
                    _log_alert(person_id, "Left seat for more than 45s", 1)
                    data["last_logged_alerts"].add("Left Seat")
            else:
                data["last_logged_alerts"].discard("Left Seat")

def detect_behavior(person_id: str, curr_kp: np.ndarray, bbox: tuple = None, all_objs_with_boxes: list = None):
    data = person_behavior[person_id]
    data["last_seen_t"] = time.time()
    all_objs_with_boxes = all_objs_with_boxes or []
    
    # ── Seat Tracking & Swap Detection ──────────────────────────────────────
    if bbox is not None:
        # Calculate seat string (grid-based coordinate)
        bx, by, bw, bh = bbox
        cx, cy = bx + bw/2, by + bh/2
        grid_x, grid_y = int(cx // 100), int(cy // 100) # 100px grid
        seat_id = f"Seat_{grid_x}_{grid_y}"
        
        with GLOBAL_SYNC_LOCK:
            if seat_id not in SEAT_REGISTRY:
                # First time a KNOWN person is seen here, register them
                if "Unknown" not in person_id:
                    SEAT_REGISTRY[seat_id] = person_id
            else:
                original_owner = SEAT_REGISTRY[seat_id]
                # Alert only if a DIFFERENT KNOWN person is here consistently
                if "Unknown" not in person_id and person_id != original_owner:
                    data["swap_flicker_count"] += 1
                    if data["swap_flicker_count"] > 20: # ~2s buffer
                        if "Seat Swap" not in data["last_logged_alerts"]:
                            _log_alert(person_id, f"Seat Swap: {person_id} at {original_owner}'s desk", 1)
                            data["last_logged_alerts"].add("Seat Swap")
                elif person_id == original_owner:
                    data["swap_flicker_count"] = 0
                    data["last_logged_alerts"].discard("Seat Swap")
                else:
                    # If Unknown, we don't count it as a "Swap" yet to avoid false positives
                    data["swap_flicker_count"] = max(0, data["swap_flicker_count"] - 1)

    prev_kp = data["prev_kp"]
    data["prev_kp"] = curr_kp.copy()
    
    def kp_ok(i):
        if i >= len(curr_kp): return False
        if len(curr_kp[i]) > 2:
            return curr_kp[i][2] > 0.2
        return curr_kp[i][0] > 5 and curr_kp[i][1] > 5
    
    alerts = []; action = "Normal"
    nose=curr_kp[0]; lsho=curr_kp[5]; rsho=curr_kp[6]
    lwri=curr_kp[9]; rwri=curr_kp[10]

    # 1. HEAD TURNS (Improved Asymmetry + Ratio)
    if kp_ok(0) and kp_ok(5) and kp_ok(6):
        sho_dist = max(abs(rsho[0]-lsho[0]), 1)
        ctr = (lsho + rsho) / 2
        ratio = abs(nose[0] - ctr[0]) / sho_dist
        
        asymmetry = 0
        if kp_ok(1) and kp_ok(2):
            dist_l = abs(nose[0] - curr_kp[1][0])
            dist_r = abs(nose[0] - curr_kp[2][0])
            asymmetry = abs(dist_l - dist_r) / max(abs(curr_kp[2][0] - curr_kp[1][0]), 1)
            
        if ratio > 0.35 or asymmetry > 0.55 or (kp_ok(3) ^ kp_ok(4)):
            now = time.time()
            data["head_turn_history"].append(now)
            data["head_turn_history"] = [t for t in data["head_turn_history"] if now-t < 10]
            data["head_turn_count"] = len(data["head_turn_history"])
            
            # Collusion Logging
            with GLOBAL_SYNC_LOCK:
                COLLUSION_LOG.append((now, person_id, "Turn"))
                # Keep last 5 seconds of global turns
                while COLLUSION_LOG and now - COLLUSION_LOG[0][0] > 5.0:
                    COLLUSION_LOG.pop(0)
                
                # Check for synchronized turns
                recent_turns = [p for t, p, a in COLLUSION_LOG if p != person_id]
                if recent_turns:
                    if "Collusion" not in data["last_logged_alerts"]:
                        _log_alert(person_id, f"Possible Collusion: Synced turn with {recent_turns[0]}", 1)
                        data["last_logged_alerts"].add("Collusion")
            
            if data["head_turn_count"] >= THRESH["turn"]:
                alerts.append("Head Turns"); action = "Suspicious"
                if "Head Turns" not in data["last_logged_alerts"]:
                    _log_alert(person_id, "Head Turns", data["head_turn_count"])
                    data["last_logged_alerts"].add("Head Turns")
        else:
            data["last_logged_alerts"].discard("Head Turns")
            data["last_logged_alerts"].discard("Collusion")

    # 2. PEEKING DOWN (Nose must drop BELOW shoulder midpoint)
    if kp_ok(0) and kp_ok(5) and kp_ok(6):
        sho_y = (lsho[1]+rsho[1])/2
        if nose[1] > sho_y + 15:  # Nose must be 15px BELOW shoulder center
            now = time.time()
            data["peek_history"].append(now)
            data["peek_history"] = [t for t in data["peek_history"] if now-t < 10]
            data["peeking_count"] = len(data["peek_history"])
            if data["peeking_count"] >= THRESH["peek"]:
                alerts.append("Peeking Down"); action = "Cheating?"
                
                # Phone Usage Pattern (Peek + Hand Move)
                now = time.time()
                recent_hand = any(now - data.get(f"last_hand_time_{s}", 0) < 2.0 for s in ["Left", "Right"])
                if recent_hand:
                    data["usage_timer"] += 1
                    if data["usage_timer"] > 15: # Sustained pattern
                        if "Phone Usage" not in data["last_logged_alerts"]:
                            _log_alert(person_id, "Sustained Phone Usage Pattern", 1)
                            data["last_logged_alerts"].add("Phone Usage")
                
                if "Peeking Down" not in data["last_logged_alerts"]:
                    _log_alert(person_id, "Peeking Down", data["peeking_count"])
                    data["last_logged_alerts"].add("Peeking Down")
        else:
            data["last_logged_alerts"].discard("Peeking Down")
            data["last_logged_alerts"].discard("Phone Usage")
            data["usage_timer"] = max(0, data["usage_timer"] - 1)

    # 3. LOOKING AWAY
    if not kp_ok(0):
        if data["away_start_time"] is None: data["away_start_time"] = time.time()
        away_dur = time.time() - data["away_start_time"]
        data["away_count"] = int(away_dur)
        if away_dur >= THRESH["away"]:
            alerts.append("Looking Away"); action = "Not Focused"
            if "Looking Away" not in data["last_logged_alerts"]:
                _log_alert(person_id, "Looking Away", int(away_dur))
                data["last_logged_alerts"].add("Looking Away")
    else:
        data["away_start_time"] = None; data["away_count"] = 0
        data["last_logged_alerts"].discard("Looking Away")

    # 4. HAND SIGNALS (Both Hands)
    for wri_idx, sho_idx, side in [(9, 5, "Left"), (10, 6, "Right")]:
        if kp_ok(wri_idx) and kp_ok(sho_idx):
            wri_y = curr_kp[wri_idx][1]
            sho_y_pt = curr_kp[sho_idx][1]
            if wri_y < sho_y_pt - 15:
                now = time.time()
                last_key = f"last_hand_time_{side}"
                if now - data.get(last_key, 0) > 2.0:
                    data["hand_count"] += 1; data[last_key] = now
                    if data["hand_count"] >= THRESH["hand"]:
                        alerts.append("Hand Signs"); action = "Signaling"
                        if "Hand Signs" not in data["last_logged_alerts"]:
                            _log_alert(person_id, f"Hand Signs ({side})", data["hand_count"])
                            data["last_logged_alerts"].add("Hand Signs")
            else:
                data["last_logged_alerts"].discard("Hand Signs")
    
    # 5. Advanced Pattern: Reaching to Bag
    wrist_pts = [curr_kp[9], curr_kp[10]]
    for label, box in all_objs_with_boxes:
        if label.lower() in ["backpack", "bag", "handbag"]:
            bx1, by1, bx2, by2 = box
            for wx, wy in [(p[0], p[1]) for p in wrist_pts if kp_ok(9 if p is curr_kp[9] else 10)]:
                # If wrist is inside or very close to bag box
                if bx1-50 <= wx <= bx2+50 and by1-50 <= wy <= by2+50:
                    if "Bag Interaction" not in data["last_logged_alerts"]:
                        _log_alert(person_id, "Suspicious Interaction with Bag", 1)
                        data["suspicious_score"] += SUSPICIOUS_WEIGHTS["bag_reach"]
                        data["last_logged_alerts"].add("Bag Interaction")
        
    # 6. Under-Desk Activity
    if kp_ok(11) and kp_ok(12): # Hips
        hip_y = (curr_kp[11][1] + curr_kp[12][1]) / 2
        for i in [9, 10]: # Wrists
            if kp_ok(i) and curr_kp[i][1] > hip_y + 40:
                data["hand_below_desk_count"] += 1
                if data["hand_below_desk_count"] > 10:
                    if "Under Desk" not in data["last_logged_alerts"]:
                        _log_alert(person_id, "Sustained Under-Desk Activity", 1)
                        data["suspicious_score"] += SUSPICIOUS_WEIGHTS["under_desk"]
                        data["last_logged_alerts"].add("Under Desk")
    else:
        data["hand_below_desk_count"] = 0
        data["last_logged_alerts"].discard("Under Desk")

    # Final Score Alerting
    if data["suspicious_score"] >= 50:
        if "High Risk" not in data["last_logged_alerts"]:
            _log_alert(person_id, f"High Suspicion Score: {data['suspicious_score']}", 1)
            data["last_logged_alerts"].add("High Risk")

    data["active_alerts"] = alerts
    data["action"] = action
    return action, alerts

# ── Visualization ───────────────────────────────────────────────────────────
def draw_skeleton(img, kp, alert=False, label=""):
    color = (40,40,230) if alert else (80,200,80)
    def get_v(idx):
        if idx >= len(kp): return 0.0
        return kp[idx][2] if len(kp[idx]) > 2 else 1.0

    for i1, i2, c in SKELETON_CONNECTIONS:
        if get_v(i1) > 0.2 and get_v(i2) > 0.2:
            p1 = (int(kp[i1][0]), int(kp[i1][1]))
            p2 = (int(kp[i2][0]), int(kp[i2][1]))
            cv2.line(img, p1, p2, color, 2)
    for i in range(len(kp)):
        if get_v(i) > 0.2:
            clr = KP_COLOURS[i] if i < len(KP_COLOURS) else (200,200,200)
            cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 3, clr, -1)
    if label and get_v(0) > 0.2:
        cv2.putText(img, label, (int(kp[0][0]), int(kp[0][1]-15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_warning_banner(img, text):
    h,w = img.shape[:2]; overlay=img.copy()
    cv2.rectangle(overlay,(0,0),(w,36),(30,30,200),-1)
    cv2.addWeighted(overlay,0.55,img,0.45,0,img)
    cv2.putText(img,f"!  {text}",(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
    return img

# ── Training / DB Helpers ───────────────────────────────────────────────────
def collect_training_image(frame_bytes: bytes, name: str, index: int):
    person_dir=os.path.join(TRAIN_DIR,name); os.makedirs(person_dir,exist_ok=True)
    arr=np.frombuffer(frame_bytes,np.uint8)
    img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if img is None: return None
    path=os.path.join(person_dir,f"{name}_{index:03d}.jpg")
    cv2.imwrite(path,img); return path

def train_person(face_app, name: str) -> str:
    person_dir=os.path.join(TRAIN_DIR,name)
    if not os.path.exists(person_dir): return f"No training folder for '{name}'"
    images=[f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    if not images: return f"No images for '{name}'"
    embeddings,failed=[],0
    for img_file in images:
        img=cv2.imread(os.path.join(person_dir,img_file))
        if img is None: failed+=1; continue
        faces=face_app.get(img)
        if not faces: failed+=1; continue
        face=max(faces,key=lambda f:(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb=face.embedding/np.linalg.norm(face.embedding)
        embeddings.append(emb)
    if not embeddings: return f"No faces detected ({failed} failed)"
    avg=np.mean(embeddings,axis=0); avg=avg/np.linalg.norm(avg)
    face_db[name]=avg; save_face_db()
    msg=f"OK '{name}' trained -- {len(embeddings)} faces, {failed} skipped"
    print(f"[TRAIN] {msg}",flush=True); return msg

def identify_persons(face_app, frame, objs: dict, return_map: bool = False) -> any:
    if "person" not in objs or not face_db: return (objs, []) if return_map else objs
    if isinstance(frame, bytes):
        arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = frame
    if img is None: return (objs, []) if return_map else objs
    faces = face_app.get(img)
    if not faces: return (objs, []) if return_map else objs
    
    result = {k:v for k,v in objs.items() if k!="person"}
    face_map = []
    for face in faces:
        emb = face.embedding/np.linalg.norm(face.embedding)
        best_name, best_score = "Unknown", -1.0
        for nm, avg in face_db.items():
            sc = float(np.dot(emb, avg))
            if sc > best_score: best_score, best_name = sc, nm
        if best_score < FACE_THRESHOLD: best_name = "Unknown Person"
        
        score = round(best_score, 2)
        result[best_name] = score
        bbox = face.bbox.astype(int)
        face_map.append((best_name, bbox))
        
    return (result, face_map) if return_map else result

# ── CSV & Export Logic ──────────────────────────────────────────────────────
def save_frame(state, frame_bytes: bytes, frame_num: int) -> bool:
    if not frame_bytes or len(frame_bytes)<1000: return False
    if SESSION_DIR is None: start_session()  # Safety fallback
    
    os.makedirs(SESSION_DIR, exist_ok=True)
    ts_str=datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    img_name=f"IMG_{ts_str}_frame{frame_num:02d}.jpg"
    img_path=os.path.join(SESSION_DIR, img_name)
    try:
        with open(img_path,"wb") as f: f.write(frame_bytes)
    except Exception as e:
        print(f"[SAVE] Image write error: {e}",flush=True); return False
    
    state["burst_image_paths"].append(img_path)
    detected=dict(state["detected_objects"])
    timestamp=datetime.now().isoformat()
    objects_str=", ".join(f"{l}({c:.2f})" for l,c in
                          sorted(detected.items(),key=lambda x:-x[1])) or "none"
    behaviors_str="; ".join(state["current_frame_behaviors"]) if state["current_frame_behaviors"] else "none"
    
    if objects_str == "none" or behaviors_str == "none":
        print(f"[CSV DEBUG] Row {frame_num}: Objects='{objects_str}', Behaviors='{behaviors_str}'", flush=True)
    
    row=[timestamp,frame_num,objects_str,behaviors_str,img_name]
    with CSV_LOCK:
        try:
            with open(CSV_FILE,"a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow(row)
            state["burst_csv_rows"].append(row)
        except Exception as e:
            print(f"[CSV] {e}",flush=True)
    state["total_captures"]+=1
    return True

# ── UI Loading ──────────────────────────────────────────────────────────────
def get_ui_html(pose_available, pose_model_name, model_ready):
    tpl_path = os.path.join(_SCRIPT_DIR, "ui_template.html")
    if not os.path.exists(tpl_path):
        return "<h1>ui_template.html not found</h1>", 404
    
    with open(tpl_path, "r", encoding="utf-8") as f:
        html_tpl = f.read()

    pose_badge = (
        f'<span style="background:#edfaf3;color:#0f8a3c;border:1px solid #a3e8c0;'
        f'font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px;'
        f'letter-spacing:.5px;">POSE ON - {pose_model_name} · CUDA</span>'
        if pose_available else
        '<span style="background:#fff8ed;color:#c06a00;border:1px solid #fdd99a;'
        'font-size:9px;font-weight:700;padding:2px 8px;border-radius:10px;'
        'letter-spacing:.5px;">POSE OFF - add yolov8l-pose.pt</span>'
    )
    return html_tpl.replace("{{POSE_BADGE}}", pose_badge)
