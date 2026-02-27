# Technical Logic Guide: Advanced Proctoring AI

This document details the mathematical and algorithmic foundations of the proctoring system to ensure clarity for developers and maintainers.

## 1. System Architecture
The system follows a **Core-Engine pattern**:
- **`core_logic.py`**: The "Brain". Contains all mathematical formulas, behavior states, and cross-platform logic.
- **`app_windows.py`**: Implementation for Windows (YOLO + FaceAnalysis).
- **`pi_code.py`**: Implementation for Raspberry Pi (Hailo-10H + CPU Pose).

---

## 2. Behavioral Detection Logic

### A. Head Turns (Asymmetry & Ratio)
Detects side-to-side head movements by analyzing facial landmarks relative to shoulders.
- **Ratio Formula**: `|Nose.x - Shoulder_Center.x| / Shoulder_Width`
  - *Trigger*: `ratio > 0.35`
- **Asymmetry Formula**: `|Dist_Left - Dist_Right| / Eye_to_Eye_Width`
  - *Calculation*: Measures if the nose is significantly closer to one eye than the other from a 2D perspective.
  - *Trigger*: `asymmetry > 0.55`

### B. Peeking Down
Uses vertical shoulder positioning as a dynamic baseline to handle varying camera heights.
- **Logic**: `Nose.y > ((L_Shoulder.y + R_Shoulder.y) / 2) - 12`
- **Mathematics**: The `-12` acts as a sensitivity offset to allow natural desk work while flagging sustained downward gaze.

### C. Hand Signs
- **Logic**: Detected when either `Wrist.y < Shoulder.y - 15`.
- **Threshold**: Must occur `THRESH["hand"]` times to trigger an alert.

---

## 3. High-Level Cheating Detection

### A. Seat Swap (Coordinate Registry)
- **Grid Conversion**: Screen is divided into a 100px grid.
  - `Seat_ID = f"Seat_{int(x//100)}_{int(y//100)}"`
- **Registration**: The first *Named Person* identified in a Grid Cell is registered to that `Seat_ID`.
- **Flicker Protection**: An alert is only raised if a different *Named Person* stays in that seat for **20 consecutive frames** (approx. 2s) to ignore identity flickering.

### B. Sustained Phone Usage
Instead of just object detection, we use **Behavior Correlation**:
- **Pattern**: `(Peeking_Down == True) && (Wrist_Movement_Detected == True)`.
- **Usage Timer**: Increments when both conditions are met. If `timer > 15`, it flags sustained hidden phone use.

### C. Collusion (Temporal Synchronization)
- **Logic**: Uses a global `COLLUSION_LOG` to track the timestamps of every person's behavior.
- **Formula**: If `Person_A.Turn_Time - Person_B.Turn_Time < 2.0s`, a Potential Collusion alert is flagged.

---

## 4. AI HAT +2: Spatial Rules

### A. Movement Scoring
- **Scoring Engine**: Weights are assigned to specific behaviors.
  - `Suspicious_Score = Σ(Weight_i * Occurrence_i)`
- **Threshold**: An alert `High Risk` is triggered at `Score >= 50`.

### B. Zone Logic
- **Zone Definition**: `(x1, y1, x2, y2)` tuples defining Aisle/Door regions.
- **Invigilator Monitoring**:
  - `Timer = Current_Time - Last_Seen_Invigilator_in_Aisle`
  - Alerts if `Timer > 120s`.
- **Seat Abandonment**:
  - `Timer = Current_Time - Last_Seen_Student_in_Seat`
  - Alerts if `Timer > 45s`.

### C. Bag Interaction
- **Spatial Distance**: Calculates the distance between `Wrist_Keypoints` and the `Backpack_BBox`.
- **Logic**: Trigger if any wrist point is within `Box_Bounds + 50px`.

---

## 5. Security & Stability
- **Thread Safety**: Uses `threading.Lock()` for all shared behavior states and registries.
- **Event-Based Logging**: All behaviors use a `last_logged_alerts` set to ensure a behavior is logged **once per continuous event**, preventing log spam.
