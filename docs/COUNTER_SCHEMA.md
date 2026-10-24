
# MM-Fit Squat Repetition Counter

## 1. Dataset Inspection

### `inspect_mmfit.py`
- Inspected the MM-Fit dataset structure.
- Explored available dataset files and components.
- Established an initial understanding of the dataset.

### `inspect_labels.py`
- Examined exercise annotation labels.
- Identified the columns: `start`, `end`, `reps`, and `exercise`.
- Found three labeled squat sets.

### `inspect_mmfit_details.py`
- Investigated dataset and recording details.
- Supported the identification of suitable samples for testing.

### `inspect_mmfit_layout.py`
- Examined the arrangement of pose data.
- Helped clarify the structure of pose streams.

## 2. Pose Data Analysis

### `inspect_pose.py`
- Inspected the pose array dimensions.
- Identified the pose shape as `(3, 73546, 18)`.

### `inspect_pose_columns.py`
- Investigated the ordering of pose columns.
- Identified joints such as hips, knees, feet, spine, shoulders, elbows, and wrists.

### `inspect_pose_coordinates.py`
- Examined numerical joint coordinates.
- Inspected coordinate ranges across three pose streams.
- Prepared the data for joint-angle calculations.

## 3. Frame Alignment

### `check_pose_alignment.py`
- Compared pose frames with labeled exercise intervals.
- First squat set: frames `2495–3144`.
- Available pose frames: `2505–3144`.
- Total available pose frames: **640**.
- Confirmed the usable pose-frame range for the first squat set.

## 4. Knee-Angle Analysis

- Calculated left and right knee angles across the first squat set.

| Measurement | Angle |
|---|---:|
| Left knee minimum | 104.41° |
| Left knee maximum | 169.88° |
| Right knee minimum | 100.65° |
| Right knee maximum | 172.76° |

- Experimental squat threshold: `120°`.
- Standing threshold: `155°`.
- These thresholds were used to distinguish the lowered squat position from the standing position.

## 5. Squat Repetition-Counter Testing

**Test file:** `test_mmfit_squat_counter.py`

- Tested the counter against three labeled squat sets.
- Compared expected repetitions with detected repetitions.

| Set | Expected Reps | Detected Reps | Result |
|---:|---:|---:|---|
| 1 | 10 | 10 | Match |
| 2 | 10 | 10 | Match |
| 3 | 10 | 10 | Match |
| **Total** | **30** | **30** | **Match** |

**Test Result:** All three tested sets matched their annotated repetition counts, with 30 expected repetitions and 30 detected repetitions. This result applies to the tested sets and does not, by itself, establish generalization to other participants or sessions.

## 6.Work Update — MM-Fit Squat Counter Validation

### a. `check_mmfit_sessions.py`
- **Purpose:** Check available MM-Fit sessions and identify suitable data for testing.
- **Working:** Inspect session data before running repetition-counter tests.

### b. `test_mmfit_squat_counter.py`
- **Purpose:** Test the squat repetition counter against expected repetition labels.
- **Working:** Validate the counter using the existing knee-angle thresholds: 120° for squat detection and 155° for standing detection.
- **Result:** Three sets matched exactly, with 30 expected and 30 detected repetitions.

### c. `test_mmfit_squat_generalization.py`
- **Purpose:** Test whether the squat counter generalizes to additional MM-Fit sessions.
- **Working:** Validate performance across sessions beyond the initial test without changing the thresholds.
- **Previously reported result:** Sessions `w02` and `w03` achieved 6/6 exact set matches and 60/60 repetitions combined.