
from pathlib import Path
import numpy as np
import pandas as pd


# 1. SELECT A DIFFERENT SESSION

# Change this to the parent folder containing w00, w01, w02...
DATASET_DIR = Path(r"C:\Users\Krushna\Desktop\VisionFit-AI\dataset\mmfit")

SESSION = "w03"
DATA_DIR = DATASET_DIR / SESSION

pose_file = DATA_DIR / f"{SESSION}_pose_3d.npy"
labels_file = DATA_DIR / f"{SESSION}_labels.csv"

if not pose_file.exists():
    raise FileNotFoundError(f"Pose file not found: {pose_file}")

if not labels_file.exists():
    raise FileNotFoundError(f"Labels file not found: {labels_file}")

pose = np.load(pose_file, allow_pickle=False)

labels = pd.read_csv(labels_file, header=None)
labels.columns = ["start", "end", "reps", "exercise"]

print("Testing session:", SESSION)
print("Pose shape:", pose.shape)


# 2. SELECT SQUAT LABELS

squat_labels = labels[
    labels["exercise"].astype(str).str.strip().str.lower() == "squats"
].copy()

print("Squat sets found:", len(squat_labels))

if squat_labels.empty:
    raise ValueError(f"No squat labels found in {SESSION}")


# 3. CALCULATE JOINT ANGLE

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return np.nan

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))


# 4. COUNT SQUATS IN ONE SET

def count_squats(frame_indices):
    left_angles = []
    right_angles = []

    for frame_index in frame_indices:

        def joint_position(joint_index):
            return pose[:, frame_index, joint_index + 1]

        left_hip = joint_position(1)
        left_knee = joint_position(2)
        left_foot = joint_position(3)

        right_hip = joint_position(4)
        right_knee = joint_position(5)
        right_foot = joint_position(6)

        left_angles.append(
            calculate_angle(left_hip, left_knee, left_foot)
        )

        right_angles.append(
            calculate_angle(right_hip, right_knee, right_foot)
        )

    left_angles = np.array(left_angles)
    right_angles = np.array(right_angles)

    average_angles = (left_angles + right_angles) / 2

    # Keep the original thresholds unchanged.
    squat_threshold = 120
    standing_threshold = 155

    repetitions = 0
    in_squat = False

    for angle in average_angles:
        if np.isnan(angle):
            continue

        if not in_squat and angle < squat_threshold:
            in_squat = True

        elif in_squat and angle > standing_threshold:
            repetitions += 1
            in_squat = False

    return repetitions


# 5. TEST EVERY SQUAT SET

frame_numbers = pose[0, :, 0]
results = []

for set_number, (_, squat) in enumerate(
    squat_labels.iterrows(), start=1
):
    start = int(squat["start"])
    end = int(squat["end"])
    expected = int(squat["reps"])

    mask = (frame_numbers >= start) & (frame_numbers <= end)
    frame_indices = np.flatnonzero(mask)

    if len(frame_indices) == 0:
        detected = None
        status = "NO POSE FRAMES"
    else:
        detected = count_squats(frame_indices)
        status = "MATCH" if detected == expected else "CHECK"

    results.append({
        "Set": set_number,
        "Start": start,
        "End": end,
        "Frames": len(frame_indices),
        "Expected": expected,
        "Detected": detected,
        "Status": status
    })


# 6. DISPLAY RESULTS

results_df = pd.DataFrame(results)

print("\n========== SQUAT TEST RESULTS ==========")
print(results_df.to_string(index=False))

valid_results = results_df.dropna(subset=["Detected"])

if len(valid_results) > 0:
    valid_results = valid_results.copy()
    valid_results["Detected"] = valid_results["Detected"].astype(int)

    matched = (
        valid_results["Expected"] == valid_results["Detected"]
    ).sum()

    print("\n=============== SUMMARY ===============")
    print("Session:", SESSION)
    print("Total labeled squat sets:", len(results_df))
    print("Sets with pose data:", len(valid_results))
    print(f"Exact set matches: {matched}/{len(valid_results)}")

    print("Expected reps in tested sets:",
          valid_results["Expected"].sum())
    print("Detected reps in tested sets:",
          valid_results["Detected"].sum())

    print("Exact set-match rate:",
          f"{matched / len(valid_results) * 100:.1f}%")
else:
    print("\nNo squat sets had usable pose frames.")
