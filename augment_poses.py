
import os
import json
import numpy as np

POSE_DIR = "poses"
OUTPUT_DIR = "poses"
CANVAS_WIDTH = 768

# Keypoint index swaps for left-right mirroring (OpenPose 25 keypoint format reduced to 18)
left_right_pairs = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13)]

def load_pose(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data[0]['people'][0]['pose_keypoints_2d']

def mirror_pose(pose):
    mirrored = pose[:]
    for i in range(0, len(pose), 3):
        mirrored[i] = CANVAS_WIDTH - pose[i]
    return mirrored

def swap_keypoints(pose):
    swapped = pose[:]
    for a, b in left_right_pairs:
        ax, ay = 3*a, 3*a+1
        bx, by = 3*b, 3*b+1
        swapped[ax], swapped[bx] = swapped[bx], swapped[ax]
        swapped[ay], swapped[by] = swapped[by], swapped[ay]
    return swapped

def jitter_pose(pose, sigma=2.0):
    jittered = pose[:]
    for i in range(0, len(pose), 3):
        jittered[i] += np.random.normal(0, sigma)
        jittered[i+1] += np.random.normal(0, sigma)
    return jittered

def save_pose(pose, original_name, suffix):
    new_name = original_name.replace('.json', f'_{suffix}.json')
    out_path = os.path.join(OUTPUT_DIR, new_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump([{"people": [{"pose_keypoints_2d": pose}]}], f, indent=2)

def augment_all():
    files = [f for f in os.listdir(POSE_DIR) if f.endswith('.json')]
    for f in files:
        try:
            pose = load_pose(os.path.join(POSE_DIR, f))
            mirrored = mirror_pose(pose)
            mirrored = swap_keypoints(mirrored)
            jittered = jitter_pose(pose)

            save_pose(mirrored, f, 'mirrored')
            save_pose(jittered, f, 'jittered')

            print(f"Augmented: {f}")
        except Exception as e:
            print(f"Skipping {f} due to error: {e}")

if __name__ == "__main__":
    augment_all()
