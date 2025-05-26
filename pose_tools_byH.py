
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

def pose_to_df(pose_list, coord_labels):
    """Convert a flat pose list into a DataFrame with named coordinates"""
    if len(pose_list) != len(coord_labels):
        raise ValueError("Pose and labels must match in length.")
    return DataFrame([pose_list], columns=coord_labels)

def poses_to_vectors(pose1, pose2):
    """Compute movement vector between two poses"""
    return [p2 - p1 for p1, p2 in zip(pose1, pose2)]

def pose_list_to_vector_df(pose_list, coord_labels):
    """Turn list of poses into a movement-vector DataFrame"""
    df = DataFrame(columns=coord_labels)
    for i in range(len(pose_list) - 1):
        vect = poses_to_vectors(pose_list[i], pose_list[i+1])
        df.loc[len(df)] = vect
    return df

def generate_vector_graph(pose, movement, pose_name1="A", pose_name2="B", save_path=None, ax=None):
    """Visualize movement from one pose to the next using vector arrows, optionally within a subplot axis."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        show_now = True
    else:
        show_now = False

    ax.set_title(f"Movement Vectors: {pose_name1} → {pose_name2}")
    for i in range(0, len(pose), 2):
        x = pose[i]
        y = pose[i+1]
        dx = movement[i]
        dy = movement[i+1]
        ax.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=1, color='blue')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xlim(0, 768)
    ax.set_ylim(768, 0)

    if save_path:
        plt.savefig(save_path)
    if show_now:
        plt.tight_layout()
        plt.show()
