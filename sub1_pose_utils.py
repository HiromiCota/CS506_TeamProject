
import copy, json
from os import listdir, path
from pandas import DataFrame

def index_poses(dir_path="poses"):
    json_list = listdir(dir_path)
    json_list.sort()
    return json_list

def get_pose_names(json_list):
    return [path.splitext(fname)[0] for fname in json_list]

def read_all_poses(json_list, dir_path="poses"):
    pose_list = []
    for fname in json_list:
        with open(f"{dir_path}/{fname}") as file:
            data = json.load(file)
            points_list = list(data[0]["people"][0]["pose_keypoints_2d"])
            unneeded_3d_coord = 1.0
            points_list = [i for i in points_list if i != unneeded_3d_coord]
            pose_list.append(points_list)
    return pose_list

def generate_column_labels(coords=18):
    labels = []
    for i in range(coords):
        formatted = f"{i:02d}"
        labels.append(formatted + "x")
        labels.append(formatted + "y")
    return labels

def prep_dataframe(labels):
    return DataFrame(columns=labels)

def store_pose(pose, labels, df):
    if len(pose) != len(labels):
        print("Error: Size mismatch between coordinates and labels")
        return df
    temp_points = copy.copy(pose)
    df.loc[len(df)] = temp_points

def store_all_poses(pose_list, labels, df):
    for pose in pose_list:
        store_pose(pose, labels, df)

def print_empties(df):
    df_nan = df[df.isna().any(axis=1)]
    if df_nan.empty:
        print("No incomplete rows")
    else:
        print(df_nan)

def coords_to_vectors(df):
    labels = df.columns
    vects = DataFrame(columns=labels)
    for i in range(0, len(df) - 1):
        pose1 = df.iloc[i].values.tolist()
        pose2 = df.iloc[i + 1].values.tolist()
        movement = [pose2[j] - pose1[j] for j in range(len(pose1))]
        vects.loc[len(df)] = movement
        vects.reset_index(drop=True, inplace=True)
    return vects
