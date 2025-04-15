import os
import numpy as np
import open3d as o3d


target_dir = 'ref/velodyne'
npy_files = os.listdir(target_dir)

npy_files = [fil for fil in npy_files if fil.endswith('.npy')]  


points = []

for file in npy_files:
    file_path = os.path.join(target_dir, file)
    data = np.load(file_path)
    points.append(data)
points = np.concatenate(points, axis=0)



target_dir = 'ref/labels'
npy_files = os.listdir(target_dir)

npy_files = [fil for fil in npy_files if fil.endswith('.npy')]  


labels = []

for file in npy_files:
    file_path = os.path.join(target_dir, file)
    data = np.load(file_path)
    labels.append(data)
labels = np.concatenate(labels, axis=0).squeeze(-1)


LEARNING_IDX_TO_NAME = {
    0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
    5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
    9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
    13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
    17: "terrain", 18: "pole", 19: "traffic-sign"
}


colors = [
    [0, 0, 0],       # unlabeled - black
    [0, 0, 1],       # car - blue
    [1, 0, 0],       # bicycle - red
    [1, 0, 1],       # motorcycle - magenta
    [0, 1, 1],       # truck - cyan
    [0.5, 0.5, 0],   # other-vehicle - olive
    [1, 0.5, 0],     # person - orange
    [1, 1, 0],       # bicyclist - yellow
    [1, 0, 0.5],     # motorcyclist - pink
    [0.5, 0.5, 0.5], # road - gray
    [0.5, 0, 0],     # parking - dark red
    [0, 0.5, 0],     # sidewalk - dark green
    [0, 0, 0.5],     # other-ground - dark blue
    [0, 0.5, 0.5],   # building - teal
    [0.5, 0, 0.5],   # fence - purple
    [0, 1, 0],       # vegetation - green
    [0.7, 0.7, 0.7], # trunk - light gray
    [0.7, 0, 0.7],   # terrain - light purple
    [0, 0.7, 0.7],   # pole - light cyan
    [0.7, 0.7, 0]    # traffic-sign - light yellow
]


def map_color(labels, colors):
    """
    Map the labels to colors.
    """
    color_map = np.zeros((labels.shape[0], 3), dtype=np.float32)


    for i in range(20):
        color_map[labels == i] = colors[i]
    return color_map


color_map = map_color(labels, colors)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
# Set points and colors
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(color_map)

o3d.io.write_point_cloud("output.ply", pcd, write_ascii=True)




