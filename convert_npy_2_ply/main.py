import os
import numpy as np
import open3d as o3d
import yaml


yaml_label_dic = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}

yaml_label_color = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    10: [245, 150, 100],
    11: [245, 230, 100],
    13: [250, 80, 100],
    15: [150, 60, 30],
    16: [255, 0, 0],
    18: [180, 30, 80],
    20: [255, 0, 0],
    30: [30, 30, 255],
    31: [200, 40, 255],
    32: [90, 30, 150],
    40: [255, 0, 255],
    44: [255, 150, 255],
    48: [75, 0, 75],
    49: [75, 0, 175],
    50: [0, 200, 255],
    51: [50, 120, 255],
    52: [0, 150, 255],
    60: [170, 255, 150],
    70: [0, 175, 0],
    71: [0, 60, 135],
    72: [80, 240, 150],
    80: [150, 240, 255],
    81: [0, 0, 255],
    99: [255, 255, 50],
    252: [245, 150, 100],
    256: [255, 0, 0],
    253: [200, 40, 255],
    254: [30, 30, 255],
    255: [90, 30, 150],
    257: [250, 80, 100],
    258: [180, 30, 80],
    259: [255, 0, 0]
}


def load_yaml(path):
    DATA = yaml.safe_load(open(path, 'r'))
    # get number of interest classes, and the label mappings
    remapdict = DATA["learning_map_inv"]
    # make lookup table for mapping
    maxkey = max(remapdict.keys())
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    return remap_lut


def remap(label):
    upper_half = label >> 16      # get upper half for instances
    lower_half = label & 0xFFFF   # get lower half for semantics
    remap_lut = load_yaml('semantic-kitti.yaml')
    lower_half = remap_lut[lower_half]  # do the remapping of semantics
    label = (upper_half << 16) + lower_half   # reconstruct full label
    label = label.astype(np.uint32)
    return label


target_dir = 'points'
npy_files = os.listdir(target_dir)

npy_files = [fil for fil in npy_files if fil.endswith('.npy')]  


points = []

for file in npy_files:
    file_path = os.path.join(target_dir, file)
    data = np.load(file_path)
    points.append(data)
points = np.concatenate(points, axis=0)



target_dir = 'labels'
npy_files = os.listdir(target_dir)

npy_files = [fil for fil in npy_files if fil.endswith('.npy')]  


labels = []

for file in npy_files:
    file_path = os.path.join(target_dir, file)
    data = np.load(file_path)
    labels.append(data)
labels = np.concatenate(labels, axis=0).squeeze(-1)
labels_ref = labels.copy()
DATA = yaml.safe_load(open('semantic-kitti.yaml', 'r'))
remap_dict = DATA["learning_map_inv"]
for key in remap_dict.keys():
    labels[labels_ref == key] = remap_dict[key]

        
def map_color(labels, colors):
    """
    Map the labels to colors.
    """
    color_map = np.zeros((labels.shape[0], 3), dtype=np.float32)


    for key in yaml_label_color.keys():
        color_map[labels == key] = (np.array(colors[key]).astype(np.float32))/255.0
    return color_map

point_colors = map_color(labels, yaml_label_color)


# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
# Set points and colors
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(point_colors)

o3d.io.write_point_cloud("output.ply", pcd, write_ascii=True)



