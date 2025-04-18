import os
import numpy as np
import open3d as o3d
import yaml
from rich import print


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


groundtruth_pth= "testing.ply"


# Read the PLY file.
ground_truth_pcd = o3d.io.read_point_cloud(groundtruth_pth)

gt_points = np.array(ground_truth_pcd.points)
gt_colors = np.array(ground_truth_pcd.colors)
gt_colors = (gt_colors *255).astype(np.uint8)

# Ensure we only use XYZ coordinates
if gt_points.shape[1] > 3:
    print(f"WARNING: Point cloud has {gt_points.shape[1]} dimensions, truncating to first 3 (XYZ)")
    gt_points = gt_points[:, :3]



prediction_pth = "segmented.ply"
# Read the PLY file.
pred_pcd = o3d.io.read_point_cloud(prediction_pth)
pred_points = np.array(pred_pcd.points)
pred_colors = np.array(pred_pcd.colors) if pred_pcd.has_colors() else None
pred_colors = (pred_colors *255).astype(np.uint8)
# Ensure we only use XYZ coordinates

if pred_points.shape[1] > 3:
    print(f"WARNING: Point cloud has {pred_points.shape[1]} dimensions, truncating to first 3 (XYZ)")
    pred_points = pred_points[:, :3]



gt_labs = np.zeros((gt_points.shape[0],), dtype=np.int32)
for key,value in yaml_label_color.items():
    print(f"{key}: {value}")
    key_bool = (gt_colors[:,0] == yaml_label_color[key][0]) & (gt_colors[:,1] == yaml_label_color[key][1]) & (gt_colors[:,2] == yaml_label_color[key][2])
    print(f"Num of Trues for {key}: {np.sum(key_bool)}")
    gt_labs[key_bool]= key



pred_labs = np.zeros((pred_points.shape[0],), dtype=np.int32)
for key,value in yaml_label_color.items():
    print(f"{key}: {value}")
    key_bool =(pred_colors[:,0] == yaml_label_color[key][0]) & (pred_colors[:,1] == yaml_label_color[key][1]) & (pred_colors[:,2] == yaml_label_color[key][2])
    print(f"Num of Trues for {key}: {np.sum(key_bool)}")
    pred_labs[key_bool]= key

# Calculate Intersection over Union (IoU) metrics

# Get all unique class labels present in either ground truth or prediction
all_classes = np.unique(np.concatenate((gt_labs, pred_labs)))
# Remove background class (0) if desired
if 0 in all_classes:
    all_classes = all_classes[all_classes != 0]

print(f"\nEvaluating IoU for {len(all_classes)} classes:")

# Initialize dictionaries to store results
class_iou = {}
class_intersection = {}
class_union = {}
class_gt_count = {}
class_pred_count = {}

# Calculate IoU for each class
for cls in all_classes:
    # Skip classes that might be in the dictionary but not actually present in the data
    if cls not in yaml_label_dic:
        continue
        
    # Get binary masks for current class
    gt_mask = (gt_labs == cls)
    pred_mask = (pred_labs == cls)
    
    # Calculate intersection and union
    intersection = np.sum(gt_mask & pred_mask)
    union = np.sum(gt_mask | pred_mask)
    
    # Store counts for later use in weighted metrics
    gt_count = np.sum(gt_mask)
    pred_count = np.sum(pred_mask)
    
    # Calculate IoU (handling edge case where union is 0)
    iou = float(intersection) / union if union > 0 else 0.0
    
    # Store results
    class_iou[cls] = iou
    class_intersection[cls] = intersection
    class_union[cls] = union
    class_gt_count[cls] = gt_count
    class_pred_count[cls] = pred_count
    
    print(f"Class {cls} ({yaml_label_dic[cls]}): IoU = {iou:.4f}")
    # print(f"  - GT points: {gt_count}, Pred points: {pred_count}")
    # print(f"  - Intersection: {intersection}, Union: {union}")

# Calculate mean IoU (mIoU)
valid_classes = [cls for cls in all_classes if cls in class_iou]
miou = np.mean([class_iou[cls] for cls in valid_classes])
print(f"\nMean IoU (mIoU): {miou:.4f}")

# Calculate weighted IoU
# Weight by the number of points in the ground truth for each class
total_gt_points = sum(class_gt_count.values())
weighted_iou = 0.0
if total_gt_points > 0:
    for cls in valid_classes:
        weight = class_gt_count[cls] / total_gt_points
        weighted_iou += weight * class_iou[cls]

print(f"Weighted IoU (by GT point count): {weighted_iou:.4f}")

# Calculate confusion matrix (optional for detailed analysis)
# This helps analyze which classes are being confused with each other
num_classes = len(valid_classes)
class_to_idx = {cls: i for i, cls in enumerate(valid_classes)}
confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

# Find points that exist in both clouds (you may need to modify this based on your data structure)
# This assumes the point clouds have exactly the same points in the same order
# If not, you would need to match points first (e.g., by position)
for i in range(min(len(gt_labs), len(pred_labs))):
    gt_class = gt_labs[i]
    pred_class = pred_labs[i]
    
    # Skip if either class is not in our valid classes list
    if gt_class not in class_to_idx or pred_class not in class_to_idx:
        continue
        
    gt_idx = class_to_idx[gt_class]
    pred_idx = class_to_idx[pred_class]
    confusion_matrix[gt_idx, pred_idx] += 1

# Output results to a file
with open("segmentation_evaluation_results.txt", "w") as f:
    f.write("POINT CLOUD SEGMENTATION EVALUATION\n")
    f.write("==================================\n\n")
    
    f.write("Per-Class IoU Results:\n")
    for cls in valid_classes:
        f.write(f"Class {cls} ({yaml_label_dic[cls]}): IoU = {class_iou[cls]:.4f}\n")
        f.write(f"  - GT points: {class_gt_count[cls]}, Pred points: {class_pred_count[cls]}\n")
    
    f.write(f"\nMean IoU (mIoU): {miou:.4f}\n")
    f.write(f"Weighted IoU (by GT point count): {weighted_iou:.4f}\n")
    
    f.write("\nConfusion Matrix:\n")
    f.write("Format: [row=ground truth, column=prediction]\n")
    f.write("Classes: " + ", ".join([f"{cls}:{yaml_label_dic[cls]}" for cls in valid_classes]) + "\n\n")
    
    # Print confusion matrix headers
    f.write("      ")
    for cls in valid_classes:
        f.write(f"{cls:>5} ")
    f.write("\n")
    
    # Print confusion matrix data
    for i, gt_cls in enumerate(valid_classes):
        f.write(f"{gt_cls:>5} ")
        for j in range(num_classes):
            f.write(f"{confusion_matrix[i, j]:>5} ")
        f.write("\n")

print(f"\nResults saved to segmentation_evaluation_results.txt")

# dump the .txt into terminal
with open("segmentation_evaluation_results.txt", "r") as f:
    print(f.read())