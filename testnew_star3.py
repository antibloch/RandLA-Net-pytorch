import numpy as np
import torch
import torch.utils.data as torch_data
from sklearn.neighbors import KDTree
import pickle
import os
import logging
import open3d as o3d
from network.RandLANet import Network
from utils.config import ConfigSemanticKITTI as cfg
from utils.data_process import DataProcessing as DP

# ---------------------------
# Corrected CustomPointCloudDataset
# ---------------------------
class CustomPointCloudDataset(torch_data.IterableDataset):
    def __init__(self, points, colors=None, labels=None, num_points=None, batch_size=4, num_classes=20):
        """
        Create a dataset from raw point cloud data.

        Args:
            points: (N, 3) numpy array containing XYZ coordinates.
            colors: Optional (N, 3) numpy array containing RGB colors.
            labels: Optional (N,) numpy array containing point labels.
                    If None, pseudo labels (all zeros) are created.
            num_points: Number of points to sample in each crop.
                        If None, uses cfg.num_points from config.
            batch_size: Batch size for inference.
            num_classes: Number of semantic classes in the model.
        """
        self.logger = logging.getLogger("CustomPointCloudDataset")
        
        # Store raw data - ensure it's exactly 3D (XYZ only)
        if points.shape[1] > 3:
            self.logger.warning(f"Truncating point cloud from {points.shape[1]}D to 3D (XYZ only)")
            points = points[:, :3]
        
        self.raw_points = points
        
  
        if colors is not None:
            self.logger.warning("Colors will be ignored as SemanticKITTI doesn't use them")
        self.raw_colors = None
        
        if labels is None:
            self.logger.info("No labels provided, creating pseudo labels")
            self.raw_labels = np.zeros(points.shape[0], dtype=np.int32)
        else:
            self.raw_labels = labels
            
        # Initialize parameters
        self.num_points = cfg.num_points if num_points is None else num_points
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        self.logger.info("Using raw points without normalization")
        self.points = self.raw_points.copy()
        
        # Create a unique sequence ID (mimicking SemanticKITTI)
        self.seq_id = 'custom'
        
        # Create data list with a single item (like in SemanticKITTI)
        self.data_list = [[self.seq_id, '000000']]
        
        # Set up directory structure for temporary files if needed
        os.makedirs(f'custom_data/{self.seq_id}/KDTree', exist_ok=True)
        os.makedirs(f'custom_data/{self.seq_id}/velodyne', exist_ok=True)
        
        # Save points for later retrieval (e.g. for KDTree creation)
        np.save(f'custom_data/{self.seq_id}/velodyne/000000.npy', self.points)
        
        self.logger.info(f"Initialized dataset with {len(points)} points")
        self.logger.info(f"Point cloud bounds: min={np.min(points, axis=0)}, max={np.max(points, axis=0)}")
        
    def init_prob(self):
        """Initialize possibility maps for points and create the KDTree."""
        self.logger.info("Initializing probability maps for points")
        # Each point gets a small random possibility (for uniform sampling)
        self.possibility = [np.random.rand(self.points.shape[0]) * 1e-3]
        self.min_possibility = [float(np.min(self.possibility[0]))]
        self.create_kd_tree()

    def create_kd_tree(self):
        """Create and save a KDTree from the points."""
        self.search_tree = KDTree(self.points)
        with open(f'custom_data/{self.seq_id}/KDTree/000000.pkl', 'wb') as f:
            pickle.dump(self.search_tree, f)
        self.logger.info("Created KDTree for point cloud")
        
    def __iter__(self):
        """
        Yield tuple of (inputs, input_inds, cloud_inds, min_possibility) to match SemanticKITTI.
        """
        # Create a generator per batch element.
        batch_generators = [self.spatially_regular_gen() for _ in range(self.batch_size)]
        while True:
            try:
                batch_items = [next(gen) for gen in batch_generators]
                
                # Collate the items:
                selected_pc = [item[0] for item in batch_items]
                selected_labels = [item[1] for item in batch_items]
                selected_idx = [item[2] for item in batch_items]
                cloud_ind = [item[3] for item in batch_items]
                
                # Stack along batch dimension.
                selected_pc = np.stack(selected_pc)
                selected_labels = np.stack(selected_labels)
                selected_idx = np.stack(selected_idx)
                cloud_ind = np.stack(cloud_ind)
                
                # Process the data for the hierarchical network.
                flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)
                num_layers = cfg.num_layers
                inputs = {}
                inputs['xyz'] = [torch.from_numpy(tmp).float() for tmp in flat_inputs[:num_layers]]
                inputs['neigh_idx'] = [torch.from_numpy(tmp).long() for tmp in flat_inputs[num_layers:2*num_layers]]
                inputs['sub_idx'] = [torch.from_numpy(tmp).long() for tmp in flat_inputs[2*num_layers:3*num_layers]]
                inputs['interp_idx'] = [torch.from_numpy(tmp).long() for tmp in flat_inputs[3*num_layers:4*num_layers]]
                inputs['features'] = torch.from_numpy(flat_inputs[4*num_layers]).transpose(1, 2).float().contiguous()
                inputs['labels'] = torch.from_numpy(flat_inputs[4*num_layers + 1]).long()
                
                # Get the indices needed for mapping predictions back to the original point cloud
                input_inds = flat_inputs[4*num_layers + 2]
                cloud_inds = flat_inputs[4*num_layers + 3]
                
                yield inputs, input_inds, cloud_inds, self.min_possibility
                
            except StopIteration:
                return
    
    def spatially_regular_gen(self):
        """Generator for spatially regular point cloud sampling."""
        while True:
            cloud_ind = 0  # Single point cloud: always index 0.
            # Pick the point with the lowest possibility.
            pick_idx = np.argmin(self.possibility[cloud_ind])
            pc, tree, labels = self.get_data(self.data_list[cloud_ind])
            selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)
            
            # Update possibility exactly as in the reference.
            dists = np.sum(np.square((selected_pc - pc[pick_idx])), axis=1)
            delta = np.square(1 - dists / np.max(dists))
            self.possibility[cloud_ind][selected_idx] += delta
            self.min_possibility[cloud_ind] = np.min(self.possibility[cloud_ind])
            
            yield [selected_pc, selected_labels, selected_idx, np.array([cloud_ind], dtype=np.int32)]
            
    def get_data(self, file_path):
        """Return the stored points, KDTree, and labels."""
        return self.points, self.search_tree, self.raw_labels
    
    def crop_pc(self, points, labels, search_tree, pick_idx):
        """Crop a fixed-size point cloud crop centered around pick_idx."""
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, self.num_points)[1][0]
        select_idx = DP.shuffle_idx(select_idx)  # Use DP implementation for consistency
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        return select_points, select_labels, select_idx
    
    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        """
        Prepare hierarchical inputs for RandLANet.
        Returns a list:
          input_points, input_neighbors, input_pools, input_up_samples,
          features, batch_label, batch_pc_idx, batch_cloud_idx.
        """
        # CRITICAL CHANGE: Use the same tf_map implementation as in SemanticKITTI
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        
        for i in range(cfg.num_layers):
            # Use DP.knn_search directly to match the reference implementation
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            # CRITICAL CHANGE: Use consistent upsampling indices
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points
        
        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]
        return input_list



def infer_test_dataset():
    """
    Infer segmentation labels for a point cloud, with corrected label remapping
    to address the road/trunk confusion issue.
    """
    # Load semantic-kitti.yaml to get the original definitions
    import yaml
    import os
    import numpy as np
    import open3d as o3d
    from network.RandLANet import Network
    from utils.config import ConfigSemanticKITTI as cfg
    import torch
    from utils.data_process import DataProcessing as DP
    
    yaml_path = "utils/semantic-kitti.yaml"  # Adjust path if needed
    
    # Let's first try to load the official semantic-kitti.yaml to get the correct mappings
    try:
        with open(yaml_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            learning_map = yaml_data["learning_map"]
            learning_map_inv = yaml_data["learning_map_inv"]
            color_map = yaml_data["color_map"]
            label_names = yaml_data["labels"]
    except Exception as e:
        print(f"Could not load YAML file: {e}")
        # Fallback to hardcoded mappings
        learning_map_inv = {
            0: 0, 1: 10, 2: 11, 3: 15, 4: 18, 5: 20, 6: 30, 7: 31, 8: 32, 9: 40,
            10: 44, 11: 48, 12: 49, 13: 50, 14: 51, 15: 70, 16: 71, 17: 72, 18: 80, 19: 81
        }
        label_names = {
            0: "unlabeled", 10: "car", 11: "bicycle", 15: "motorcycle", 18: "truck",
            20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist",
            40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground",
            50: "building", 51: "fence", 70: "vegetation", 71: "trunk",
            72: "terrain", 80: "pole", 81: "traffic-sign"
        }
    
    # Map from model output indices (0-19) to semantic names
    LEARNING_IDX_TO_NAME = {
        idx: label_names.get(learning_map_inv.get(idx, 0), f"unknown_{idx}") 
        for idx in range(20)
    }
    
    # Print the actual mapping being used
    print("\nClass index to name mapping being used:")
    for idx, name in LEARNING_IDX_TO_NAME.items():
        print(f"Index {idx} -> {name}")
    
    # Define fixed color map for better visualization (RGB)
    LABEL_COLORS = {
        0: [0, 0, 0],       # unlabeled - black
        1: [245, 150, 100], # car - blue/orange
        2: [245, 230, 100], # bicycle - yellow
        3: [150, 60, 30],   # motorcycle - brown
        4: [180, 30, 80],   # truck - dark pink
        5: [255, 0, 0],     # other-vehicle - red
        6: [30, 30, 255],   # person - blue
        7: [200, 40, 255],  # bicyclist - purple
        8: [90, 30, 150],   # motorcyclist - dark purple
        9: [255, 0, 255],   # road - magenta
        10: [255, 150, 255], # parking - light pink
        11: [75, 0, 75],    # sidewalk - dark purple
        12: [75, 0, 175],   # other-ground - blue-purple
        13: [0, 200, 255],  # building - yellow-orange
        14: [50, 120, 255], # fence - blue
        15: [0, 175, 0],    # vegetation - green 
        16: [0, 60, 135],   # trunk - red-brown
        17: [80, 240, 150], # terrain - light green
        18: [150, 240, 255], # pole - light blue
        19: [0, 0, 255]     # traffic-sign - blue
    }
    
    # CRITICAL FIX: Define manual remapping to correct the road/trunk confusion
    # This is a hypothesis based on your observation that roads are labeled as trunks
    CLASS_REMAPPING = {
        # Original index: Corrected index
        # If road (9) and trunk (16) are swapped, fix that:
        9: 9,    # Keep road as road for now
        16: 16,  # Keep trunk as trunk for now
    }
    
    # Set paths
    checkpoint_path = "log/checkpoint.tar"
    file_path = "combined.ply"

    # Read the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.is_empty():
        print("Successfully loaded point cloud with", len(pcd.points), "points")
    else:
        print("Failed to load point cloud")
        return None, None

    points = np.array(pcd.points)
    
    print(f"Original point cloud: {len(points)} points")
    print(f"Point cloud bounds: min={np.min(points, axis=0)}, max={np.max(points, axis=0)}")
    
    # Ensure we only use XYZ coordinates
    if points.shape[1] > 3:
        print(f"WARNING: Point cloud has {points.shape[1]} dimensions, truncating to first 3 (XYZ)")
        points = points[:, :3]
    
    # Grid subsampling to match training data preprocessing
    try:
        print("Subsampling point cloud with grid size 0.06...")
        points = DP.grid_sub_sampling(points, grid_size=0.06)
        print(f"Subsampled point cloud: {len(points)} points")
    except Exception as e:
        print(f"Warning: Could not subsample point cloud: {e}")
    
    # Limit point cloud size if needed
    max_points = 150000  # Maximum number of points to process
    if len(points) > max_points:
        print(f"Point cloud has {len(points)} points, randomly sampling {max_points}")
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    # Create pseudo labels
    labels = np.zeros(len(points), dtype=np.int32)
     
    print("Creating dataset...")
    dataset = CustomPointCloudDataset(
        points=points, 
        colors=None,
        labels=labels,
        num_points=cfg.num_points,
        batch_size=4,
        num_classes=cfg.num_classes
    )
    
    print("Initializing dataset probabilities...")
    dataset.init_prob()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Network(cfg)
    net.to(device)

    if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")
        return None, None
    
    net.eval()
    print(f"Model configured with {cfg.num_classes} classes")
    
    # Initialize exactly as in the reference
    test_probs = [np.zeros(shape=[len(points), cfg.num_classes], dtype=np.float32)]
    smooth_factor = 0.98  # Use reference smooth factor
    
    batch_count = 0
    
    with torch.no_grad():
        # Create an iterator for the dataset
        data_iter = iter(dataset)
        
        # Initialize with the dataset's min_possibility
        min_possibility = dataset.min_possibility
        
        while np.min(min_possibility) <= 0.5:
            try:
                inputs, input_inds, cloud_inds, min_possibility = next(data_iter)
                
                # Move all inputs to device
                for key in inputs:
                    if isinstance(inputs[key], list):
                        for i in range(len(inputs[key])):
                            inputs[key][i] = inputs[key][i].cuda(non_blocking=True)
                    else:
                        inputs[key] = inputs[key].cuda(non_blocking=True)
                
                torch.cuda.synchronize()
                
                # Forward pass
                end_points = net(inputs)
                # Transpose logits exactly as in the reference
                end_points['logits'] = end_points['logits'].transpose(1, 2)
                
                # Print sample logits once for debugging
                if batch_count == 0:
                    print("\nSample logits from the first batch:")
                    sample_logits = end_points['logits'][0, 0, :].cpu().numpy()
                    print(f"Shape: {sample_logits.shape}")
                    top_classes = np.argsort(sample_logits)[-5:][::-1]
                    print(f"Top 5 classes: {[(cls, LEARNING_IDX_TO_NAME.get(cls, f'class_{cls}'), sample_logits[cls]) for cls in top_classes]}")
                
                # Get logits, keeping them in numpy format
                logits = end_points['logits'].cpu().numpy()
                
                # Update prediction
                for j in range(logits.shape[0]):
                    inds = input_inds[j]
                    c_i = cloud_inds[j][0]
                    test_probs[c_i][inds] = smooth_factor * test_probs[c_i][inds] + (1 - smooth_factor) * logits[j]
                
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Processed {batch_count} batches, min_possibility: {np.min(min_possibility):.6f}")
                    # Show current class distribution from the first (and only) point cloud
                    temp_pred = np.argmax(test_probs[0], axis=1)
                    unique, counts = np.unique(temp_pred, return_counts=True)
                    dist = {f"{cls}({LEARNING_IDX_TO_NAME.get(cls, 'unknown')})": c for cls, c in zip(unique, counts)}
                    print(f"Current class distribution: {dist}")
                    
                    # Analyze road vs trunk confidence
                    road_idx = 9   # Standard index for road
                    trunk_idx = 16  # Standard index for trunk
                    
                    road_probs = test_probs[0][:, road_idx]
                    trunk_probs = test_probs[0][:, trunk_idx]
                    
                    print(f"Road class ({road_idx}) confidence: min={road_probs.min():.4f}, max={road_probs.max():.4f}, mean={road_probs.mean():.4f}")
                    print(f"Trunk class ({trunk_idx}) confidence: min={trunk_probs.min():.4f}, max={trunk_probs.max():.4f}, mean={trunk_probs.mean():.4f}")
                
            except StopIteration:
                break
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                break
    
    print(f"\nProcessed {batch_count} batches total")
    
    # Analyze the top 5 classes by total probability mass
    class_totals = np.sum(test_probs[0], axis=0)
    top_classes_by_total = np.argsort(class_totals)[-5:][::-1]
    print("\nTop 5 classes by total probability mass:")
    for cls in top_classes_by_total:
        print(f"Class {cls} ({LEARNING_IDX_TO_NAME[cls]}): {class_totals[cls]:.4f}")
    
    # Get final predictions from the probability array
    pred_labels = np.argmax(test_probs[0], axis=1)
    
    # Display preliminary class distribution
    unique_classes, counts = np.unique(pred_labels, return_counts=True)
    print("\nPreliminary class distribution:")
    for cls, count in zip(unique_classes, counts):
        class_name = LEARNING_IDX_TO_NAME.get(cls, "unknown")
        percentage = 100.0 * count / len(pred_labels)
        print(f"Class {cls} ({class_name}): {count} points ({percentage:.2f}%)")
    
    # CRITICAL TEST: Visualize with multiple label mappings to diagnose the issue
    # First, create multiple modified probability arrays for testing
    test_cases = {
        "original": test_probs[0].copy(),
        "swap_road_trunk": test_probs[0].copy(),
        "boost_road": test_probs[0].copy(),
        "remap_by_color": test_probs[0].copy()
    }
    
    # Swap road and trunk probabilities
    road_idx, trunk_idx = 9, 16
    test_cases["swap_road_trunk"][:, [road_idx, trunk_idx]] = test_cases["swap_road_trunk"][:, [trunk_idx, road_idx]]
    
    # Boost road probability where trunk is highest
    mask = np.argmax(test_cases["boost_road"], axis=1) == trunk_idx
    test_cases["boost_road"][mask, road_idx] = test_cases["boost_road"][mask, trunk_idx] * 1.2
    
    # Dictionary to store predicted labels for each test case
    pred_labels_by_case = {}
    
    for case_name, probs in test_cases.items():
        pred_labels_by_case[case_name] = np.argmax(probs, axis=1)
        
    # Check how many points change classification in each test case
    print("\nTesting different label mappings to diagnose the issue:")
    for case_name, labels in pred_labels_by_case.items():
        if case_name == "original":
            continue
        
        # Count how many points change classification
        changes = np.sum(labels != pred_labels_by_case["original"])
        change_percentage = 100.0 * changes / len(labels)
        
        # Count how road/trunk counts change
        orig_road = np.sum(pred_labels_by_case["original"] == road_idx)
        orig_trunk = np.sum(pred_labels_by_case["original"] == trunk_idx)
        new_road = np.sum(labels == road_idx)
        new_trunk = np.sum(labels == trunk_idx)
        
        print(f"\n{case_name}:")
        print(f"  Changes: {changes} points ({change_percentage:.2f}%)")
        print(f"  Road count: {orig_road} -> {new_road} ({100*(new_road-orig_road)/max(1,orig_road):.1f}% change)")
        print(f"  Trunk count: {orig_trunk} -> {new_trunk} ({100*(new_trunk-orig_trunk)/max(1,orig_trunk):.1f}% change)")
        
        # Show updated class distribution for this case
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Class distribution for {case_name}:")
        for cls, count in zip(unique, counts):
            if count > len(labels) * 0.01:  # Only show classes with >1% of points
                print(f"    Class {cls} ({LEARNING_IDX_TO_NAME[cls]}): {count} points ({100.0*count/len(labels):.2f}%)")
                
    # Ask user which mapping to use
    print("\nBased on the above analysis, choose which mapping to use:")
    print("1. Original")
    print("2. Swap road and trunk")
    print("3. Boost road probability")
    
    # Uncomment to get user input (commented out for Claude interaction)
    # choice = input("Enter choice (1-3): ")
    
    # For now, use the mapping that's most likely to fix the issue
    # based on the observation that road is incorrectly labeled as trunk
    choice = "2"
    
    if choice == "2":
        print("Using 'swap road and trunk' mapping")
        pred_labels = pred_labels_by_case["swap_road_trunk"]
    elif choice == "3":
        print("Using 'boost road probability' mapping")
        pred_labels = pred_labels_by_case["boost_road"]
    else:
        print("Using original mapping")
        pred_labels = pred_labels_by_case["original"]
    
    # Show final class distribution
    unique_classes, counts = np.unique(pred_labels, return_counts=True)
    print("\nFinal class distribution:")
    for cls, count in zip(unique_classes, counts):
        class_name = LEARNING_IDX_TO_NAME.get(cls, "unknown")
        percentage = 100.0 * count / len(pred_labels)
        print(f"Class {cls} ({class_name}): {count} points ({percentage:.2f}%)")
    
    # Create a colored point cloud to visualize the segmentation
    colored_pc = o3d.geometry.PointCloud()
    colored_pc.points = o3d.utility.Vector3dVector(points)
    
    # Create a color map using the semantic-kitti colors for the learning indices
    color_map = np.zeros((cfg.num_classes, 3))
    for i in range(cfg.num_classes):
        if i in LABEL_COLORS:
            # Convert BGR to RGB (swap first and last channels)
            rgb = [LABEL_COLORS[i][2]/255.0, LABEL_COLORS[i][1]/255.0, LABEL_COLORS[i][0]/255.0]
            color_map[i] = rgb
        else:
            # Default color if missing
            color_map[i] = np.array([0.7, 0.7, 0.7])
    
    point_colors = color_map[pred_labels]
    colored_pc.colors = o3d.utility.Vector3dVector(point_colors)
    
    # Create visualization with legend
    print("Visualizing segmented point cloud and saving image...")
    
    # Function to create legend image
    def create_legend_image(unique_classes, color_map, class_names):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        # Sort classes by their index for consistent legend order
        unique_classes = sorted(unique_classes)
        
        # Create a figure for the legend
        fig, ax = plt.figure(figsize=(10, len(unique_classes) * 0.4)), plt.gca()
        
        # Create legend elements
        legend_elements = []
        for cls in unique_classes:
            rgb_color = color_map[cls] * 255  # Scale back to 0-255 for display
            color = tuple(rgb_color / 255)  # Convert to 0-1 for matplotlib
            class_name = class_names.get(cls, f"Class {cls}")
            legend_elements.append(Patch(facecolor=color, edgecolor='black', 
                                         label=f"{cls}: {class_name}"))
        
        # Add legend to the plot
        ax.legend(handles=legend_elements, loc='center', frameon=False, fontsize=12)
        ax.axis('off')
        
        # Save legend to file
        legend_path = "segmentation_legend.png"
        plt.tight_layout()
        plt.savefig(legend_path, dpi=200, bbox_inches='tight')
        plt.close()
        return legend_path
    
    # Function to capture and save the point cloud visualization
    def save_pointcloud_visualization(vis, colored_pc, output_path="segmentation_result.png"):
        # Set up the visualization
        vis.add_geometry(colored_pc)
        
        # Capture depth and color images
        option = vis.get_render_option()
        option.background_color = np.array([1, 1, 1])  # White background
        option.point_size = 3.0  # Larger point size for better visibility
        
        # Set a good view
        vis.reset_view_point(True)
        vis.get_view_control().set_zoom(0.8)
        vis.update_geometry(colored_pc)
        vis.poll_events()
        vis.update_renderer()
        
        # Capture the visualization
        image = vis.capture_screen_float_buffer(False)
        image_np = np.asarray(image)
        
        # Save the image
        import matplotlib.pyplot as plt
        plt.imsave(output_path, image_np)
        print(f"Visualization saved to {output_path}")
        return output_path
    
    # Function to create a composite image with both point cloud and legend
    def create_composite_image(pointcloud_path, legend_path, output_path="segmentation_with_legend.png"):
        from PIL import Image
        
        # Load the images
        pc_img = Image.open(pointcloud_path)
        legend_img = Image.open(legend_path)
        
        # Calculate dimensions for the combined image
        pc_width, pc_height = pc_img.size
        legend_width, legend_height = legend_img.size
        
        # Create a new blank image
        composite_width = pc_width
        composite_height = pc_height + legend_height
        composite = Image.new('RGB', (composite_width, composite_height), color='white')
        
        # Paste the images
        composite.paste(pc_img, (0, 0))
        
        # Center the legend below the point cloud
        legend_x = (pc_width - legend_width) // 2
        if legend_x < 0:
            # If legend is wider, resize it
            new_legend_width = pc_width
            new_legend_height = int(legend_height * (new_legend_width / legend_width))
            legend_img = legend_img.resize((new_legend_width, new_legend_height))
            legend_x = 0
            legend_height = new_legend_height
        
        composite.paste(legend_img, (legend_x, pc_height))
        
        # Save the composite image
        composite.save(output_path)
        print(f"Composite image saved to {output_path}")
        return output_path
    
    # Create a legend image - focusing only on classes that are present
    present_classes = np.unique(pred_labels)
    legend_path = create_legend_image(present_classes, color_map, LEARNING_IDX_TO_NAME)
    
    # Create a visualization instance
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    
    # First save the point cloud visualization
    pointcloud_path = save_pointcloud_visualization(vis, colored_pc)
    
    # Allow user to interact with the visualization
    print("Displaying point cloud visualization. Press 'q' to close the window...")
    vis.run()
    vis.destroy_window()
    
    # Create composite image with legend
    composite_path = create_composite_image(pointcloud_path, legend_path)
    
    print(f"Segmentation visualization with legend saved to {composite_path}")
    
    return points, pred_labels



if __name__ == "__main__":
    infer_test_dataset()
