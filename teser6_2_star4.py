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
    Infer segmentation labels for a point cloud, matching exactly
    the semantic-kitti reference implementation.
    """
    # Map prediction indices to class names.
    LEARNING_IDX_TO_NAME = {
        0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
        5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
        9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
        13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
        17: "terrain", 18: "pole", 19: "traffic-sign"
    }

    checkpoint_path = "log/checkpoint.tar"
    file_path = "combined.ply"
    
    # Import DataProcessing module for grid subsampling.
    from utils.data_process import DataProcessing as DP

    # Read the PLY file.
    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd.is_empty():
        print("Successfully loaded point cloud with", len(pcd.points), "points")
    else:
        print("Failed to load point cloud")
        return None, None

    points = np.array(pcd.points)
    colors = np.array(pcd.colors) if pcd.has_colors() else None
    
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
    
    # Create pseudo labels.
    labels = np.zeros(len(points), dtype=np.int32)
     
    print("Creating dataset...")
    dataset = CustomPointCloudDataset(
        points=points, 
        colors=None,  # Don't use colors to match SemanticKITTI
        labels=labels,
        num_points=cfg.num_points,  # Use exact same num_points as config
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
            model_keys = set(net.state_dict().keys())
            ckpt_keys = set(checkpoint['model_state_dict'].keys())
            if len(model_keys.difference(ckpt_keys)) > 0:
                print(f"Warning: {len(model_keys.difference(ckpt_keys))} keys in model not in checkpoint")
            if len(ckpt_keys.difference(model_keys)) > 0:
                print(f"Warning: {len(ckpt_keys.difference(model_keys))} keys in checkpoint not in model")
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
        
        # CRITICAL CHANGE: Follow the exact same loop condition as in reference
        while np.min(min_possibility) <= 0.5:
            try:
                # CRITICAL CHANGE: Unpack the tuple returned by the dataset
                # This matches the reference implementation where each iteration returns
                # (inputs, input_inds, cloud_inds, min_possibility)
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
                    print(f"Min: {sample_logits.min():.4f}, Max: {sample_logits.max():.4f}")
                    print(f"Mean: {sample_logits.mean():.4f}, Std: {sample_logits.std():.4f}")
                    # Show top 3 classes for this point
                    top_classes = np.argsort(sample_logits)[-3:][::-1]
                    print(f"Top classes: {[(cls, sample_logits[cls]) for cls in top_classes]}")
                
                # Get logits, keeping them in numpy format
                logits = end_points['logits'].cpu().numpy()
                
                # CRITICAL CHANGE: Update prediction exactly like in reference code
                # The reference uses a nested loop over batch samples
                for j in range(logits.shape[0]):
                    # Get the indices for this batch item
                    inds = input_inds[j]
                    c_i = cloud_inds[j][0]  # Should always be 0 in our case with 1 point cloud
                    
                    # This is the exact updating logic from the reference implementation
                    test_probs[c_i][inds] = smooth_factor * test_probs[c_i][inds] + (1 - smooth_factor) * logits[j]
                
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"Processed {batch_count} batches, min_possibility: {np.min(min_possibility):.6f}")
                    # Show current class distribution from the first (and only) point cloud
                    temp_pred = np.argmax(test_probs[0], axis=1)
                    unique, counts = np.unique(temp_pred, return_counts=True)
                    dist = {LEARNING_IDX_TO_NAME.get(u, f"class_{u}"): c for u, c in zip(unique, counts)}
                    print(f"Current class distribution: {dist}")
                    
                    # Print statistics about probability distribution for first few classes
                    for cls in range(min(5, cfg.num_classes)):
                        cls_probs = test_probs[0][:, cls]
                        print(f"Class {cls} ({LEARNING_IDX_TO_NAME.get(cls, 'unknown')}) stats: " 
                              f"min={cls_probs.min():.4f}, max={cls_probs.max():.4f}, "
                              f"mean={cls_probs.mean():.4f}, std={cls_probs.std():.4f}")
                
            except StopIteration:
                break
            except Exception as e:
                print(f"Error processing batch: {e}")
                import traceback
                traceback.print_exc()
                break
    
    print(f"\nProcessed {batch_count} batches total")
    
    # Use the first (and only) probability array to get final predictions
    pred_labels = np.argmax(test_probs[0], axis=1)
    print(f"Final predictions shape: {pred_labels.shape}")
    
    # Report class distribution
    unique_classes, counts = np.unique(pred_labels, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique_classes, counts):
        class_name = LEARNING_IDX_TO_NAME.get(cls, "unknown")
        percentage = 100.0 * count / len(pred_labels)
        print(f"Class {cls} ({class_name}): {count} points ({percentage:.2f}%)")
    
    # Try temperature scaling if we have limited class diversity
    if len(unique_classes) <= 3:  # If we have 3 or fewer classes
        print("\nTrying temperature scaling to improve class diversity...")
        
        # Store best predictions across temperatures
        best_pred_labels = pred_labels
        best_class_count = len(unique_classes)
        
        # Try multiple temperature values
        for temp in [1.5, 2.0, 3.0, 5.0, 10.0]:
            # Apply temperature scaling
            scaled_probs = test_probs[0] / temp
            scaled_pred = np.argmax(scaled_probs, axis=1)
            scaled_unique = np.unique(scaled_pred)
            
            print(f"Temperature {temp}: {len(scaled_unique)} unique classes")
            
            # If this temperature gives more classes, use these predictions
            if len(scaled_unique) > best_class_count:
                best_class_count = len(scaled_unique)
                best_pred_labels = scaled_pred
        
        # If we found a better distribution, use it
        if best_class_count > len(unique_classes):
            print(f"Found better class distribution with {best_class_count} classes")
            pred_labels = best_pred_labels
            unique_classes, counts = np.unique(pred_labels, return_counts=True)
            
            print("\nImproved class distribution:")
            for cls, count in zip(unique_classes, counts):
                class_name = LEARNING_IDX_TO_NAME.get(cls, "unknown")
                percentage = 100.0 * count / len(pred_labels)
                print(f"Class {cls} ({class_name}): {count} points ({percentage:.2f}%)")
    
    # Create a colored point cloud to visualize the segmentation
    colored_pc = o3d.geometry.PointCloud()
    colored_pc.points = o3d.utility.Vector3dVector(points)
    
    # Use a fixed color map for better visualization
    color_map = np.zeros((cfg.num_classes, 3))
    # Define distinct colors for different classes
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
    for i, color in enumerate(colors):
        if i < cfg.num_classes:
            color_map[i] = color
    
    point_colors = color_map[pred_labels]
    colored_pc.colors = o3d.utility.Vector3dVector(point_colors)
    
    print("Visualizing segmented point cloud...")
    o3d.visualization.draw_geometries([colored_pc])

    o3d.save_point_cloud("segmented_point_cloud.ply", colored_pc)
    
    return points, pred_labels


if __name__ == "__main__":
    infer_test_dataset()

