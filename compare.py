import numpy as np
import open3d as o3d

# Read .ply file outputted by the Shape-estimating model
input_file = "outputs/Video/smpl_scan_continuous/rendering/4000.ply"
pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
# Convert open3d format to numpy array
# Here, you have the point cloud in numpy format. 
point_cloud_in_numpy = np.asarray(pcd.points) 

#Next we extract the point cloud
breakpoint()