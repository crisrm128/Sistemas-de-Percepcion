
from multiprocessing import shared_memory
import open3d as o3d
from open3d import *
import numpy as np
import copy
import time

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 0, 1])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 3
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size,pcd4,object):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(object)
    target = pcd4

    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


points = []
for i in range(100):
    for j in range(100):
        points.append([i,j,0])
#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(np.array(points))

# Leer nube de puntos
pcd = o3d.io.read_point_cloud("clouds/scenes/snap_0point.pcd")

#o3d.visualization.draw_geometries([pcd])

shape = np.array(pcd.points).shape
print("Nube:", pcd)
print("Shape del tensor que contiene la imagen:", shape)
point = pcd.points[200]
print("Posición XYZ del punto:", point)
color = pcd.colors[200]
print("Color del punto:", color) # RGB en el rango [0...1]

#ELIMINAR POLANOS DOMINANTES
#Primer iteracion
plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)

[a, b, c, d] = plane_model
#print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                  zoom=0.8,
#                                  front=[-0.4999, -0.1659, -0.8499],
#                                  lookat=[2.1813, 2.0619, 2.0999],
#                                  up=[0.1204, -0.9852, 0.1215])

pcd2 = outlier_cloud
#o3d.visualization.draw_geometries([pcd2])

#Segunda iteracion
plane_model, inliers = pcd2.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)

[a, b, c, d] = plane_model
#print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd2.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 0, 1])
outlier_cloud = pcd2.select_by_index(inliers, invert=True)
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                  zoom=0.8,
#                                  front=[-0.4999, -0.1659, -0.8499],
#                                  lookat=[2.1813, 2.0619, 2.0999],
#                                  up=[0.1204, -0.9852, 0.1215])

pcd3 = outlier_cloud

#Tercera iteracion
plane_model, inliers = pcd3.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)

[a, b, c, d] = plane_model
#print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd3.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 1, 0])
outlier_cloud = pcd3.select_by_index(inliers, invert=True)
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                  zoom=0.8,
#                                  front=[-0.4999, -0.1659, -0.8499],
#                                  lookat=[2.1813, 2.0619, 2.0999],
#                                  up=[0.1204, -0.9852, 0.1215])


pcd4 = outlier_cloud
#o3d.visualization.draw_geometries([pcd4])


#Reducir numero de puntos - FILTRAR
pcd_sub = pcd4.voxel_down_sample(0.01) # Tamaño de la hoja de 0.1
#print("Shape del tensor que contiene la imagen:", np.array(pcd_sub.points).shape)
#o3d.visualization.draw_geometries([pcd_sub])

#-----------------------------------------------------------------------------------


voxel_size = 0.01  # means 5cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_piggybank_corr.pcd")

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)
draw_registration_result(source_down, pcd, result_ransac.transformation)


voxel_size = 0.01  # means 5cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_plant_corr.pcd")

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)
draw_registration_result(source_down, pcd, result_ransac.transformation)

voxel_size = 0.008  # means 5cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_mug_corr.pcd")

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)
draw_registration_result(source_down, pcd, result_ransac.transformation)

voxel_size = 0.008  # means 5cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_plc_corr.pcd")

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac)
draw_registration_result(source_down, pcd, result_ransac.transformation)