from multiprocessing import shared_memory
from tracemalloc import start
import open3d as o3d
from open3d import *
import numpy as np
import copy
import time

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

one = 0
total_time = 0

#---------------------------------------------- DIBUJAR RESULTADO ---------------------------------------------------


def draw_registration_result(source, target, transformation,tree):
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

    tam = len(source_temp.points)
    total_error=0
    for i in range(tam):
        p = source_temp.points[i]
        [k, idx, d] = tree.search_knn_vector_3d(p, 1)
        total_error = total_error + d[0]
        #print(idx)

    return total_error/float(tam) 
#-------------------------------------------------- KEYPOINTS --------------------------------------------------------


def get_keypopints_iss(pcd):

    start = time.time()       

    pcd_keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                        salient_radius=0.005,
                                                        non_max_radius=0.0065,
                                                        gamma_21=0.45,
                                                        gamma_32=0.5)

    end = time.time()

    return pcd_keypoints, end-start

#-------------------------------------------------- NORMALES --------------------------------------------------------

def estimate_normals(pcd, voxel_size):
    
    start = time.time() 

    radius_normal = voxel_size * 3
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    end = time.time()

    time_normals = end-start

    return pcd, time_normals


#-------------------------------------------------- DESCRIPTORES -----------------------------------------------------

def get_features_fpfh(pcd_keypoints,voxel_size): #Pasas la lista de keypoints como pcd_down y tamaño de voxel

    start = time.time()       

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_keypoints,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    end = time.time()

    return pcd_keypoints, pcd_fpfh, end-start

#-------------------------------------------------- FILTRADO ---------------------------------------------------------


def filtering(pcd,voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)

    shape_pre = np.array(pcd.points).shape
    pre_points = shape_pre[0]

    start = time.time()
    pcd_down = pcd.voxel_down_sample(0.01)
    #pcd_down = pcd.uniform_down_sample(every_k_points=5)
    end = time.time()
    time_filter = end-start

    shape_post = np.array(pcd_down.points).shape
    post_points = shape_post[0]


    return pcd_down, pre_points, post_points, time_filter

#--------------------------------------------------------------------------------------------------------------------

def prepare_dataset(voxel_size,pcd4,object):
    #print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(object)
    target = pcd4

    #draw_registration_result(source, target, np.identity(4))

    source, pre_source_points, post_source_points, time_source_filter = filtering(source, voxel_size)
    o3d.visualization.draw_geometries([source])
    target,  pre_target_points, post_target_points,time_target_filter = filtering(target, voxel_size)
    o3d.visualization.draw_geometries([target])

    print(f"Número de puntos de {object} antes de filtro: {pre_source_points} y después de filtro: {post_source_points}. Ha tardado: {time_source_filter} s.")
    print(f"Número de puntos de la escena antes de filtro: {pre_target_points} y después de filtro: {post_target_points}. Ha tardado: {time_target_filter} s.\n")

    source, time_source_normals = estimate_normals(source, voxel_size)
    o3d.visualization.draw_geometries([source])
    target, time_target_normals = estimate_normals(target,voxel_size)
    o3d.visualization.draw_geometries([target])

    print(f"Tiempo de obtención de normales de {object}: {time_source_normals} s.")
    print(f"Tiempo de obtención de normales de la escena: {time_target_normals} s.\n")

    source_keypoints, time_source_keypoints = get_keypopints_iss(source)
    o3d.visualization.draw_geometries([source_keypoints])
    target_keypoints, time_target_keypoint = get_keypopints_iss(target)
    o3d.visualization.draw_geometries([target_keypoints])

    shape_source = np.array(source_keypoints.points).shape
    shape_target = np.array(target_keypoints.points).shape

    print(f"Keypoints del {object}: {shape_source[0]} y se ha tardado: {time_source_keypoints} s.")
    print(f"Keypoints de la escena: {shape_target[0]} y se ha tardado: {time_target_keypoint} s.\n")

    source_down, source_fpfh, source_time_fpfh = get_features_fpfh(source_keypoints,voxel_size)
    target_down, target_fpfh, target_time_fpfh = get_features_fpfh(target_keypoints,voxel_size)

    print(f"Tiempo de obtención de los descriptores de {object}: {source_time_fpfh} s.")
    print(f"Tiempo de obtención de los descriptores de la escena: {target_time_fpfh} s.\n")

    return source, target, source_down, target_down, source_fpfh, target_fpfh, total_time

#-------------------------------------------------- TRANSFORMACIÓN -------------------------------------------------------------

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    start = time.time()
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000000, 0.999))

    end = time.time()

    return result, end-start

def refine_registration(source, target, voxel_size):
    start = time.time()

    distance_threshold = voxel_size * 0.2
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    end = time.time()
    return result, end-start

#----------------------------------------------------COMIENZO DEL PROGRAMA-----------------------------------------------------

total_start = time.time()

# Leer nube de puntos
pcd = o3d.io.read_point_cloud("clouds/scenes/snap_0point.pcd")

o3d.visualization.draw_geometries([pcd])

#ELIMINAR POLANOS DOMINANTES
#Primera iteracion

start = time.time()
plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)

[a, b, c, d] = plane_model
#print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                 zoom=0.8,
                                 front=[-0.4999, -0.1659, -0.8499],
                                 lookat=[2.1813, 2.0619, 2.0999],
                                 up=[0.1204, -0.9852, 0.1215])

shape_in = np.array(inlier_cloud.points).shape
shape_out = np.array(outlier_cloud.points).shape

print(f"Puntos eliminados 1 : {shape_in[0]} y Puntos seleccionados 1: {shape_out[0]}")

total_eliminated = shape_in[0]

pcd2 = outlier_cloud
o3d.visualization.draw_geometries([pcd2])

end = time.time()

time1 = end-start

#Segunda iteracion

start = time.time()
plane_model, inliers = pcd2.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)

[a, b, c, d] = plane_model
#print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd2.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 0, 1])
outlier_cloud = pcd2.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                 zoom=0.8,
                                 front=[-0.4999, -0.1659, -0.8499],
                                 lookat=[2.1813, 2.0619, 2.0999],
                                 up=[0.1204, -0.9852, 0.1215])

shape_in = np.array(inlier_cloud.points).shape
shape_out = np.array(outlier_cloud.points).shape

print(f"Puntos eliminados 2 : {shape_in[0]} y Puntos seleccionados 2: {shape_out[0]}")

total_eliminated = total_eliminated + shape_in[0]

pcd3 = outlier_cloud

end = time.time()
time2 = end-start

#Tercera iteracion

start = time.time()
plane_model, inliers = pcd3.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)

[a, b, c, d] = plane_model
#print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd3.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 1, 0])
outlier_cloud = pcd3.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                 zoom=0.8,
                                 front=[-0.4999, -0.1659, -0.8499],
                                 lookat=[2.1813, 2.0619, 2.0999],
                                 up=[0.1204, -0.9852, 0.1215])

shape_in = np.array(inlier_cloud.points).shape
shape_out = np.array(outlier_cloud.points).shape

print(f"Puntos eliminados 3 : {shape_in[0]} y Puntos seleccionados 3: {shape_out[0]}\n")

total_eliminated = total_eliminated + shape_in[0]

#Resultado final:
pcd4 = outlier_cloud

end = time.time()
time3 = end-start

o3d.visualization.draw_geometries([pcd4])

print(f"Puntos totales eliminados: {total_eliminated}; Puntos resultantes: {shape_out[0]}")
print(f"Tiempo plano fondo: {time1}; Timepo plano lateral: {time2}; Tiempo plano mesa: {time3}; Tiempo total eliminación planos: {time1+time2+time3}\n\n")

#FILTRADO - KEYPOINTS - CALCULO DE NORMALES - MATCHING

pcd_tree = o3d.geometry.KDTreeFlann(pcd)

print(f"---------------------------------------------------- BANK ----------------------------------------------------\n")

voxel_size = 0.002  # means 2cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh, total_time] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_piggybank_corr.pcd")

result_ransac, time_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

result_icp, time_icp = refine_registration(source, target, voxel_size)

error = draw_registration_result(source_down, pcd, result_icp.transformation,pcd_tree)

print(f"Fitness bank: {result_icp} en {time_ransac + time_icp}s.")
print(f"Error bank= {error}\n\n")

print(f"---------------------------------------------------- PLANT ----------------------------------------------------\n")

voxel_size = 0.002  # means 2cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh,total_time] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_plant_corr.pcd")

result_ransac, time_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

result_icp, time_icp = refine_registration(source, target, voxel_size)

error = draw_registration_result(source_down, pcd, result_icp.transformation,pcd_tree)

print(f"Fitness plant: {result_icp} en {time_ransac + time_icp}s.")
print(f"Error plant= {error}\n\n")

print(f"---------------------------------------------------- MUG ----------------------------------------------------\n")

voxel_size = 0.002 # means 2cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh,total_time] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_mug_corr.pcd")

result_ransac, time_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

result_icp, time_icp = refine_registration(source, target, voxel_size)

error = draw_registration_result(source_down, pcd, result_icp.transformation,pcd_tree)

print(f"Fitness mug: {result_icp} en {time_ransac + time_icp}s.")
print(f"Error mug= {error}\n\n")

print(f"---------------------------------------------------- PLC ----------------------------------------------------\n")

voxel_size = 0.00145 # means 2cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh,total_time] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_plc_corr.pcd")

result_ransac, time_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

result_icp, time_icp = refine_registration(source, target, voxel_size)

error = draw_registration_result(source_down, pcd, result_icp.transformation,pcd_tree)

print(f"Total time of filtering: {total_time}")

print(f"Fitness plc: {result_icp} en {time_ransac + time_icp}s.")
print(f"Error plc= {error}\n\n")

total_end = time.time()

print(f"Tiempo total de ejecución: {total_end-total_start}")