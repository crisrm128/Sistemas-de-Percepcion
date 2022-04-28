from multiprocessing import shared_memory
from tracemalloc import start
import open3d as o3d
from open3d import *
import numpy as np
import copy
import time

one = 0
total_time = 0

#---------------------------------------------- DIBUJAR RESULTADO ---------------------------------------------------


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



#-------------------------------------------------- KEYPOINTS --------------------------------------------------------


#def get_keypopints_iss_source(pcd):

#    pcd_keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
 #                                                       salient_radius=0.004,
 #                                                       non_max_radius=0.006,
 #                                                       gamma_21=0.5,
 #                                                       gamma_32=0.7)


#    return pcd_keypoints


def get_keypopints_iss_target(pcd):

    pcd_keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd,
                                                        salient_radius=0.004,
                                                        non_max_radius=0.006,
                                                        gamma_21=0.5,
                                                        gamma_32=0.7)


    return pcd_keypoints

#-------------------------------------------------- DESCRIPTORES -----------------------------------------------------

def get_features_fpfh(pcd_down,pcd_keypoints,voxel_size): #Pasas la lista de keypoints como pcd_down y tamaño de voxel

    start = time.time()
    radius_normal = voxel_size * 3
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #print(pcd_down.normals.shape)

    #kp_normals = np.zeros(np.array(pcd_keypoints.points).shape[0])
    # From numpy to Open3D
    #pcd_keypoints.normals = open3d.utility.Vector3dVector(kp_normals)

    #print(np.array(pcd_keypoints.points).shape[0])
    #print(np.array(pcd_down.points).shape[0])

    for i in range(np.array(pcd_down.points).shape[0]):
        for j in range(np.array(pcd_keypoints.points).shape[0]):
            if pcd_down.points[i][0] == pcd_keypoints.points[j][0] and pcd_down.points[i][1] == pcd_keypoints.points[j][1] and pcd_down.points[i][2] == pcd_keypoints.points[j][2]:
                pcd_keypoints.normals.insert(j, pcd_down.normals[i])
                #print(np.array(pcd_keypoints.normals).shape)              

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_keypoints,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    shape_kp = np.array(pcd_fpfh)
    points_kp = shape_kp

    end = time.time()
    time_keypoints = end-start


    return pcd_keypoints, pcd_fpfh, time_keypoints, points_kp

#-------------------------------------------------- FILTRADO --------------------------------------------------------


def filtering(pcd,voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)

    shape_pre = np.array(pcd.points).shape
    pre_points = shape_pre[0]

    start = time.time()
    pcd_down = pcd.voxel_down_sample(voxel_size)
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
    source_keypoints = get_keypopints_iss_target(source)
    o3d.visualization.draw_geometries([source_keypoints])
    target_keypoints = get_keypopints_iss_target(target)
    o3d.visualization.draw_geometries([target_keypoints])

    source_down, source_fpfh, source_time_keypoints, source_points_kp = get_features_fpfh(source,source_keypoints,voxel_size)
    target_down, target_fpfh, target_time_keypoints, target_points_kp = get_features_fpfh(target,target_keypoints,voxel_size)

    #print("Descriptores terminados")

    global total_time
    #total_time = total_time + (time_source_filter) + (time_target_filter)
    global one
    if(one==0):
        one = 1
    #    print(f"Points of scene before filter: {pre_target_points}; Points of scene after filter: {post_target_points}.")
    #    print(f"Time for filter of scene: {time_source_filter}.")
        print("")
        #print(f"Keypoints of scene: {points_source_kp}.")
        #print(f"Time for keypoints of scene: {time_source_keypoints}.")

    #print(f"Ponts of {object} before filter: {pre_source_points}; Points of {object} after filter: {post_source_points}.")
    #print(f"Time for filter of {object}: {time_target_filter}.")
    print("")
    #print(f"Keypoints of {object}: {points_target_kp}.")
    #print(f"Time for keypoints of {object}: {time_target_keypoints}.")

    return source, target, source_down, target_down, source_fpfh, target_fpfh, total_time

#--------------------------------------------------------------------------------------------------------------------

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 10
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
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

#----------------------------------------------------COMIENZO DEL PROGRAMA-----------------------------------------------------


# Leer nube de puntos
pcd = o3d.io.read_point_cloud("clouds/scenes/snap_0point.pcd")

#o3d.visualization.draw_geometries([pcd])

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
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                  zoom=0.8,
#                                  front=[-0.4999, -0.1659, -0.8499],
#                                  lookat=[2.1813, 2.0619, 2.0999],
#                                  up=[0.1204, -0.9852, 0.1215])

shape_in = np.array(inlier_cloud.points).shape
shape_out = np.array(outlier_cloud.points).shape

print(f"Puntos eliminados 1 : {shape_in[0]} y Puntos seleccionados 1: {shape_out[0]}")

total_eliminated = shape_in[0]

pcd2 = outlier_cloud
#o3d.visualization.draw_geometries([pcd2])

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
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                  zoom=0.8,
#                                  front=[-0.4999, -0.1659, -0.8499],
#                                  lookat=[2.1813, 2.0619, 2.0999],
#                                  up=[0.1204, -0.9852, 0.1215])

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
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
#                                  zoom=0.8,
#                                  front=[-0.4999, -0.1659, -0.8499],
#                                  lookat=[2.1813, 2.0619, 2.0999],
#                                  up=[0.1204, -0.9852, 0.1215])

shape_in = np.array(inlier_cloud.points).shape
shape_out = np.array(outlier_cloud.points).shape

print(f"Puntos eliminados 3 : {shape_in[0]} y Puntos seleccionados 3: {shape_out[0]}")

total_eliminated = total_eliminated + shape_in[0]

#Resultado final:
pcd4 = outlier_cloud

end = time.time()
time3 = end-start

o3d.visualization.draw_geometries([pcd4])

print(f"Puntos totales eliminados: {total_eliminated}; Puntos resultantes: {shape_out[0]}")
print(f"Tiempo plano fondo: {time1}; Timepo plano lateral: {time2}; Tiempo plano mesa: {time3}; Tiempo total eliminación planos: {time1+time2+time3}")

#print("Shape del tensor que contiene la imagen:", np.array(pcd_sub.points).shape)

#FILTRADO - KEYPOINTS - CALCULO DE NORMALES - MATCHING

voxel_size = 0.002  # means 1cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh, total_time] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_piggybank_corr.pcd")

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
#print(result_ransac)
draw_registration_result(source_down, pcd, result_ransac.transformation)


voxel_size = 0.002  # means cm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh,total_time] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_plant_corr.pcd")

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
#print(result_ransac)
draw_registration_result(source_down, pcd, result_ransac.transformation)

voxel_size = 0.002  # means 8mm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh,total_time] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_mug_corr.pcd")

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
#print(result_ransac)
draw_registration_result(source_down, pcd, result_ransac.transformation)

voxel_size = 0.002  # means 8mm for this dataset
[source, target, source_down, target_down, source_fpfh, target_fpfh,total_time] = prepare_dataset(voxel_size,pcd4,"clouds/objects/s0_plc_corr.pcd")

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
#print(result_ransac)
draw_registration_result(source_down, pcd, result_ransac.transformation)

print(f"Total time of filtering: {total_time}")