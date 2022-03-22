
shape = np.array(pcd.points).shape
print("Nube:", pcd)
print("Shape del tensor que contiene la imagen:", shape)
point = pcd.points[200]
print("Posición XYZ del punto:", point)
color = pcd.colors[200]
print("Color del punto:", color) # RGB en el rango [0...1]

#Mallas
#Traslacion
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame() 
mesh_trans = copy.deepcopy(mesh).translate((2, 2, 2), relative=False)
print("Centro del SC:", mesh.get_center())
print("Centro del SC transladado:", mesh_trans.get_center())
o3d.visualization.draw_geometries([mesh, mesh_trans])


#Rotacion
mesh_rot = copy.deepcopy(mesh).translate((4, 2, 2), relative=False)
R = mesh_rot.get_rotation_matrix_from_xyz((np.pi/2.0, 0, 0)) # 90 grados eje x
mesh_rot.rotate(R, center=(0, 0, 0)) 
# La rotacion se ejecuta con respecto del 0,0,0, no con respecto al propio objeto
o3d.visualization.draw_geometries([mesh, mesh_rot])

#Redecir numero de puntos
pcd_sub = pcd.voxel_down_sample(0.1) # Tamaño de la hoja de 0.1
print("Shape del tensor que contiene la imagen:", np.array(pcd_sub.points).shape)
o3d.visualization.draw_geometries([pcd_sub])

#Buscar punto mas cercano con KDTree
pcd_tree = o3d.geometry.KDTreeFlann(pcd_sub)
p = [-0.76023054, -0.63303238, 1.55300009]
[k, idx, _] = pcd_tree.search_knn_vector_3d(p, 100)
# k es un entero que indica como de cerca esta el vecino
# idx es la posicion del vecino dentro de la lista de puntos de la nube
#Pintar de color la nube de puntos
tin = time.time()
np.asarray(pcd_sub.colors)[idx[1:], :] = [0, 0, 1] # Los pinto de azul
o3d.visualization.draw_geometries([pcd_sub])
tfin = time.time()
print("Vecino mas cercano:", k)

#----------------------------------
new_points = []
new_colors = []

for x in range(shape[0]):
    b = pcd.colors[x][0]
    g = pcd.colors[x][1]
    r = pcd.colors[x][2]

    
    if (r<=1 and r>0.4 and g<=1 and g>0.4 and b<0.4):
        np.append(new_points, pcd.points[x])
        np.append(new_colors,pcd.colors[x])

pcd2 = o3d.t.geometry.PointCloud(new_points,new_colors)

#--------------------------------------------------------------


# Compute ISS Keypoints on IMAGEN

keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd4,
                                                        salient_radius=0.004,
                                                        non_max_radius=0.005,
                                                        gamma_21=0.5,
                                                        gamma_32=0.5)

#keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([keypoints])


# Compute ISS Keypoints on HUCHA

pcd_banck = o3d.io.read_point_cloud("clouds/objects/s0_piggybank_corr.pcd")

keypoints_banck = o3d.geometry.keypoint.compute_iss_keypoints(pcd_banck,
                                                        salient_radius=0.005,
                                                        non_max_radius=0.005,
                                                        gamma_21=0.5,
                                                        gamma_32=0.5)
#keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([keypoints_banck])

# Compute ISS Keypoints on FLORERO

pcd_plant = o3d.io.read_point_cloud("clouds/objects/s0_plant_corr.pcd")

keypoints_plant = o3d.geometry.keypoint.compute_iss_keypoints(pcd_plant,
                                                        salient_radius=0.005,
                                                        non_max_radius=0.005,
                                                        gamma_21=0.5,
                                                        gamma_32=0.5)
#keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([keypoints_plant])

# Compute ISS Keypoints on TAZA

pcd_mug = o3d.io.read_point_cloud("clouds/objects/s0_mug_corr.pcd")

keypoints_mug = o3d.geometry.keypoint.compute_iss_keypoints(pcd_mug,
                                                        salient_radius=0.005,
                                                        non_max_radius=0.005,
                                                        gamma_21=0.5,
                                                        gamma_32=0.5)

#keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([keypoints_mug])

# Compute ISS Keypoints on PLC

pcd_plc = o3d.io.read_point_cloud("clouds/objects/s0_plc_corr.pcd")

keypoints_plc = o3d.geometry.keypoint.compute_iss_keypoints(pcd_plc,
                                                        salient_radius=0.005,
                                                        non_max_radius=0.005,
                                                        gamma_21=0.5,
                                                        gamma_32=0.5)

#keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([keypoints_plc])