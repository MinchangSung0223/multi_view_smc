#!/usr/bin/env python
import rospy
import tf
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CameraInfo
import sys
import numpy as np
import cv2
import argparse

from open3d_ros_helper import open3d_ros_helper as orh
import open3d as o3d
import tf2_ros
import geometry_msgs.msg as gmsg
import copy
import pudb

def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trans', help='Transformation matrix between two other cameras')
    args = parser.parse_args()
    return args


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    max_nn_neighbors = 30   # At maximum, max_nn neighbors will be searched.
    o3d.geometry.estimate_normals(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_neighbors))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 4
    ransac_n = 4
    max_iteration = 4000000
    max_validation = 10000
    corre_edgelength = 0.9  # Similarity_threshold is a number between 0 (loose) and 1 (strict)

    print("Downsampling voxel size is %.3f," % voxel_size)
    print("Distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), ransac_n, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(corre_edgelength),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(max_iteration, max_validation))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    global result_ransac
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result


def callback_cam(pc2_msg): 
    global required_ransac, cnt_ransac
    o3dpc = orh.rospc_to_o3dpc(pc2_msg)
    T_rgb_to_ir_base, T_rgb_to_ir_side = np.eye(4), np.eye(4)

    
    global rgb_to_ir_basecam, rgb_to_ir_sidecam
    #global extrinsic_depth_to_color_1, extrinsic_depth_to_color_2
    T_rgb_to_ir_base[:3, :3] = np.reshape(rgb_to_ir_basecam[:-3], (3, 3))
    T_rgb_to_ir_base[:3, 3] = rgb_to_ir_basecam[-3:]
    T_ir_to_rgb_base = tf.transformations.inverse_matrix(T_rgb_to_ir_base)

    # from Cam 0 to Cam 1
    T_btw_cameras_ = t_btw_cameras.dot(R_btw_cameras)

    T_rgb_to_ir_side[:3, :3] = np.reshape(rgb_to_ir_sidecam[:-3], (3, 3))
    T_rgb_to_ir_side[:3, 3] = rgb_to_ir_sidecam[-3:]
    #T_ir_to_rgb_side = tf.transformations.inverse_matrix(T_rgb_to_ir_sidecam)
    
    #T_btw_cameras_with_extrinsic = T_rgb_to_ir_0.dot(T_btw_cameras).dot(T_ir_to_rgb_1)
    #T_btw_cameras_with_extrinsic = T_ir_to_rgb_0.dot(T_btw_cameras).dot(T_rgb_to_ir_1)
    #T_btw_cameras_with_extrinsic = T_btw_cameras
    T_btw_cameras = T_ir_to_rgb_base.dot(T_btw_cameras_).dot(T_rgb_to_ir_side)
    o3dpc.transform(T_btw_cameras)
    #o3dpc_1.transform(T_btw_cameras_with_extrinsic)
        
    # Transform the point cloud data from cam 2 coordinate to cam 1 coordinate
    rospc = orh.o3dpc_to_rospc(o3dpc)
    rospc.header.frame_id = base_coordi
    pub.publish(rospc)
    #pub_r.publish(rospc)


if __name__ == '__main__':
    required_ransac = False
    cnt_ransac = 0

    if len(sys.argv) == 2:
        filename = sys.argv[1]
        s = cv2.FileStorage()
        s.open(filename, cv2.FileStorage_READ)
        if not s.isOpened():
            print('Failed to open,', filename)
            exit(1)
    else:
        print('Write a transformation file...(config/trans**.xml)')
        exit(1)

    print('Start!')
    base_coordi = 'cam_' + filename[-6] + '_link'
    side_cam = 'cam_' + filename[-5]
    R_btw_cameras = np.eye(4)
    t_btw_cameras = np.eye(4)
    R_btw_cameras[:3, :3] = s.getNode('R').mat()
    t_btw_cameras[:3, 3] = s.getNode('tvec').mat().reshape((3))

    rgb_to_ir_basecam = []
    rgb_to_ir_sidecam = []

    for i in range(2):
        f = open('config/rgb2ir_' + filename[-6+i] + '.txt', 'r')
        lines = f.readlines()
        for line in lines:
            extrinsic_line = line.find('[')
            if extrinsic_line != -1:
                str_ex = line[extrinsic_line+1:len(line)-2].split(',')
                for str_ex_ in str_ex:
                    if i == 0:
                        rgb_to_ir_basecam.append(float(str_ex_))
                    else:
                        rgb_to_ir_sidecam.append(float(str_ex_))
        f.close()

    rospy.init_node('transform_' + filename[-5] + '_cam_to_base_cam', disable_signals=True)
    rospy.Subscriber('/'+side_cam+'/depth/color/points', pc2.PointCloud2, callback_cam)
    pub = rospy.Publisher('/transformed_' + filename[-5], pc2.PointCloud2, queue_size=1000)
    rospy.spin()
