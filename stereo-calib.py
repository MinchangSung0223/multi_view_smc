#!/usr/bin/env python3
import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import glob
import os

import argparse
import pyrealsense2 as rs
import time
import math
import pudb
import shutil
from itertools import combinations


def setArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--stream', action='store_true')
    parser.add_argument('-c', '--calibrate', action='store_true')
    parser.add_argument('-sc', '--stereo', action='store_true')
    parser.add_argument('-bw', '--board_width', type=int, default=9)
    parser.add_argument('-bh', '--board_height', type=int, default=6)
    parser.add_argument('-bs', '--board_size', type=float, default=0.0235)
    parser.add_argument('-fh', '--frame_height', type=int, default=1080)
    parser.add_argument('-n', '--total_count', type=int, default=13)
    parser.add_argument('-d', '--selected_devices', nargs='*', help='Set of the devices you want to use')

    args = parser.parse_args()
    return args


class StereoCalibration:
    def __init__(self, board_width, board_height, board_size, frame_width, frame_height, total_count, selected_devices):
        self.board_width = board_width
        self.board_height = board_height
        self.board_size = board_size
        self.total_count = total_count
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.selected_devices = selected_devices

        self.imgpoints_l = []
        self.imgpoints_r = []   

        # Subpixel corner
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((board_width*board_height, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2) * board_size
        self.objpoints = []     # 3d point in real world space
        self.imgpoints = []     # 2d points in image plane.

        self.save_cnt = 0
        ctx = rs.context()
        self.pipeline = []
        
        # for stereo-calibrate
        self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        self.mtx = [[]]           # Camera matrix
        self.dist = [[]]          # Distortion
        self.corners2 = []

        if len(ctx.devices) > 0:
            for i in ctx.devices:
                self.pipeline.append(rs.pipeline())
                self.mtx.append([])
                self.dist.append([])
                self.corners2.append([])

            for idx_d in self.selected_devices:
                d = ctx.devices[idx_d]
                print ('Found device: ', d.get_info(rs.camera_info.name), ' ', d.get_info(rs.camera_info.serial_number))
                config = rs.config()
                config.enable_device(d.get_info(rs.camera_info.serial_number))
                config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.bgr8, 30)
                print('Frame size: ', self.frame_width, 'x', self.frame_height)
                self.pipeline[idx_d].start(config)
            self.total_pipe = len(self.pipeline)
            # for stereo-calibrate
            self.imgpoints2 = np.zeros((self.total_pipe, self.total_count, self.board_height * self.board_width, 1, 2), dtype=np.float32)
        else:
            print("No Intel Device connected")
            sys.exit(0)

    def stream(self):
        while True:
            for i in self.selected_devices:
                pipe = self.pipeline[i]
                frame = pipe.wait_for_frames()
                color_frm = frame.get_color_frame()
                color_img = np.asanyarray(color_frm.get_data())
                color_img = cv2.resize(color_img, (640, 480))
                cv2.imshow('realsense' + str(i), color_img)
                key = cv2.waitKey(1)
                if key == 27:
                    cv2.destroyAllWindows()
                    pipe.stop()
                    sys.exit(0)

    def calibrate(self):

        dir_name = 'config'
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
        os.makedirs(dir_name, exist_ok=True)

        print('Initializing..')

        for idx_pipe in self.selected_devices:
            print('---------------')
            print('Camera', str(idx_pipe), 'Calibration')
            for i in range(3):
                time.sleep(1)
                print(3-i)
            print('---------------')
            n_img = 0
            pipe = self.pipeline[idx_pipe]

            while n_img < self.total_count:
                frame = pipe.wait_for_frames()
                color_frm = frame.get_color_frame()
                color_img = np.asanyarray(color_frm.get_data())
                gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

                found_board, corners = cv2.findChessboardCorners(gray_img, (self.board_width, self.board_height), None)

                if found_board: 
                    corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(gray_img, (self.board_width, self.board_height), corners2, found_board)
                    self.imgpoints.append(corners2)
                    self.objpoints.append(self.objp)
                    gray_img = cv2.resize(gray_img, (640, 480))
                    print(n_img + 1, '/', self.total_count)
                    n_img += 1
                    cv2.imshow('cam', gray_img)
                    key = cv2.waitKey(1000)
                    if key == 27:
                        cv2.destroyAllWindows()
                        pipe.stop()
                        sys.exit(0)

            cv2.destroyAllWindows()

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, (self.frame_width, self.frame_height), None, None)

            # Re-projection error
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            print("Reprojection error: {}".format(mean_error / len(self.objpoints)))

            # Save camera parameters
            s = cv2.FileStorage('config/cam_calib_'+str(idx_pipe)+'.xml', cv2.FileStorage_WRITE)
            s.write('mtx', mtx)
            s.write('dist', dist)
            s.release()
        pipe.stop()

    def stereo_calibrate(self, folder_name):

        # Load previously saved data
        for i in self.selected_devices:
            filename = 'config/cam_calib_'+str(i)+'.xml'
            s = cv2.FileStorage()
            s.open(filename, cv2.FileStorage_READ)
            if not s.isOpened():
                print('Failed to open,', filename)
                exit(1)
            self.mtx[i] = s.getNode('mtx').mat()
            self.dist[i] = s.getNode('dist').mat()

        print('Initializing..')
        for i in range(3):
            time.sleep(1)
            print(3-i)
        print('---------------')
        n_img = 0

        while n_img < self.total_count:
            gray_img = np.zeros((self.total_pipe, self.frame_height, self.frame_width), dtype=np.uint8)
            found_board = True

            for i in self.selected_devices:
                pipe = self.pipeline[i]
                frame = pipe.wait_for_frames()
                color_frm = frame.get_color_frame()
                color_img = np.asanyarray(color_frm.get_data())
                gray_img_ = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                found_board_, corners = cv2.findChessboardCorners(gray_img_, (self.board_width, self.board_height), None)
                found_board = found_board * found_board_ # When every camera detects the chessboard, found_board will be True

                if found_board_:    # When each camera detects the chessboard
                    corners2_ = cv2.cornerSubPix(gray_img_, corners, (11, 11), (-1, -1), self.criteria)
                    cv2.drawChessboardCorners(gray_img_, (self.board_width, self.board_height), corners2_, found_board_)
                    self.corners2[i] = corners2_
                    gray_img[i, :, :] = gray_img_
                    gray_img = np.concatenate((gray_img, np.expand_dims(gray_img_, axis=0)), axis=0)

            if found_board:
                total_gray_img = np.zeros((self.frame_height, 1), dtype=np.uint8)
                for i in self.selected_devices:
                    dir_name = folder_name + "/cam_" + str(i)

                    if n_img == 0:
                        if os.path.exists(dir_name):
                            shutil.rmtree(dir_name)
                        os.makedirs(dir_name, exist_ok=True)

                    cv2.imwrite(dir_name + '/img' + f'{n_img:02}.png', gray_img[i])
                    cv2.putText(gray_img[i], 'cam'+str(i), (960, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
                    total_gray_img = np.hstack((total_gray_img, gray_img[i]))
                    self.imgpoints2[i, n_img] = self.corners2[i]

                self.objpoints.append(self.objp)
                print(n_img + 1, '/', self.total_count)
                cv2.putText(total_gray_img, str(n_img+1) + '/' + str(self.total_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 2, cv2.LINE_AA)
                total_gray_img = cv2.resize(total_gray_img, (640 * len(self.selected_devices), 480))
                cv2.imshow('stereo-calibrate', total_gray_img)
                key = cv2.waitKey(1000)
                n_img += 1

                if key == 27:
                    cv2.destroyAllWindows()
                    pipe.stop()
                    sys.exit(0)

        cv2.destroyAllWindows()
        for i in self.selected_devices:
            self.pipeline[i].stop()

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH

        # cam_combi = list(combinations(range(self.total_pipe), 2))
        # for c in cam_combi:
        # ret, M1, d1, M2, d2, R, tvec, E, F = cv2.stereoCalibrate(self.objpoints, self.imgpoints[c[0]], self.imgpoints[c[1]], self.mtx[c[0]], self.dist[c[0]], self.mtx[c[1]], self.dist[c[1]], (self.frame_width, self.frame_height), criteria=self.stereocalib_criteria, flags=flags)

        imgpoints_l = self.imgpoints2[self.selected_devices[0]]
        imgpoints_r = self.imgpoints2[self.selected_devices[1]]
        _, _, _, _, _, R, tvec, _, _ = cv2.stereoCalibrate(self.objpoints, imgpoints_l, imgpoints_r, self.mtx[self.selected_devices[0]], self.dist[self.selected_devices[0]], self.mtx[self.selected_devices[1]], self.dist[self.selected_devices[1]], (self.frame_width, self.frame_height), criteria=self.stereocalib_criteria, flags=flags)

        # R, tvec: Transformation from Cam 2 to Cam 1
        tvec = -R.T.dot(tvec)
        rotation_matrix = R.T
        eulerAngle = self.rotationMatrixToEulerAngles(rotation_matrix) 

        print('-----------------------------------------')
        print('Combination:', self.selected_devices[0], self.selected_devices[1])
        print('Translation vector(X, Y, Z) [meter]', tvec.T)
        print('Rotation matrix', rotation_matrix)
        print('Euler angle(Rx, Ry, Rz) [deg]', eulerAngle * 180 / math.pi)

        # Save camera parameters
        # s = cv2.FileStorage('config/trans'+str(c[0])+str(c[1])+'.xml', cv2.FileStorage_WRITE)     # If you want to save the file at once(using combination) ex) n=3 -> 01, 12, 02
        s = cv2.FileStorage('config/trans'+str(self.selected_devices[0])+str(self.selected_devices[1])+'.xml', cv2.FileStorage_WRITE)
        s.write('R', rotation_matrix)
        s.write('tvec', tvec)
        s.release()
        
    def rotationMatrixToEulerAngles(self, R):
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        #sy = math.sqrt(R[2, 2] * R[2, 2] +  R[2, 1] * R[2, 1])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])


if __name__ == '__main__':

    args = setArgs()

    if args.frame_height == 480:
        frame_width = 640
    elif args.frame_height == 1080:
        frame_width = 1920

    if args.selected_devices:
        selected_devices = list(map(int, args.selected_devices))
    else:
        ctx = rs.context()
        selected_devices = list(range(len(ctx.devices)))

    s = StereoCalibration(args.board_width, args.board_height, args.board_size, frame_width, args.frame_height, args.total_count, selected_devices)

    if args.stream:
        s.stream()
    elif args.calibrate:
        s.calibrate()
    elif args.stereo:
        s.stereo_calibrate('images')
    else:
        print('Need to check the option...')
