import os
import pickle

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque


class CameraCalibator:

    def __init__(self, camera_cal_images_folder='camera_cal',
                 grid_size=(9, 6), image_size=(720, 1280),
                 calibration_file='calibration.pkl'):
        self.camera_cal_images_folder = camera_cal_images_folder
        self.grid_size = grid_size
        self.image_size = image_size
        self.calibration_file = calibration_file
        self.mtx = None
        self.dist = None

    def __calibrate_camera(self, images, grid_size, image_size):
        objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        for filename in images:
            image = mpimg.imread(filename)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
            else:
                print("Unable to find appropriate number of corners on {0}".format(filename))

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        return mtx, dist

    def get_calibration(self):
        if self.mtx is not None:
            return self.mtx, self.dist  # calibration data already loaded

        if not os.path.exists(self.calibration_file):
            # calibrate camera and save calibration data
            images = glob.glob(os.path.join(self.camera_cal_images_folder, 'calibration*.jpg'))
            self.mtx, self.dist = self.__calibrate_camera(images, self.grid_size, self.image_size)
            pickle.dump({'mtx': self.mtx, 'dist': self.dist}, open(self.calibration_file, 'wb'))
        else:
            # load calibration data from file
            with open('calibration.pkl', mode='rb') as f:
                calibration = pickle.load(f)
                self.mtx = calibration['mtx']
                self.dist = calibration['dist']

        return self.mtx, self.dist

    def get_image_size(self):
        return self.image_size


class ImageProcessor:
    def __init__(self, camera_calibator):
        self.camera_calibator = camera_calibator
        self.image_size = self.camera_calibator.get_image_size()
        self.src_points = self.__get_src_points(self.image_size)
        self.dst_points = self.__get_dst_points(self.image_size)
        self.M, self.inv = self.__get_warp_matrices(self.src_points, self.dst_points)

    def __get_src_points(self, image_size):
        height, width = image_size
        midpoint = width // 2
        bottom = height - int(height * 0.05)  # small distance from bottom
        top_distance = int(width * .06)
        bottom_distance = int(width * .31)
        points = np.float32([(midpoint - bottom_distance, bottom),
                             (midpoint - top_distance, height - int(height * .35)),
                             (midpoint + top_distance, height - int(height * .35)),
                             (midpoint + bottom_distance, bottom)])
        return points

    def __get_dst_points(self, image_size):
        height, width = image_size
        margin = width / 4
        dst = np.float32([[margin, height],
                          [margin, 0],
                          [width - margin, 0],
                          [width - margin, height]])
        return dst

    def __get_warp_matrices(self, src_points, dst_points):
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        return M, Minv

    def __undistort(self, image):
        mtx, dist = self.camera_calibator.get_calibration()
        return cv2.undistort(image, mtx, dist, None, mtx)

    def __abs_sobel_threshold(self, image, orient='x', thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if orient == 'x':
            d = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
            d = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

        abs_d = np.absolute(d)
        scaled = np.uint8(255 * abs_d / np.max(abs_d))

        binary_output = np.zeros_like(scaled)
        binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
        return binary_output

    def __mag_threshold(self, image, kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def __dir_threshold(self, image, kernel=3, thresh=(0, np.pi / 2)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

        abs_x = np.absolute(sobelx)
        abs_y = np.absolute(sobely)

        dir = np.arctan2(abs_y, abs_x)

        binary_output = np.zeros_like(dir)
        binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
        return binary_output

    def __birds_eye(self, image):
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, self.M, img_size)
        return warped

    def __draw_object_points(self, image, color=[255, 0, 0], thickness=10):
        cv2.line(image, tuple(self.dst_points[0]), tuple(self.dst_points[1]), color, thickness)
        cv2.line(image, tuple(self.dst_points[1]), tuple(self.dst_points[2]), color, thickness)
        cv2.line(image, tuple(self.dst_points[2]), tuple(self.dst_points[3]), color, thickness)
        cv2.line(image, tuple(self.dst_points[3]), tuple(self.dst_points[0]), color, thickness)

    def preprocess(self, image):
        image = self.__undistort(image)
        image = self.__birds_eye(image)
        image = self.__mag_threshold(image, thresh=(25, 200))
        # self.__draw_object_points(image)
        return image


class LaneFinder:

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def process(self, image):
        image = image_processor.preprocess(image)
        return image


image = mpimg.imread(os.path.join('test_images', 'test2.jpg'))

image_processor = ImageProcessor(CameraCalibator())
lane_finder = LaneFinder(image_processor)
image = lane_finder.process(image)

plt.imshow(image, cmap="gray")
plt.show()
