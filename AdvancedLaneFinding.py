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
        self.M, self.invM = self.__get_warp_matrices(self.src_points, self.dst_points)

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

    def warp(self, image):
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, self.M, img_size)
        return warped

    def unwarp(self, image):
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, self.invM, img_size)
        return warped

    def get_undistort(self, image):
        mtx, dist = self.camera_calibator.get_calibration()
        return cv2.undistort(image, mtx, dist, None, mtx)

    def get_binary_image(self, image):
        binary_image = self.__mag_threshold(image, thresh=(25, 200))
        return binary_image


class LaneFinder:

    def __init__(self, image_processor):
        self.image_processor = image_processor

    def process(self, image):
        undist_img = image_processor.get_undistort(image)
        birds_eye_img = image_processor.warp(undist_img)
        binary_img = image_processor.get_binary_image(birds_eye_img)
        left_fit, right_fit = self.__find_lane_points(binary_img, nwindows=9, margin=100, minpix=50)
        lane_overlay = self.__draw_lane_lines(birds_eye_img, left_fit, right_fit)
        lane_overlay = image_processor.unwarp(lane_overlay)
        output_img = self.__combinbe_images(undist_img, lane_overlay)
        return output_img

    def __poly_it(self, p, x):
        f = np.poly1d(p)
        return f(x)

    def __draw_lane_lines(self, image, left_fit, right_fit, color=(0, 255, 0), thickness=10):
        color_warp = np.zeros_like(image).astype(np.uint8)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

        if left_fit is not None:
            left_fitx = self.__poly_it(left_fit, ploty)

        if right_fit is not None:
            right_fitx = self.__poly_it(right_fit, ploty)

        if left_fit is not None and right_fit is not None:
            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            cv2.fillPoly(color_warp, np.int_([pts]), color)
        elif left_fit is not None:
            points = np.stack((left_fitx, ploty), axis=1).astype(np.int32)
            cv2.polylines(color_warp, [points], False, color, thickness=thickness)
        elif right_fit is not None:
            points = np.stack((right_fitx, ploty), axis=1).astype(np.int32)
            cv2.polylines(color_warp, [points], False, color, thickness=thickness)

        return color_warp

    def __draw_object_points(self, image, points, color=(255, 0, 0), thickness=10):
        cv2.line(image, tuple(points[0]), tuple(points[1]), color, thickness)
        cv2.line(image, tuple(points[1]), tuple(points[2]), color, thickness)
        cv2.line(image, tuple(points[2]), tuple(points[3]), color, thickness)
        cv2.line(image, tuple(points[3]), tuple(points[0]), color, thickness)

    def __combinbe_images(self, image1, image2):
        return cv2.addWeighted(image1, 1, image2, 1, 0)

    # HYPERPARAMETERS
    # nwindows : number of sliding windows
    # margin   : width of the windows +/- margin
    # minpix   : minmum number of pixels found to recenter window
    def __find_lane_pixels(self, binary_warped, nwindows=9, margin=100, minpix=50):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            ### Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            ### Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            ### If you found > minpix pixels, recenter next window ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def __find_lane_points(self, binary_warped, nwindows, margin, minpix):
        # Find our lane pixels first
        leftx, lefty, rightx, righty = self.__find_lane_pixels(binary_warped, nwindows, margin, minpix)

        ### Fit a second order polynomial to each using `np.polyfit` ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit


image = mpimg.imread(os.path.join('test_images', 'test6.jpg'))

image_processor = ImageProcessor(CameraCalibator())
lane_finder = LaneFinder(image_processor)
image = lane_finder.process(image)

mpimg.imsave('test2.jpg', image)

plt.imshow(image, cmap="gray")
plt.show()
