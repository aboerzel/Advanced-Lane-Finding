import argparse
import glob
import os
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="input image")
ap.add_argument("-v", "--video", help="input video")
args = vars(ap.parse_args())


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

    @staticmethod
    def _calibrate_camera(images, grid_size, image_size):
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
                print("Unable to find chessboard corners on {0}".format(filename))

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        return mtx, dist

    def get_calibration(self):
        if self.mtx is not None:
            return self.mtx, self.dist  # calibration data already loaded

        if not os.path.exists(self.calibration_file):
            # calibrate camera and save calibration data
            images = glob.glob(os.path.join(self.camera_cal_images_folder, 'calibration*.jpg'))
            self.mtx, self.dist = self._calibrate_camera(images, self.grid_size, self.image_size)
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

        # build vertices for perspective transformation
        self.src_points = self._get_src_points(self.image_size)
        self.dst_points = self._get_dst_points(self.image_size)

        # build warp matrix and inverse warp matrix
        self.M, self.invM = self._get_warp_matrices(self.src_points, self.dst_points)

    @staticmethod
    def _get_src_points(image_size):
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

    @staticmethod
    def _get_dst_points(image_size):
        height, width = image_size
        margin = width / 4
        dst = np.float32([[margin, height],
                          [margin, 0],
                          [width - margin, 0],
                          [width - margin, height]])
        return dst

    @staticmethod
    def _get_warp_matrices(src_points, dst_points):
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        Minv = cv2.getPerspectiveTransform(dst_points, src_points)
        return M, Minv

    def warp(self, image):
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, self.M, img_size)
        return warped

    def unwarp(self, image):
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, self.invM, img_size)
        return warped

    def undistort(self, image):
        mtx, dist = self.camera_calibator.get_calibration()
        return cv2.undistort(image, mtx, dist, None, mtx)

    @staticmethod
    def abs_sobel_threshold(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        # 1) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 2) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # 4) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return sxbinary

    def get_binary_image(self, image):
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        S = hls[:, :, 2]
        L = hls[:, :, 1]

        light_mask = np.zeros_like(L)
        light_mask[(S >= 6) & (L >= 78)] = 1

        gradx_l = self.abs_sobel_threshold(L, orient='x', sobel_kernel=3, thresh=(17, 40))
        gradx_s = self.abs_sobel_threshold(S, orient='x', sobel_kernel=3, thresh=(7, 47))

        binary_image = np.zeros_like(gradx_s)
        # For some images S channel works better, while for others L channel does
        # combine binary masks
        binary_image[((gradx_l == 1) | (gradx_s == 1)) & (light_mask == 1)] = 1
        return binary_image


class Line:

    def __init__(self, name):
        self.name = name

        # x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # polynomial coefficients for previous frame
        self.last_fit = None


class LaneFinder:

    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.leftLine = Line("left")
        self.rightLine = Line("right")
        self.count = 0
        self.ym_per_pix = 3 / 88  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 630  # meters per pixel in x dimension

    def process(self, image):
        undist_img = image_processor.undistort(image)
        binary_img = image_processor.get_binary_image(undist_img)
        birds_eye_img = image_processor.warp(binary_img)

        # find lane points
        leftx, lefty, rightx, righty, detected = self._find_lane_points(
            birds_eye_img, self.leftLine.last_fit, self.rightLine.last_fit, nwindows=9, margin=100, minpix=50)

        if not detected:
            # if no points detected use values from previous frame
            leftx = self.leftLine.X
            lefty = self.leftLine.Y
            rightx = self.rightLine.X
            righty = self.rightLine.Y

        # Calculate polynomial fit based on detected pixels
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, birds_eye_img.shape[0] - 1, birds_eye_img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        rigth_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Compute radius of curvature and distance from center in meters
        curve_radius, distance_from_center = self._get_curve_radius_and_distance_from_center(
            leftx, lefty, rightx, righty, undist_img.shape)

        # Sanity check
        if self.count > 0 and not self._lanes_sanity_check(left_fitx, rigth_fitx):
            # if sanity check failed use values from previous frame
            leftx = self.leftLine.X
            lefty = self.leftLine.Y
            rightx = self.rightLine.X
            righty = self.rightLine.Y
        else:
            self.count += 1

        # fill area between lanes
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rigth_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        lanes_img = np.zeros_like(undist_img).astype(np.uint8)
        cv2.fillPoly(lanes_img, np.int_([pts]), color=(0, 255, 0))

        # Update lines
        self.leftLine.X = leftx
        self.leftLine.Y = lefty
        self.rightLine.X = rightx
        self.rightLine.Y = righty

        self.leftLine.last_fit = left_fit
        self.rightLine.last_fit = right_fit

        unwarped_lanes_img = image_processor.unwarp(lanes_img)
        output_img = self._combinbe_images(undist_img, unwarped_lanes_img)

        # print curve radius and distance from center
        lane_position = "Lane Position: {:.2f}m".format(distance_from_center)
        lane_curvature = "Lane Curvature Radius: {:d}m".format(int(round(curve_radius)))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output_img, lane_position, (30, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(output_img, lane_curvature, (30, 80), font, 1, (255, 255, 255), 2)

        return output_img

    @staticmethod
    def _combinbe_images(image1, image2):
        return cv2.addWeighted(image1, 1, image2, 0.3, 0)

    # HYPERPARAMETERS
    # nwindows : number of sliding windows
    # margin   : width of the windows +/- margin
    # minpix   : minmum number of pixels found to recenter window
    @staticmethod
    def _blind_search(binary_warped, nwindows=9, margin=100, minpix=50):

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        left_half = histogram[:midpoint]
        right_half = histogram[midpoint:]

        left_x_base = np.argmax(left_half)
        right_x_base = np.argmax(right_half) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        leftx_current = left_x_base
        rightx_current = right_x_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            # Find the four below boundaries of the window #
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window #
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        return leftx, lefty, rightx, righty, (sum(leftx) > 0 and sum(rightx) > 0)

    # HYPERPARAMETER
    # margin : width of the margin around the previous polynomial to search
    @staticmethod
    def _search_around_last_fit(binary_warped, left_fit, right_fit, margin=100):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the area of search based on activated x-values
        # within the +/- margin of our polynomial function
        # Hint: consider the window areas for the similarly named variables
        # in the previous quiz, but change the windows to our new search area
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                          (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))

        right_lane_inds = (
                (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) &
                (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # extract line pixel positions
        left_x = nonzerox[left_lane_inds]
        left_y = nonzeroy[left_lane_inds]
        right_x = nonzerox[right_lane_inds]
        right_y = nonzeroy[right_lane_inds]
        return left_x, left_y, right_x, right_y, (sum(left_x) > 0 and sum(right_x) > 0)

    def _find_lane_points(self, binary_warped, left_fit, right_fit, nwindows, margin, minpix):

        if left_fit is not None or right_fit is not None:
            leftx, lefty, rightx, righty, detected = self._search_around_last_fit(
                binary_warped, left_fit, right_fit, margin)
            if detected:
                return leftx, lefty, rightx, righty, detected

        return self._blind_search(binary_warped, nwindows, margin, minpix)

    def _get_curve_radius_and_distance_from_center(self, left_x, left_y, right_x, right_y, image_shape):
        left_fit_cr = np.polyfit(left_y * self.ym_per_pix, left_x * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_y * self.ym_per_pix, right_x * self.xm_per_pix, 2)

        left_radius_m = ((1 + (2 * left_fit_cr[0] * np.max(left_y) + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])

        right_radius_m = ((1 + (2 * right_fit_cr[0] * np.max(left_y) + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        # Calculate curve radius from left curve radius and right curve radius
        curve_radius_m = int((left_radius_m + right_radius_m) / 2)

        # Calculate bottom points for each lane
        left_fitx_bottom_m = self._get_x_at_y(left_fit_cr, image_shape[0] * self.ym_per_pix)
        right_fitx_bottom_m = self._get_x_at_y(right_fit_cr, image_shape[0] * self.ym_per_pix)

        # Calculate image center, in meters
        center_ideal_m = image_shape[1] * self.xm_per_pix / 2
        # Calculate actual center of the lane, in meters
        center_actual_m = np.mean([left_fitx_bottom_m, right_fitx_bottom_m])

        # Calculate distance from center, in meters
        distance_from_center = abs(center_ideal_m - center_actual_m)

        return curve_radius_m, distance_from_center

    @staticmethod
    def _get_x_at_y(line_fit, line_y):
        poly = np.poly1d(line_fit)
        return poly(line_y)

    @staticmethod
    def _lanes_sanity_check(left_x, right_x, parallelism=260, min_distance=300, max_distance=650):
        delta_x = []
        for i in range(len(left_x) - 1):
            delta_x.append(abs(right_x[i] - left_x[i]))
        delta_x = np.array(delta_x)

        # check horizontal lane distance
        # the horizontal distance between the two lanes should be rather constant (between min and max)
        delta_min = delta_x.min()
        delta_max = delta_x.max()
        if delta_min < min_distance or delta_max > max_distance:
            return False

        # check lane parallelism
        # if the lanes are parallel, all horizontal distances should be rather equal
        delta_x -= delta_min
        delta_max = delta_x.max()
        if delta_max > parallelism:
            return False

        return True


image_processor = ImageProcessor(CameraCalibator())
lane_finder = LaneFinder(image_processor)


def process_image(image_input, image_output):
    image = mpimg.imread(os.path.join('test_images', image_input))
    image = lane_finder.process(image)
    mpimg.imsave(os.path.join('output_images', image_output), image)
    plt.imshow(image)
    plt.show()


def process_video(video_input, video_output):
    clip = VideoFileClip(video_input)
    processed = clip.fl_image(lane_finder.process)
    processed.write_videofile(os.path.join('output_videos', video_output), audio=False)


if args['image'] is not None:
    process_image(args['image'], args['image'])
elif args['video'] is not None:
    process_video(args['video'], args['video'])
else:
    print("USAGE:")
    print("      python AdvancedLaneFinding.py -i image.jpg")
    print("      python AdvancedLaneFinding.py -v video.mp4")
