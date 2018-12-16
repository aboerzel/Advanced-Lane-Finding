import glob
import os
import pickle
import argparse
from collections import deque

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
                print("Unable to find appropriate number of corners on {0}".format(filename))

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
        self.src_points = self._get_src_points(self.image_size)
        self.dst_points = self._get_dst_points(self.image_size)
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

    @staticmethod
    def _abs_sobel_threshold(image, orient='x', thresh=(0, 255)):
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

    @staticmethod
    def _mag_threshold(image, kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))

        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    @staticmethod
    def _dir_threshold(image, kernel=3, thresh=(0, np.pi / 2)):
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

    def undistort(self, image):
        mtx, dist = self.camera_calibator.get_calibration()
        return cv2.undistort(image, mtx, dist, None, mtx)

    def get_binary_image(self, image):
        binary_image = self._mag_threshold(image, thresh=(25, 200))
        return binary_image


class Line:

    def __init__(self, name, n=10):
        self.name = name
        # Was the line found in the previous frame?
        self.detected = False

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        #  polynomial coefficients for the most recent fit
        self.last_fit = None

        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=n)
        # Remember previous x intercept to compare against current one
        self.last_x_int = None

        # Remember radius of curvature
        self.radius = None

        # Count the number of frames
        self.count = 0


class LaneFinder:

    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.leftLine = Line("left")
        self.rightLine = Line("right")
        self.ym_per_pix = 3 / 88  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 630  # meters per pixel in x dimension

    def process(self, image):
        undist_img = image_processor.undistort(image)
        birds_eye_img = image_processor.warp(undist_img)
        binary_img = image_processor.get_binary_image(birds_eye_img)
        self._find_lane_points(binary_img, nwindows=9, margin=100, minpix=50)
        lane_overlay = self._draw_lane(birds_eye_img)
        lane_overlay = image_processor.unwarp(lane_overlay)
        output_img = self._combinbe_images(undist_img, lane_overlay)
        self._draw_curvature_and_position(output_img)
        return output_img

    @staticmethod
    def _combinbe_images(image1, image2):
        return cv2.addWeighted(image1, 1, image2, 0.3, 0)

    # HYPERPARAMETERS
    # nwindows : number of sliding windows
    # margin   : width of the windows +/- margin
    # minpix   : minmum number of pixels found to recenter window
    @staticmethod
    def _blind_search(binary_warped, x_base, nwindows=9, margin=100, minpix=50):
        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated later for each window in nwindows
        x_current = x_base

        # Create empty lists to receive lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            ### Find the four below boundaries of the window ###
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            ### Identify the nonzero pixels in x and y within the window ###
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the list
            lane_inds.append(good_inds)

            ### If you found > minpix pixels, recenter next window ###
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            lane_inds = np.concatenate(lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        return x, y, np.sum(x) > 0

    # HYPERPARAMETER
    # margin : width of the margin around the previous polynomial to search
    @staticmethod
    def _search_around_poly(binary_warped, last_fit, margin=100):
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        lane_inds = ((nonzerox > (last_fit[0] * (nonzeroy ** 2) + last_fit[1] * nonzeroy + last_fit[2] - margin)) &
                     (nonzerox < (last_fit[0] * (nonzeroy ** 2) + last_fit[1] * nonzeroy + last_fit[2] + margin)))

        # extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        return x, y, np.sum(x) > 0

    @staticmethod
    def _get_intercepts(polynomial, image_height):
        bottom = polynomial[0] * image_height ** 2 + polynomial[1] * image_height + polynomial[2]
        top = polynomial[2]
        return bottom, top

    @staticmethod
    def _sort_by_y_vals(xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]
        return sorted_xvals, sorted_yvals

    def _get_curve_radius(self, xvals, yvals):
        fit_cr = np.polyfit(yvals * self.ym_per_pix, xvals * self.xm_per_pix, 2)
        radius = ((1 + (2 * fit_cr[0] * np.max(yvals) + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
        return radius

    def _find_lane_points(self, binary_warped, nwindows, margin, minpix):

        if self.leftLine.detected:
            leftx, lefty, self.leftLine.detected = self._search_around_poly(binary_warped, self.leftLine.last_fit,
                                                                            margin)

        if self.rightLine.detected:
            rightx, righty, self.rightLine.detected = self._search_around_poly(binary_warped, self.rightLine.last_fit,
                                                                               margin)

        if not self.leftLine.detected or not self.rightLine.detected:
            # Take a histogram of the bottom half of the image
            histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

            # Find the peak of the left and right halves of the histogram
            # These will be the starting point for the left and right lines
            midpoint = np.int(histogram.shape[0] // 2)
            left_half = histogram[:midpoint]
            right_half = histogram[midpoint:]

            self.leftLine.x_base = np.argmax(left_half)
            leftx, lefty, self.leftLine.detected = self._blind_search(
                binary_warped, self.leftLine.x_base, nwindows, margin, minpix)

            self.rightLine.x_base = np.argmax(right_half) + midpoint
            rightx, righty, self.rightLine.detected = self._blind_search(
                binary_warped, self.rightLine.x_base, nwindows, margin, minpix)

        if not self.leftLine.detected or not self.rightLine.detected:
            return  # skip if no lines detected

        lefty = np.array(lefty).astype(np.float32)
        leftx = np.array(leftx).astype(np.float32)
        righty = np.array(righty).astype(np.float32)
        rightx = np.array(rightx).astype(np.float32)

        # Calculate polynomial fit based on detected pixels
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Calculate intercepts to extend the polynomial to the top and bottom of warped image
        left_x_bottom, left_x_top = self._get_intercepts(left_fit, binary_warped.shape[0])
        right_x_bottom, right_x_top = self._get_intercepts(right_fit, binary_warped.shape[0])

        # Average intercepts across n frames
        self.leftLine.x_int.append(left_x_bottom)
        self.leftLine.last_x_int = np.mean(self.leftLine.x_int)

        self.rightLine.x_int.append(right_x_bottom)
        self.rightLine.last_x_int = np.mean(self.rightLine.x_int)

        # Add averaged intercepts to current x and y vals
        leftx = np.append(leftx, left_x_bottom)
        leftx = np.append(leftx, left_x_top)
        lefty = np.append(lefty, 0)  # y value for x_bottom
        lefty = np.append(lefty, binary_warped.shape[0])  # y value for x_top

        rightx = np.append(rightx, right_x_bottom)
        rightx = np.append(rightx, right_x_top)
        righty = np.append(righty, 0)  # y value for x_bottom
        righty = np.append(righty, binary_warped.shape[0])  # y value for x_top

        # Sort detected pixels by y values
        leftx, lefty = self._sort_by_y_vals(leftx, lefty)
        rightx, righty = self._sort_by_y_vals(rightx, righty)

        self.leftLine.X = leftx
        self.leftLine.Y = lefty

        self.rightLine.X = rightx
        self.rightLine.Y = righty

        # Recalculate polynomial with intercepts and average across n frames
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Fit polynomial to detected pixels
        self.leftLine.last_fit = left_fit
        self.rightLine.last_fit = right_fit

        # Compute radius of curvature for each lane in meters
        self.leftLine.radius = self._get_curve_radius(leftx, lefty)
        self.rightLine.radius = self._get_curve_radius(rightx, righty)

        self.leftLine.count += 1
        self.rightLine.count += 1

    def _draw_lane(self, image, color=(0, 255, 0)):
        color_warp = np.zeros_like(image).astype(np.uint8)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = self.leftLine.last_fit[0] * ploty ** 2 + self.leftLine.last_fit[1] * ploty + \
                    self.leftLine.last_fit[2]
        rigth_fitx = self.rightLine.last_fit[0] * ploty ** 2 + self.rightLine.last_fit[1] * ploty + \
                     self.rightLine.last_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rigth_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), color)

        return color_warp

    def _draw_curvature_and_position(self, image):
        # Calculate the vehicle position relative to the center of the lane
        position = (self.leftLine.last_x_int + self.rightLine.last_x_int) / 2
        distance_from_center = abs((image.shape[1] / 2 - position) * self.xm_per_pix)

        # Calculate curve radius from left curve radius and right curve radius
        cureve_radius = int((self.leftLine.radius + self.rightLine.radius) / 2)

        lane_position = "Lane Position: {:.2f}m".format(distance_from_center)
        lane_curvature = "Lane Curvature Radius: {:.2f}m".format(cureve_radius)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, lane_position, (30, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(image, lane_curvature, (30, 80), font, 1, (255, 255, 255), 2)


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
