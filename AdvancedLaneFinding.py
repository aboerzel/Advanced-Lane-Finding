import glob
import os
import pickle
import argparse
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

    def get_undistort(self, image):
        mtx, dist = self.camera_calibator.get_calibration()
        return cv2.undistort(image, mtx, dist, None, mtx)

    def get_binary_image(self, image):
        binary_image = self._mag_threshold(image, thresh=(25, 200))
        return binary_image


class Line:

    def __init__(self, name):
        self.name = name
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.last_fitx = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.last_fit = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        # self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        # self.allx = None
        # y values for detected line pixels
        # self.ally = None

        self.last_fits = []
        self.x_base = None

        # these two are for debugging/pipeline imaging
        self.last_inds = np.array([])
        self.fail_count = 0

    def add(self, leftx, lefty, image_height):

        self.detected = False
        self.last_inds = np.array([])

        if leftx is not None and lefty is not None:
            ploty = np.linspace(0, image_height - 1, image_height)
            self.last_fit = np.polyfit(lefty, leftx, 2)
            self.last_fitx = self._poly_it(self.last_fit, ploty)
            self.last_fits.append(self.last_fit)
            self.detected = True

            if len(self.last_fits) > 5:
                self.last_fits.pop(0)

            self.best_fit = np.mean(np.array(self.last_fits), axis=0)
            self.bestx = self._poly_it(self.best_fit, ploty)

    @staticmethod
    def _poly_it(p, x):
        f = np.poly1d(p)
        return f(x)

    def incrementFailCount(self):
        self.fail_count += 1

        if self.fail_count > 20:
            # line failed too many times. resetting
            self.last_fits = []
            self.best_fit = None
            self.fail_count = 0


class LaneFinder:

    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.leftLine = Line("left")
        self.rightLine = Line("right")

    def process(self, image):
        undist_img = image_processor.get_undistort(image)
        birds_eye_img = image_processor.warp(undist_img)
        binary_img = image_processor.get_binary_image(birds_eye_img)
        self._find_lane_points(binary_img, nwindows=9, margin=100, minpix=50)
        lane_overlay = self._draw_lane_lines(birds_eye_img)
        lane_overlay = image_processor.unwarp(lane_overlay)
        output_img = self._combinbe_images(undist_img, lane_overlay)
        return output_img

    def _draw_lane_lines(self, image, color=(0, 255, 0), thickness=10):
        color_warp = np.zeros_like(image).astype(np.uint8)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])

        left_fit = self.leftLine.last_fit
        right_fit = self.rightLine.last_fit

        if left_fit is not None:
            left_fitx = self.leftLine.last_fitx

        if right_fit is not None:
            right_fitx = self.rightLine.last_fitx

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

    @staticmethod
    def _draw_object_points(image, points, color=(255, 0, 0), thickness=10):
        cv2.line(image, tuple(points[0]), tuple(points[1]), color, thickness)
        cv2.line(image, tuple(points[1]), tuple(points[2]), color, thickness)
        cv2.line(image, tuple(points[2]), tuple(points[3]), color, thickness)
        cv2.line(image, tuple(points[3]), tuple(points[0]), color, thickness)

    @staticmethod
    def _combinbe_images(image1, image2):
        return cv2.addWeighted(image1, 1, image2, 0.3, 0)

    # HYPERPARAMETERS
    # nwindows : number of sliding windows
    # margin   : width of the windows +/- margin
    # minpix   : minmum number of pixels found to recenter window
    @staticmethod
    def _find_lane_pixels(binary_warped, x_base, nwindows=9, margin=100, minpix=50):

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

            # Append these indices to the lists
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

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        return x, y

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
        left_lane_inds = ((nonzerox > (last_fit[0] * (nonzeroy ** 2) + last_fit[1] * nonzeroy + last_fit[2] - margin)) &
                          (nonzerox < (last_fit[0] * (nonzeroy ** 2) + last_fit[1] * nonzeroy + last_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        x = nonzerox[left_lane_inds]
        y = nonzeroy[left_lane_inds]
        return x, y

    def _find_lane_points(self, binary_warped, nwindows, margin, minpix):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        left_half = histogram[:midpoint]
        right_half = histogram[midpoint:]

        if not self.leftLine.detected:
            self.leftLine.x_base = np.argmax(left_half)
            leftx, lefty = self._find_lane_pixels(binary_warped, self.leftLine.x_base, nwindows, margin, minpix)
        else:
            leftx, lefty = self._search_around_poly(binary_warped, self.leftLine.last_fit, margin)

        if leftx is not None and lefty is not None:
            self.leftLine.add(leftx, lefty, binary_warped.shape[0])
        else:
            self.leftLine.incrementFailCount()

        if not self.rightLine.detected:
            self.rightLine.x_base = np.argmax(right_half) + midpoint
            rightx, righty = self._find_lane_pixels(binary_warped, self.rightLine.x_base, nwindows, margin, minpix)
        else:
            rightx, righty = self._search_around_poly(binary_warped, self.rightLine.last_fit, margin)

        if rightx is not None and righty is not None:
            self.rightLine.add(rightx, righty, binary_warped.shape[0])
        else:
            self.rightLine.incrementFailCount()


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
    print("parameter missing")
