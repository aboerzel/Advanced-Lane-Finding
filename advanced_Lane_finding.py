import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from collections import deque


def read_images(imagepath):
    images = []
    image_paths = glob.glob(imagepath)
    for image_path in image_paths:
        images.append(cv2.imread(image_path))
    return images


def get_cam_calibration_points(images, nx, ny):
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    imgs = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            imgs.append(image)

    return imgs, objpoints, imgpoints


def calibrate_camera(objpoints, imgpoints, img_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def undistore(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)


def birds_eye(image):
    img_size = (image.shape[1], image.shape[0])
    p1 = (1250, 720)
    p2 = (810, 482)
    p3 = (490, 482)
    p4 = (40, 720)

    src = np.float32([list(p3), list(p2), list(p1), list(p4)])
    dst = np.float32([[0, 0], [1280, 0], [1250, 720], [40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)


    color = [0, 0, 255]  # green
    thickness = 10

    cv2.line(image, p1, p2, color, thickness)
    cv2.line(image, p2, p3, color, thickness)
    cv2.line(image, p3, p4, color, thickness)
    cv2.line(image, p4, p1, color, thickness)

    return warped, M


nx = 9
ny = 6
calibration_images = read_images('camera_cal/calibration*.jpg')
images, objpoints, imgpoints = get_cam_calibration_points(calibration_images, nx, ny)
mtx, dist = calibrate_camera(objpoints, imgpoints, images[0].shape[1::-1])

# for image, corners in zip(images, imgpoints):
#     original_img = image.copy()
#     cv2.drawChessboardCorners(image, (nx, ny), corners, True)
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
#     ax1.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
#     ax1.set_title('Original Image', fontsize=18)
#     ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     ax2.set_title('With Corners', fontsize=18)

images = read_images('test_images/test*.jpg')
# for image in images:
#     undist = undistore(image, mtx, dist)
#
#     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
#     ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     ax1.set_title('Original Image', fontsize=20)
#     ax2.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
#     ax2.set_title('Undistorted Image', fontsize=20)

for image in images:
    undist = undistore(image, mtx, dist)
    warped, _ = birds_eye(undist)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    f.tight_layout()
    ax1.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    ax1.set_title('Undistorted Image', fontsize=20)
    ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax2.set_title('Undistorted and Warped Image', fontsize=20)

plt.show()
