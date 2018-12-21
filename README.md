# Advanced Lane Finding Project
---
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Calibration1"
[image2]: ./output_images/undist_calibration1.jpg "Undist_Calibration1"
[image3]: ./output_images/original.jpg "Original"
[image4]: ./output_images/undistort.jpg "Undistorted"
[image5]: ./output_images/binary.jpg "Binary Example"
[image6]: ./output_images/roi_binary.jpg "ROI Binary"
[image7]: ./output_images/roi_warped_binary.jpg "ROI Warped Binary"
[image8]: ./output_images/binary_warped_sliding_window.jpg "Sliding Window"
[image9]: ./output_images/histogram.jpg "Histogramm"
[image10]: ./output_images/binary_warped_search_around_poly.jpg "Search Arounf Poly"
[image11]: ./output_images/final_image.jpg "Final Image"

[video1]: ./output_videos/project_video.mp4 "Video"

---

## Writeup / README

### Solution and Results

* The whole source source code of my solution is located in the file [AdvancedLaneFinding.py](AdvancedLaneFinding.py)

#####Usage:

To find lanes in an image file use the following command:\
```python AdvancedLaneFinding.py -i test1.jpj``` \
The result image will be shown an stored in `output_images` (with same name)

To find lanes in an video file use the following command:\
```python AdvancedLaneFinding.py -v project_video.mp4```  \
The result image will be stored in `output_videos` (with same name)

* The solution can be found in [AdvancedLaneFinding.ipynb](AdvancedLaneFinding.ipynb)


###My solution consists of the following steps:

### Camera Calibration

The code for this step is contained in the class `CameraCalibator`.  

The `CameraCalibator` uses the chessboard images in `camera_cal/*.jpg` to calculate the distortion coefficients. 

For technical reasons, every camera delivers images with some distortions. These distortions are systematic errors that are the same for every image taken with the same camera. Thus, the distortion in each single image can be corrected, if the deviation between the image supplied by the camera and the correct image without distortion is known.
To determine this difference, some pictures of a known chessboard pattern from different perspectives are made with the camera. For each of these images, the crossing points can be determined using the `cv2.findChessboardCorners()` function.
With the obtained crossing points and the ideal crossing points (known from the chessboard pattern), the `cv2.calibrateCamera()` function can be used to calculate the calibration coefficients for this camera.
Once you know the calibration coefficients of a camera, any image taken with this camera can be corrected using the `cv2.undistort()` function.

Distored chessboard image, taken from the camera
![alt text][image1]


Unistored chessboard image, corrected using the calibration coefficients determined with the `CameraCalibator`
![alt text][image2]

### Pipeline (single image)

The code for the image transformations is located in the class `ImageProcessor`. 

The image pipeline will be demonstrated with the sample image 'test5.jpg`
![alt text][image3]

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
def get_src_points(image_size):
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

def get_dst_points(image_size):
    height, width = image_size
    margin = width / 4
    dst = np.float32([[margin, height],
                      [margin, 0],
                      [width - margin, 0],
                      [width - margin, height]])
    return dst
    
# get points for perspective transformation
src = get_src_points(binary_image.shape)
dst = get_dst_points(binary_image.shape)  
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]

Warped Binary...
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

Sliding Window Search
![alt text][image8]

Histogram

![alt text][image9]

Search around last fit
![alt text][image10]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

```python
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
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image11]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
