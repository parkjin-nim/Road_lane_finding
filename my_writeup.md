## Writeup

---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/test1_undistorted.jpg "Road Transformed"
[image3]: ./output_images/thresholded_binary_images.png "Binary Example"
[image4]: ./output_images/create_birdeye_binary_image.jpg "Warp Example"
[image5]: ./output_images/lane_identification.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./output_images/project_video.mp4 "Video"
[video2]: ./output_images/project_video_extra.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

--- 

All output images can be found under ./output_images directory !

### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Again, `objp` is just coordinates of 6 x 9 chessboard like ((0,0), (1,0), (2,0) ....,(8,5)), obtained by 'np.mgrid[0:9,0:6].T.reshape(-1,2)'. It has no measurement unit yet. Then, in the second sell, i then used the output `objpoints` and `imgpoints` to compute the camera calibration(mtx) and distortion coefficients using the `cv2.calibrateCamera()` function.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline

#### 1. Distortion-corrected image

Distortion correction step was done on ./test_images/test1.jpg. I apply the distortion correction to one of the test images like this one and we can observe that the back part of white car was cut:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. More specifically color `saturation` channel and `x gradient` information. The observation is that distand lanes may be well found by x gradient, and lanes under shadow may be well found by saturation information. The threshold value detected the lanes okay under both situations relatively less noise. Each threshold values are shown at the line 4 of the first cell:
```python
def create_binary_image(img, s_thresh=(170, 225), sx_thresh=(15, 100)):
```
Two thresholding steps at lines 19 through 25 in `example.ipynb`. Two binary files were combined using OR operation. Here's an example of my output for this step. Test image is test5.jpg from ./test_images directory.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_birdeyeview()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
   [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])

dst = np.float32(
   [[(img_size[0] / 4 - 30), 0],
    [(img_size[0] / 4 - 30), img_size[1]],
    [(img_size[0] * 3 / 4 + 30), img_size[1]],
    [(img_size[0] * 3 / 4 + 30), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 290, 0        | 
| 203, 720      | 290, 720      |
| 1127, 720     | 990, 720      |
| 695, 460      | 990, 0        |

Destination x points are 30 pixels wider than the initially given dst. x points. I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

With my color & gradient threshold at , I first binarized the image, and then performed perspective transform to get a bird-eye view. This order showed less noise than the other way around. 

And then, at `get_basex()`, i took the lower half of binarized bird-eye view image and captured histogram to find peaks. Two peaks play the role as the base position of bottom window. 

At `find_lane_pixels()`, the base x positions are feed in order to search lane pixels. As hyperparameters setting, # of windows is 9, search window width is around 200 pixels (margin = 100), and min. number of pixels found to recenter window is 51(minpix). Each window position is decided by the mean value of pixels distribution within the one-level lower window. So, the lost track of one lower window may lead to that of upper level window.

Combining all pixels captured within 9 windows are feed to the fitting function, `fit_polynomial()`, The numpy function, `np.polyfit(lefty, leftx, 2)` did fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

At `measure_curvature_pixels()`, i used the 1st & 2nd derivatives of quadratic polynomial model to calculate cavature. Cavature is important in deciding streering angle. The radius of it is just inverse of cavature, namely Radius = 1/Cavature.

The curvature of a given curve at a particular point is the curvature of the approximating circle at that point. The radius of curvature of the curve at a particular point is defined as the radius of the approximating circle. This radius changes as we move along the curve. The curvature depends on the radius - the smaller the radius, the greater the curvature (approaching a point at the extreme) and the larger the radius, the smaller the curvature(A very large approximating circle means the curve is almost a straight line at that point)

```python
    f_1prime = 2*left_fit[0]*y_eval + left_fit[1]
    f_2prime = 2*left_fit[0]
    left_curverad = ((1+f_1prime**2)**(1.5))/np.absolute(f_2prime)
    
    f_1prime = 2*right_fit[0]*y_eval + right_fit[1]
    f_2prime = 2*right_fit[0]
    right_curverad = ((1+f_1prime**2)**(1.5))/np.absolute(f_2prime)
```
Roadway design manual about horozontal alignment from the government says the min. curvature radius is 2195(ft) at the speed of 60(mph) on non-urban highway superelevation 6%. In meters, it is around 658 at 100 kmph. Now, at `measure_curvature_real`, the left & right curvatures of `test5.jpg` are 696m, 644m respectively. And using 3.7/700 ratio, I found the vehicle is around 0.138m left from the center.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step at the first cell of section 6. The same transform function, `cv2.getPerspectiveTransform(dst, src)` is used with reversed order of input arguments, (dst, src).
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

At `find_lane_pixels()` in section 4, i noted that each window position is decided by the mean value of pixels distribution within the one-level lower window, such that the lost track of one lower window may lead to that of upper level window. This usually happens when burst noise makes high jitter of window position. I basicallty imposed limitations on how quickly each window position may change as below. 

```python
if (new_leftx_current != leftx_current) and np.abs(new_leftx_current-leftx_current) > 30 :
    leftx_current = leftx_current + np.int((new_leftx_current - leftx_current) * 0.5)

if (new_rightx_current != rightx_current) and np.abs(new_rightx_current-rightx_current) > 30:
    rightx_current = rightx_current + np.int((new_rightx_current - rightx_current) * 0.5)

```

In addition, i applied moving averages on left,right base x coordinates. Since the base x decides initial search window depending on the peaks on the histogram, the peaks can be captured incorrectly by the noise. For this, i used two circular buffers with 20 integers long and simply took the mean value from it. 

```python
global cum_rb, cum_lb
cum_lb.append(lx_base); cum_rb.append(rx_base)

if len(cum_lb) >= 20:
    lx_base = np.sum(cum_lb[-N:]) // 20
    rx_base = np.sum(cum_rb[-N:]) // 20
```

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Like usual, parameter tunnings were very time-comsumming. De-noising issue was also a big struggling point. The empirical setting of limitaions needs to be further improved with more sophisticated de-nosing filters or calculation.
