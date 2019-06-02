---
layout: page
title: Indoor Tracking
subtitle: George Orwell knew it !
bigimg: /img/projects/indoortracking/bigimage.png
---

[Indoor Tracking](https://github.com/johan-gras/Indoor-Tracking) is an *image processing* project made in **C**.
A couple of low level algorithms are implemented *relying on few external libraries*.
The final part of the project was the implementation of *a multi-criteria tracking system*.

<div style="text-align: center;">
	<video src="/img/projects/indoortracking/video.mp4" autoplay controls loop>Indoor Tracking Video</video>
</div>

![alt text](/img/projects/indoortracking/result.gif "t")

![alt text](/img/projects/indoortracking/resultclean.gif "t")

![alt text](/img/projects/indoortracking/resultmove.gif "t")

![alt text](/img/projects/indoortracking/resultregion.gif "t")


![alt text](/img/projects/indoortracking/harison.ppm "t")

## Image processing algorithms
### Mathematical morphology operators
*Erosion, dilation, opening and closing*, they are techniques for the analysis and processing of *geometrical structures* (read: discrete-binary images) based on *the theory of ensemble*.

### Movement detection
Two techniques, one on *temporal difference* of images and the second, on a difference *with a reference*.
More concretely, the first method is an absolute difference of images at the time t and t-1, then *threshold is applied* to detect the presence of movement.
The latter method is doing the same absolute-threshold difference, but between the image of time t and a reference image. This reference image *needs to be as close as possible to the fixed background*, therefore, the *temporal mean* or *temporal median* image is used as the reference.

### Image segmentation and region characterization
Image segmentation is a range of techniques that assign labels to particular region of an image.
Then, it is possible to extract statistics from each region (read: pixels with same label).
For each region is computed its : pixel size, barycentre, co-variance matrix, main direction, RGB mean and grey histogram.


## Multi-criteria tracking

(one can obtain binary image from a grey scale image, by applying a )