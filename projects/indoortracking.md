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

## Image processing algorithms

### Motion detection
[Show image of TD image somewhere]

Two techniques implements **the detection of motion**.
One based on the **temporal difference** of images and the second on a **difference with a reference**.
More concretely, the first method is *an absolute difference of images* at the time t and t-1 (of the sequence of image), then *threshold is applied* to detect *the presence of motion*.
The latter method is doing the same absolute-threshold difference, but between the image of time t and *a reference image*. This reference image *needs to be as close as possible to the fixed background*, therefore, the *temporal mean* or *temporal median* image of the full sequence is used as the reference.

<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/indoortracking/mean.png" alt="Mean temporal filter"/>
	  <img src="/img/projects/indoortracking/median.png" alt="Median temporal filter"/>
	  <figcaption>Mean (left) and median (right) temporal filters. The mean filter is generally sightly more noisy (red box) than the median filter.</figcaption>
	</figure>
</div>

### Mathematical morphology operators
*Erosion, dilation, opening and closing*, they are techniques for *the analysis and processing* of **geometrical structures** (read: discrete-binary images) based on the **theory of ensemble**.
Those techniques allow to denoise...

### Image segmentation and region characterization
**Image segmentation** is a range of techniques that *assign labels to particular region of an image*.
Then, it is possible to **extract characteristics** from *each region* (read: pixels with the same label).
Therefore, for each region is computed its : pixel size, barycentre, covariance matrix, main direction, mean gray level, means for each RGB component and gray histogram.

### Interest point detection
A number of techniques are possible to the **detection of interest points** (used for subsequent processing).
In this project, the **Harris method** a *corner detection operator* is implemented. 
A corner can be interpreted as *the junction of two edges*, where an edge is *a sudden change in image brightness*.
*Corners are the important features* in the image, and they are generally termed as interest points which are invariant to translation, rotation and illumination.

<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/indoortracking/harris.png" alt="Harris Corner Detector"/>
	  <figcaption>Points of interest based on the Harris Corner Detector.</figcaption>
	</figure>
</div>

## Multi-criteria tracking system
The final goal of this project was the implementation of **a multi-criteria tracking system**.
What this fancy name even mean ~~you may ask~~ ? This is *an end to end method*, that is where the **system** is coming from. The user of the method *can choose which object to follow* during the complete sequence of images, that is for the **tracking**. And the computation that tracks this object is based on not one but *an ensemble of [to see] image processing techniques*, there you go with your **multi-criteria** !

### Ok, and what your tracking stuff is doing ?
One can take *a full sequence of images* (based on a video), the system is first going [to compute the detection of motion](#motion-detection) on each frame (using our second technique, with the median as reference).

<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/indoortracking/resultmove.gif" alt="Sequence of binary motion images"/>
	  <figcaption>Sequence of binary motion images.</figcaption>
	</figure>
</div>

A *noisy binary image* is obtained (the camera may be imperfect), thus we use a combination of [opening and closing](#mathematical-morphology-operators) *to denoisify* the motion images' sequence.

<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/indoortracking/resultclean.gif" alt="Sequence of denoisify images"/>
	  <figcaption>Sequence of denoisify images.</figcaption>
	</figure>
</div>

The system use [our segmentation algorithm](#image-segmentation-and-region-characterization) and [characterized each labeled region](#image-segmentation-and-region-characterization).
Then, for each image and each moving region, the characteristics are saved in a .json file to abstract some constraints of computation time.

Once those pre-processing steps are made the user can choose on the first frame, the region (e.g.: a person) to be tracked during the full sequence. The algorithm now, need to choose the most likelihood region in the subsequent frames. This is not trivial, since image data are noisy even after some pre-processing.

Then, all the characteristics of the initial region are loaded.
For every other frames, an innovative algorithm (read: a home-made algorithm for the specific task) decided what the most likelihood region is based on past information.
In short the algorithm look at the proximity of the chose region on the last frame :  
- If there is only one region, this is the chosen one.
- If there is no region, then we keep the focus on the last region, but warn the user of the momentary loss of tracking.
- If there are multiple regions, then we need to solve conflicts. The region that has the most similar (L1 distance) statistics based on features such as the histogram, color average) to the previous tracked regions is selected.

<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/indoortracking/tracking1.png" alt="Tracking 1"/>
	  <img src="/img/projects/indoortracking/tracking2.png" alt="Tracking 2"/>
	  <img src="/img/projects/indoortracking/tracking3.png" alt="Tracking 3"/>
	  <figcaption>Sequence of denoisify images.</figcaption>
	</figure>
</div>