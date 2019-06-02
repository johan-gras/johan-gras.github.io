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


## Image processing algorithms
### Movement detection
[Show image of TD image somewhere]

Two techniques implements **the detection of movement**.
One based on the **temporal difference** of images and the second on a **difference with a reference**.
More concretely, the first method is *an absolute difference of images* at the time t and t-1 (of the sequence of image), then *threshold is applied* to detect *the presence of movement*.
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
What this fancy name even mean ~~you may ask~~ ? This *an end to end method*, that is where the **system** is comming from. The user of the method *can choose witch object to follow* during the complete sequence of images, that is for the **tracking**. And the computation that track this object is based on not one but *an ensemble of [to see] image processing techniques*, there you go with your **multi-criteria** !

### And what your tracking stuff is doing ?
One can take a full sequence of images (based on a video).
Our system is first going to compute the movement detection on each frames (using the second techique, with the median as reference).
Then, we obtain a noisy binary image, therefore we use a combinaison of opening and closing to de-noisify the movement image.
The system use our segmentation algorithm and characterized each labeled regions.
For each images and each moving regions, the characteristics are saved in a .json file to abstract some constraints of computation time.



(one can obtain binary image from a grey scale image, by applying a )