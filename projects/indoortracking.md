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
- **Mathematical morphology operators** : erosion, dilation, opening and closing. They are techniques for the analysis and processing of *geometrical structures* (read: discrete-binary images here) based on *the theory of ensemble*.

- **Movement detection** : either based on *temporal difference* of images or on a difference *with a reference*.  
   More concretely, the first method is an absolute difference of images at the time $t$ and $t-1$, then *threshold is applied* to detect the presence of movement.  
   The latter method is doing the same absolute-threshold difference, but between the image of time $t$ and a reference image. This reference image *needs to be as close as possible to the fixed background*. Therefore, the *temporal mean* or *temporal median* image is used as the reference.


## Multi-criteria tracking

