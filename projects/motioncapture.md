---
layout: page
title: Motion Capture
subtitle: Frankly idk right now :)
bigimg: /img/projects/motioncapture/bigimage.jpg
---

**[Motion Capture](https://github.com/johan-gras/Motion-Capture) is an on-board computing project for motion capture using accelerometer data.**
The source code is written in C and is compiled for a [Gumxtix controller](https://www.gumstix.com/).
A GUI application on a remote computer **let you observe the motion in real time**.

<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/motioncapture/gui.jpeg" alt="GUI application"/>
	</figure>
</div>

## Overview

**Setup:** this project is built upon a [Gumstix Overo Airstorm](https://store.gumstix.com/coms/overo-coms/overo-airstorm-y-com.html) and an [extension card](https://store.gumstix.com/development-boards/gallop43.html) (accelerometer) under [Linux Angström](http://www.angstrom-distribution.org/).

**Project description:** displaying on a remote computer, the real-time position and motion of the Gumstix controller.

**Functional challenges:** blabla
- *Recover accelerometer data* from the registers in an optimal way.
- *Compute the position of the controller* from noisy accelerations data.
- *Guarantee real-time constraints* with a minimum of missed deadlines.
- *Communication* between the system and the computer with an ad hoc wifi.
- *Graphic display* of the system position and motion.



<p></p>

## Technical choices and implementation

### Communication

### From acceleration to position

### Real-time constraint (task period)

### Desktop GUI

## Results and performances