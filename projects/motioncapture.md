---
layout: page
title: Motion Capture
subtitle: Your next Wiimote ;)
bigimg: /img/projects/motioncapture/bigimage.jpg
---

**[Motion Capture](https://github.com/johan-gras/Motion-Capture) is an on-board computing project for motion capture using accelerometer data.**
The source code is written in C and is compiled for a [Gumxtix controller](https://www.gumstix.com/).
A GUI application on a remote computer lets you observe the motion in real time.

<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/motioncapture/gui.jpeg" alt="GUI application"/>
	</figure>
</div>

## Overview

**Setup:** this project is built upon a [Gumstix Overo Airstorm](https://store.gumstix.com/coms/overo-coms/overo-airstorm-y-com.html) and an [extension card](https://store.gumstix.com/development-boards/gallop43.html) (accelerometer) under [Linux Angström](http://www.angstrom-distribution.org/).

**Project description:** displaying on a remote computer, the real-time position and motion of the Gumstix controller.

**Functional challenges:**
- **Recover accelerometer data** from the registers in an optimal way.

- **Compute the position** of the controller from noisy accelerations data.

- **Guarantee real-time constraints** with a minimum of missed deadlines.

- **Communication** between the system and the computer with an ad hoc wifi.

- **Graphic display** of the system position and motion.


<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/motioncapture/overview.png" alt="Overview of the architecture."/>
	  <figcaption>Overview of the architecture.</figcaption>
	</figure>
</div>

[](## Technical choices and implementation)

## Recovering accelerometer data

**Accelerometer data are saved in a specific register**, thus we need to recover them from the [I²C](https://en.wikipedia.org/wiki/I%C2%B2C) communication bus.
Functions from the manufacturer are already available to do this job, but the code is large, complex and with multiple dependencies.
Therefore, I've significantly reduced the code to gain a **speedup in performance of 1.5**.

## Computing the position

To compute the position from acceleration, one can take the mathematical relation between the two:
**by doing a double integral onto the acceleration signal, the relative position may be obtained**.

<div style="text-align: center;">
	<figure>
	  <img src="/img/projects/motioncapture/integral.png" alt="From acceleration to relative position."/>
	  <figcaption>From acceleration to relative position.</figcaption>
	</figure>
</div>

*Sadly, reality is not as perfect as theory...*

First, **the acceleration signal is not continuous but discretized**.
Second, **data are noisy** in many ways due to the sensor imperfection.
Thus, positions calculated from this data would be inaccurate.

**Different techniques are applied to partially correct existing incertitudes**:

- [Trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) for integrals: reduce integration error on discrete values.
- [Sensor calibration](https://learn.adafruit.com/calibrating-sensors/why-calibrate): reduce measurement bias.
- [Low pass filter](https://en.wikipedia.org/wiki/Low-pass_filter) onto the acceleration signal: help to reduce the mechanical and electrical noise of the accelerometer.
- [Window filtering](https://en.wikipedia.org/wiki/Window_function): ignore acceleration values near zero to annihilate noise during stationary periods.
- Motion verification: force the estimated speed to zero if acceleration is null for long enough.

<!---
### Real-time constraint task period

### Communication

### Desktop GUI

## Results and performances)
-->
