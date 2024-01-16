---
title: Fourier_Transform_Simulation
app_file: demo.py
sdk: gradio
sdk_version: 4.13.0
---

# Fourier Transform

## Description

This gradio app showcases the fourier transformation using the winding method showcased
in 3Blue1Brown video: https://www.youtube.com/watch?v=spUNpyF58BY&t=183s&ab_channel=3Blue1Brown

The app provides 3 animation videos to showcase the transformation. The first video shows the
complex wave along with vertical lines indicating the winding frequency. The second video shows
the complex wave wrapped around the origin on a complex plane using the winding frequency. The
third video shows the location of the center of mass of the wrapped complex wave, by either 
specifying the x, y coordinates or magnitude. For more information, please check the video.

The app is able to apply the winding method on a complex wave built using multiple 
sinusoidal waves by specifiying their frequency and phase (-1.0 - 1.0). Additional parameters can 
be tweaked to adjust the video animations, e.g. initial cycling frequency, signal time, ...


## Setup

Install poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

Install dependencies

```
poetry install
```

## Run

```
 poetry run python demo.py
```
