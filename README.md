# Rendering Lecture using Mitsuba3
Seminar materials of physically based rendering with tutorials and homework using mitsuba3-python

Please do not upload your solution to online publicly.



## Dependencies

Python modules

* NumPy
* Matplotlib
* mitsuba 3
  * Dr. Jit


You can install all the dependencies by simply typing in your terminal:

`pip install numpy matplotlib ipykernel ipywidgets PyQt5 mitsuba`

Note that `pip install mitsuba` automatically installs Dr. Jit



To use vectorized variants for fast rendering in python, install `NVidia driver >= 495.89` (Windows / Linux) or `LLVM >= 11.1` (Mac OS)

In Mac OS, you may install llvm just by

`brew install llvm`



## Installing troubleshooting

### LLVM variant

https://github.com/NVlabs/sionna/discussions/160



* If `mi.set_variant('llvm_ad_rgb')` works on terminal->python, but not in VSCode

https://stackoverflow.com/questions/43983718/how-can-i-globally-set-the-path-environment-variable-in-vs-code
