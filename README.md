# Rendering Lecture using Mitsuba3
Seminar materials of physically based rendering with tutorials and homework using mitsuba3-python

Please do not upload your solution to online publicly.



## Difference from other rendering materials

Physically based rendering, typically ray tracing, is an important foundation of computer graphics and vision. However, the barriers to learning it has been considered to be high due to complex C++ coding and theoretical difficulty. The materials in this repository are aimed at lowering the barrier to learning rendering, and we believe that if you follow the provided lecture materials (`./slides/*.pdf`), tutorials (`./tutotial*.ipynb`), and homework (`./hw*.ipynb`), you should be able to implement path tracing, the baseline algorithm for ray tracing, in less than **a day**.



### Features

In order to minimize the barrier to learning rendering, this material have some features that are somewhat different from other materials.



* Codes are written entirely in Python code using [Mitsuba 3](https://www.mitsuba-renderer.org/).
* Tutorials and homework are written in Jupyter notebooks, making it easy to visualize and check intermediate variables even before a final rendered image is obtained.
* Prioritizes understanding the overall ray tracing algorithm, leaving the lower-level queries used within ray tracing as optional contents. 
* In particular, the ray intersection query is not covered in this material. While the purpose of the query is clear, the implementation is complex and virtually impossible to be done in Python.



## Contents

## Learning sequence

You can understand ray tracing algorithms very quickly by studying the following sections.

1. Radiometry: `./slides/01. radiometry and light transport.pdf`, `./hw1_radiometry.ipynb`
   * HW1 uses Mitsuba 3's API to use values which the scene consists of, but you don't need to study the `./tutorial*.ipynb` tutorial in this step because I have already provided the usage in the skeleton code for each problem.
2. Probability: `./slides/02. probability and statistical inference.pdf`, `./hw2_probability.ipynb`
3. Path Tracing: `./slides/TBA`, `./tutorial*.ipynb`, `./hw3_pathtracing.ipynb`, `./hw4_pathtracing.ipynb`
   * HW3, HW4 need to be started after reading the tutorial in `./tutorial*.ipynb`.



TBA below

4. Advanced sampling: (limited form of) bidirectional path tracing, combining with analytic integration, etc.
5. Volumetric rendering
6. Differentiable rendering: detached vs. attached sampling, boundary integral vs. reparameterization, etc. It would be nice to have a compelling visualization of the complex methodologies that appear only in differentiable rendering, and easy examples to make the concepts clear in simple situations.

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
