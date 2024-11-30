# Rendering Lecture using Mitsuba3
Seminar materials of physically based rendering with tutorials and homework using mitsuba3-python

Please do not upload your solution to online publicly.



본 자료는 완벽하지 않거나 작업 갱신이 느릴 수 있다. [Discussion](https://github.com/shinyoung-yi/lecture-rendering-mitsuba/discussions) 페이지를 활용한 모든 **질문, 오류 정정, 컨텐츠 제안, 업데이트에 대한 종용** 등은 언제나 환영된다.

## 다른 렌더링 학습 자료와의 차이점

물리 기반 렌더링, 대표적으로 광선 추적(ray tracing)은 컴퓨터 그래픽스 및 비전에서 중요한 기반 기술이다. 하지만 복잡한 C++ 코딩, 이론적 난이도 등으로 인해 진입 장벽이 높은 것이 사실이다. 본 리포지토리에 담긴 자료는 렌더링 교육의 진입 장벽을 낮추는 것을 가장 큰 목표로 하고 있으며, 제공된 강의 자료(`./slides/*.pdf`), 튜토리얼(`./tutotial*.ipynb`), 과제(`./hw*.ipynb`)를 잘 따라올 경우 **하루**의 시간 안에 광선 추적 분야의 베이스라인 알고리즘인 경로 추적(path tracing)을 구현할 수 있을 것이라 생각한다.



### 특징

렌더링 학습을 진입 장벽을 최소한으로 낮추기 위해, 본 자료는 여타 자료들과는 다소 다른 특징들이 있다.

* [Mitsuba 3](https://www.mitsuba-renderer.org/)를 이용하여 Python 코드로만 모든 과정이 이루어짐
* 튜토리얼과 과제가 Jupyter notebook으로 이루어져, 최종 렌더링 이미지를 얻기 전이어도 중간 과정의 변수들을 시각화하여 확인하기 용이함
* 전체적인 광선 추적 알고리즘에 대한 이해를 우선시하여, 광선 추적 내부에서 사용되는 하위레벨의 쿼리들은 선택 학습 요소로 남겨둠. 
* 특히, 광선 교차(ray intersection) 쿼리는 본 자료에서 다루지 않음. 쿼리의 목적성이 명료한 데에 반하여 구현이 복잡한데다가 파이썬 구현이 사실상 불가능



## 학습 순서

다음 단원만 학습하면 아주 빠르게 광선 추척 알고리즘에 대한 이해를 얻을 수 있다.

1. Radiometry: `./slides/01. radiometry and light transport.pdf`, `./hw1_radiometry.ipynb`
   * HW1에서 장면을 구성하는 값에 참조하기 위해 Mitsuba 3의 API를 사용하지만, 각 문제의 스켈레톤 코드에 이미 사용법을 제공하였기에 1단원에서는 `./tutorial*.ipynb` 튜토리얼을 학습하지 않아도 무방
2. Probability: `./slides/02. probability and statistical inference.pdf`, `./hw2_probability.ipynb`
3. Path Tracing: `./slides/TBA`, `./tutorial*.ipynb`, `./hw3_pathtracing.ipynb`, `./hw4_pathtracing.ipynb`
   * HW3, HW4는 `./tutorial*.ipynb` 튜토리얼을 읽은 후 시작하는 것이 필요



추후 추가 예정

4. Advanced samplings: (제한된 형태의) bidirectional path tracing, 해석적 적분과의 결합, 등 다양한 알고리즘 중 몇 개
5. Volumetric rendering
6. Differentiable rendering: detached와 attached sampling, boundary integral과 reparameterization 등 differentiable rendering에만 등장하는 복잡한 방법론들의 전달력 있는 시각화 및 단순한 상황에서 개념을 확실히 구분할 수 있는 쉬운 예제들을 만들면 좋겠다고 생각 중



## 의존성

본 자료에 포함된 코드들을 실행시키기 위해서는 아래와 같은 파이썬(3.11) 패키지가 필요하다.

* NumPy
* Matplotlib
* mitsuba 3
  * Dr. Jit


터미널에 다음 명령을 입력하여 한번에 설치할 수 있다.

```cmd
pip install numpy==1.26 matplotlib ipykernel ipywidgets PyQt5 mitsuba=3.6.0
```

처음에 나열된 의존성 중 Dr. Jit `pip install mitsuba`을 실행할 때 자동으로 설치된다.



To use vectorized variants for fast rendering in python, install `NVidia driver >= 495.89` (Windows / Linux) or `LLVM >= 11.1` (Mac OS)

In Mac OS, you may install llvm just by

`brew install llvm`



## Installing troubleshooting

### LLVM variant

https://github.com/NVlabs/sionna/discussions/160



* If `mi.set_variant('llvm_ad_rgb')` works on terminal->python, but not in VSCode

https://stackoverflow.com/questions/43983718/how-can-i-globally-set-the-path-environment-variable-in-vs-code
