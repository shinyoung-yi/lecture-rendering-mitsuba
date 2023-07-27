# Homework 1: Probelm 1. (e) Jacobian determinant of Rusinkiewicz coordinates

$$
\def\homega{\hat\omega}

\def\rmd{\mathrm{d}}
\def\rmf{\mathrm{f}}
\def\bfu{\mathbf{u}}
\def\bfB{\mathbf{B}}
\def\bfJ{\mathbf{J}}

\def\floor#1{\left\lfloor #1 \right\rfloor}
\def\ceil#1{\left\lceil #1 \right\rceil}

\def\abs#1{\left|{#1}\right|}
\def\pfrac#1#2{\frac{\partial#1}{\partial#2}}
$$

***TODO: polish more***

## Basic Differential Geometry

$\phi:U\subset\R^n \to M\subset \R^m$
$$
\int_{V}{ f\left(x,y\right) \rmd M\left(x\right) }
= \int_{f^{-1}\left(V\right)}{f\left(\bfu\right)J\left(\bfu\right)\rmd^n \bfu}
, \\
\text{ where }
J\left(\bfu\right) = \left|\frac{
	\rmd M
}{
	\rmd^n \bfu
}\right|
= \sqrt{\det \bfJ_\phi^T\bfJ_\phi}
\text{ and }
$$




A parameterized product manifold of surfaces: $\phi:U\subset\R^4 \to M\times N\subset\R^3\times\R^3$.

Differential measure:
$$
\int_{V}{ f\left(x,y\right) \rmd M\left(x\right)\rmd N\left(y\right) }
= \int_{f^{-1}\left(V\right)}{f\left(\bfu\right)J\left(\bfu\right)\rmd^4 \bfu}
,\text{ where }
J\left(\bfu\right) = \left|\frac{
	\rmd M\rmd N
}{
	\rmd^4 \bfu
}\right|. \\

\text{Here, } \int_{V}{ f\left(x,y\right) \rmd M\left(x\right)\rmd N\left(y\right) }
= \int_{\Pi_N\left(V\right)}{
	\int_{\left\{ x\in M\mid\left(x,y\right)\in V \right\}}{f\left(x,y\right) \rmd M\left(x\right)}
\rmd N\left(y\right)}
$$

$$
\left(\bfJ_\phi^T\bfJ_{\phi}\right)_{ij}\left(\bfu\right) = \pfrac{\mathbf x}{u_i}\cdot \pfrac{\mathbf x}{u_j} + \pfrac{\mathbf y}{u_i}\cdot \pfrac{\mathbf y}{u_j}
$$



## Intermediate condition

$M^n,N^n \subset \R^{n+1}$

For any $R \in G$, $R\left(M\right)=M$ and $R\left(N\right)=N$, where $G\subset GL\left(n+1\right)$ is a 1-dim. matrix group.

Partial coordinates $\bfu_{1:2n-1}:M\times N\to \R^{2n-1}$, where $\bfu\left(x,y\right)=\bfu\left(x',y'\right)$ then $\left(x',y'\right)=R\left(x,y\right)$ for some $R\in G$.

#### **Proposition.** Suppose that there are two (total) coordinate systems $\bfu$ and $\bfu'$ which extending the partial coordinates $\bfu_{1:3}$. Then the Jacobians of the induced parameterizations satisfies $R\bfJ_{\bfu^{-1}}=\bfJ_{\bfu'^{-1}}$ for some $R\in G$.

> By supposition:
> $$
> \bfu:M\times N \to U \subset \R^{2n}, \\
> \bfu':M\times N \to U' \subset \R^{2n}.
> $$
> By the property of the partial coordinates:
> $$
> \forall \left(x,y\right)\in M\times N,\ \bfu'^{-1}\circ\bfu\left(x,y\right) = R_\left(x,y\right)\left(x,y\right)
> $$
> 

$$
\int_{M\times N}{f\left(\bfu_{1:2n-1}\right)\rmd M\rmd N} = \int_{U}{f\left( \bfu_{1:2n-1}\right) \left|\bfJ_{\bfu^{-1}}\right|\rmd^{2n}\bfu}
$$



## Rusinkiewicz Jacobian (differential measure)

$$
\mathrm{sph}\left(\theta,\phi\right) = \mathrm{sph}\left(\theta,\phi;\hat x,\hat y, \hat z\right) = \sin\theta\cos\phi\hat x+\sin\theta\sin\phi\hat y+\cos\theta\hat z, \\
\homega_h = \left(\sin\theta_h \cos\phi_h,\sin\theta_h\sin\phi_h, \cos\theta_h\right), \\
\homega_{i|o} = \cos\theta_d\homega_h \pm \sin\theta_d\cos\phi_d\left(\cos\theta_h\cos\phi_h,\cos\theta_h\sin\phi_h,-\sin\theta_h\right) \pm \sin\theta_d\sin\phi_d\left( -\sin\phi_h,\cos\phi_h,0 \right) \\
= \cos\theta_d\homega_h \pm \sin\theta_d\cos\phi_d\hat\theta_h \pm \sin\theta_d\sin\phi_d\hat\phi_h, \\

\pfrac{\homega_{i|o}}{\theta_h} = \mp \sin\theta_d\cos\phi_d\homega_h+\cos\theta_d \hat\theta_h +0, \\

\pfrac{\homega_{i|o}}{\phi_h} = \left(\sin\theta_h \cos\theta_d \pm \cos\theta_h\sin\theta_d\cos\phi_d \right)\hat\phi_h \mp \sin\theta_d\sin\phi_d\left(\cos\phi_h,\sin\phi_h,0\right), \\

\pfrac{\homega_{i|o}}{\theta_d} = -\sin\theta_d\homega_h \pm \cos\theta_d\cos\phi_d\hat\theta_h \pm \cos\theta_d\sin\phi_d\hat\phi_h, \\

\pfrac{\homega_{i|o}}{\phi_d} = 0\mp \sin\theta_d\sin\phi_d\hat\theta_h \pm \sin\theta_d\cos\phi_d\hat\phi_h
$$

> ### Rusinkiewicz to spherical coordinates of two directions
>
> $$
> \cos\theta_{i|o} = \cos\theta_h\cos\theta_d \mp\sin\theta_h\sin\theta_d\cos\phi_d, \\
> \tan\phi_{i|o} = \frac{
> 	\sin\theta_h\sin\phi_h\cos\theta_d\pm\cos\theta_h\sin\phi_h\sin\theta_d\cos\phi_d \pm \cos\phi_h\sin\theta_d\sin\phi_d
> }{
> 	\sin\theta_h\cos\phi_h\cos\theta_d\pm\cos\theta_h\cos\phi_h\sin\theta_d\cos\phi_d \mp \sin\phi_h\sin\theta_d\sin\phi_d
> } \\
> 
> \mathrm{numer}^2+\mathrm{denom}^2 = \sin^2\theta_h\cos^2\theta_d+\cos^2\theta_h\sin^2\theta_d\cos^2\phi_d+\sin^2\theta_d\sin^2\phi_d \pm 4\sin\cos\theta_h\sin\cos\theta_d\cos\phi_d \\
> = \sin^2\theta_h\left(\cos^2\theta_d -\sin^2\theta_d\cos^2\phi_d\right)+\sin^2\theta_d \pm 4\sin\cos\theta_h\sin\cos\theta_d\cos\phi_d \eqqcolon r_{i|o}^2 \\
> 
> r_ir_o = \sqrt{
> 	\left[\sin^2\theta_h\left(\cos^2\theta_d-\sin^2\theta_d\cos^2\phi_d\right)+\sin^2\theta_d\right]^2 - 16\sin^2\cos^2\theta_h \sin^2\cos^2\theta_d\cos^2\phi_d
> }\\
> 
> \sin\left(\phi_i-\phi_o\right) = \sin\phi_i\cos\phi_o-\cos\phi_i\sin\phi_o = \frac{1}{r_ir_o}\left[
> 	 2 \sin\theta_h \sin\cos\theta_d \sin\phi_d
> \right], \\
> \cos\left(\phi_i-\phi_o\right) = \cos\phi_i\cos\phi_o+\sin\phi_i\sin\phi_o = \frac1{r_ir_o}\left[ \sin^2\theta_h\cos^2\theta_d - \cos^2\theta_h\sin^2\theta_d\cos^2\phi_d-\sin^2\theta_d\sin^2\phi_d \right] \\
> = \frac1{r_ir_o}\left[
> 	\sin^2\theta_h\left( \cos^2\theta_d + \sin^2\theta_d \cos^2\phi_d \right) - \sin^2\theta_d
> \right], \\
> = \frac1{r_ir_o}\left[
> 	\sin^2\theta_h\left( 1 - \sin^2\theta_d \sin^2\phi_d \right) - \sin^2\theta_d
> \right]
> $$
>
> when $\homega_i$ denotes material to light direction.

> When $\theta_d=\frac\pi2$:
> $$
> \homega_{i|o}=\pm \cos\phi_p \left(\cos\theta_h\cos\phi_h,\cos\theta_h\sin\phi_h,-\sin\theta_h \right) \pm \sin\phi_d\left(-\sin\phi_h,\cos\phi_h,0\right)
> $$
> Note that
> $$
> x_{i|o}^2 + y_{i|o}^2 =\cos^2\phi_p\cos^2\theta_h+\sin^2\phi_d = 1 - \sin^2\theta_h\cos^2\phi_p
> $$
>
>
> Suppose that there is $\theta_h'$, $\phi_h'$, and $\phi_d'$ such that $\homega_{i|o}=\homega_{i|o}'$ (equal both $i$ and $o$).
> $$
> \sin\theta_h\cos\phi_p = \sin\theta_h' \cos\phi_p', \\
> \mathrm{or}\cases{1\\1}
> $$
>
> 
>
> ​	\sin\theta_h\cos\phi_p
> }{
> ​	\sin\theta_h'
> }\right) \\

Note that:
$$
\homega_h\cdot \left(\cos\phi_h,\sin\phi_h,0\right)=\sin\theta_h, \\
\hat \theta_h \cdot\left(\cos\phi_h,\sin\phi_h,0\right)=\cos\theta_h,\\ \hat\phi_h\cdot\left(\cos\phi_h,\sin\phi_h,0\right)=0
$$
$$
\bfJ_{\homega_{i|o}}^T \bfJ_{\homega_{i|o}} = 2\pmatrix{
	\sin^2\theta_d\cos^2\phi_d + \cos^2\theta_d & \sin\theta_h\sin^2\theta_d\sin\cos\phi_d & 0 & 0 \\
	\sin\theta_h\sin^2\theta_d\sin\cos\phi_d & \sin^2\theta_h\left(\cos^2\theta_d-\sin^2\theta_d\cos^2\phi_d\right) + \sin^2\theta_d & 0 & \cos\theta_h \sin^2\theta_d \\
	0 & 0 & 1 & 0 \\
	0 & \cos\theta_h\sin^2\theta_d & 0 & \sin^2\theta_d
}
$$

$$
\frac1{16} \det \bfJ^T\bfJ
= \left(\sin^2\theta_d\cos^2\phi_d+\cos^2\theta_d\right) \sin^2\theta_d\left[ \sin^2\theta_h\left(1-\sin^2\theta_d\cos^2\phi_d\right)  \right]
- \sin^2\theta_h\sin^6\theta_d\sin^2\cos^2\phi_d \\

= \sin^2\theta_h\sin^2\theta_d\left[
	\left(\sin^2\theta_d\cos^2\phi_d+\cos^2\theta_d\right)\left(1-\sin^2\theta_d\cos^2\phi_d\right) - \sin^4\theta_d\sin^2\cos^2\phi_d
\right] \\

= \sin^2\theta_h\sin^2\theta_d\cos^2\theta_d, \\

\therefore\ J_{\mathrm{Rus}} \coloneqq \sqrt{\det \bfJ^T\bfJ} = 4\left|\sin\theta_h\sin\theta_d\cos\theta_d\right|
$$

$$
16\pi^2 = \int_{S^2\times S^2}{\rmd \homega_i\rmd \homega_o}
= \int_{\left[0,\pi\right]\times\left[0,2\pi\right]\times\left[0,\frac\pi 2\right]\times\left[0,2\pi\right]}{4\left|\sin\theta_h\sin\theta_d\cos\theta_d\right|\rmd\left(\theta_h,\phi_h,\theta_d,\phi_d\right)} \\
= 8\pi \int_{\left[0,\pi\right]\times\left[0,\frac\pi 2\right]\times\left[0,2\pi\right]}{\left|\sin\theta_h\sin\theta_d\cos\theta_d\right|\rmd\left(\theta_h,\theta_d,\phi_d\right)}
= 4\times2\times 2\pi \times \frac12 \times 2\pi = 16\pi^2
$$
