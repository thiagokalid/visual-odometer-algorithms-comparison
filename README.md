# 0. Comparison between Visual Odometry algorithms.

This repository provides a framework for implementing a **visual odometry system**. It also includes a [`benchmark suite`](visual-odometer-algorithms-comparison/scripts/benchmark.ipynb) for comparing different image registration algorithms.

# 1. Introduction <a name="introduction"></a>

A visual odometer is a system capable of estimating spatial displacement between two images. It can be associated with a sensing probe (such as an ultrasonic transducer) to aid in giving spatial information during a mechanical sweep. 

There are many types of algorithms that can estimate the displacement between two images. In this work, I'll compare some algorithms that perform this task mostly in the frequency domain, namely:

* Time-domain phase-correlation [1]
* Displacement estimation based on Singular Value Decomposition [2]
* Displacement estimation based on a Projection Singular Value Decomposition [3]
* Time-domain phase-amplified-correlation [4]

In the next section, each algorithm will be briefly explained.

# 2. Modelling <a name="modelling"></a>

Let $g[x, y]$ be a gray-scale image shifted version of $f[x, y]$, i.e.

$$
g[x, y] = f[y - \Delta x, y - \Delta y],
$$

the main goal of image registration methods, the core of a visual odometer, is to estimate the horizontal and vertical shift $\Delta x$ and $\Delta y$.

## 2.1. Time-domain phase-correlation

If $G(u, v)$ and $F(u, v)$ are the spectra of $g(x, y)$ and $f(x, y)$, respectively, the cross-power spectrum between them is

$$
CPS(u, v) = \frac{F(u, v) \odot G(u, v)^*}{F(u, v) \odot G(u, v)}
$$

where $\odot$ is the element-wise product (Hadamard product). Since $f(x,y)$ and $g(x,y)$ are related by a spatial shift, the shift property of the Fourier transform lets us rewrite the previous equation as 

$$
CPS(u, v) = \frac{F(u, v) \odot (F(u, v)\exp(-j(u\Delta y + v\Delta x))^*}{|F(u, v) \odot (F(u, v)\exp(-j(u\Delta y + v\Delta x))|} = \exp(-j(u\Delta y + v\Delta x))
$$

which have a known inverse transform:

$$
r(x, y) = IFT{CPS(u, v)}(x,y) = \delta(x - \Delta x, y - \Delta x)
$$

Therefore, by spotting the location of the maximum of $r(x,y)$, we are, in fact, estimating the relative shift between $f(x,y)$ and $g(x,y)$:

$$
\Delta x, \Delta y = \text{arg}\max_{x,y} r(x,y)
$$

## 2.2 Displacement estimation based on Singular Value Decomposition

If one pays attention to the cross-power spectrum result:

$$
CPS(u, v) = \frac{F(u, v) \odot (F(u, v)\exp(-j(u\Delta y + v\Delta x))^*}{|F(u, v) \odot (F(u, v)\exp(-j(u\Delta y + v\Delta x))|} = \exp(-j(u\Delta y + v\Delta x))
$$

it is possible to split into two orthogonal functions, i.e.:

$$
CPS(u,v) = \exp(-j u\Delta x) \exp(-j v\Delta y)
$$

If the cross-power spectrum is, instead, interpreted as a matrix:

$$
CPS_{matrix}[u, v] = \exp(-j u\Delta x) \exp(-j v\Delta y)
$$

one can easily conclude that $CPS_{matrix}[u, v]$ singular. The method proposed by Hoge [2] takes advantage of this property by applying the rank-one truncated SVD decomposition of $CPS_{matrix}[u, v]$ which yields

$$
q_u = \exp(-j u \Delta x)
$$
and

$$
q_v = \exp(-j v \Delta y)
$$

and finally, by extracting the phase of $q_u$ and $q_v$ we can estimate the displacement through a linear regression, since both function have linear phases

$$
p_u(u) = \angle q_u =  u \Delta x
$$
$$
p_v(v) = \angle q_v =  v \Delta y
$$

## 2.3 Displacement estimation based on a Projection Singular Value Decomposition

The inverse transform of the cross-power spectrum is an image with, theoretically, a single maximum value located at coordinates proportional to the displacement between f(x,y) and g(x,y). The idea behind the projection-SVD method [3] is to define maximum and minimum displacement values that the algorithm is capable of identifying

$$
r'(x,y) = r(x,y) |  \Delta x_{min} < x < \Delta x_{max}, \Delta y_{min} < y < \Delta y_{max}
$$

then a forward DFT is applied to $r′(x,y)$, and the displacement is estimated using the same SVD-based method proposed by Hoge (taking into account the image size prior to the filtering). While the filtering step might appear redundant—since the displacement could be obtained directly by locating the peak in the spatial domain—it actually enhances robustness and algorithm complexity. Compared to traditional phase-correlation, the SVD method offers greater resistance to noise. Moreover, the filtering step reduces the size of the matrix involved in the singular value decomposition (reducing the computational complexity) and effectively acts as a low-pass filter on the signal phase.

## 2.4 Time-domain phase-amplified-correlation

The classic phase-correlation algorithm is capable of estimating only integer displacement values, since it relies on image peak-detection. One way of estimating sub-pixel displacements would be, somehow, multiplying the original shift $\Delta x$ and $\Delta y$ by a gain $m$ resulting in $g'(x,y)=f(x-m\Delta x, y - m\Delta y)$, estimating the bigger shifts and, finally, dividing the bigger estimated shift by $m$ to compensate the previously applied gain. That is the idea behind phase-amplified correlation [4]. The CPS between two images shifted by $\Delta x$ and $\Delta y$ is

$$
CPS(u,v) = \exp(-j u\Delta x) \exp(-j v\Delta y)
$$

applying element-wise power is equivalent to applying a gain to the displacements:

$$
g'(x,y)=f(x-m\Delta x, y - m\Delta y) \Longleftrightarrow  CPS(u,v)^{1 + m} = (\exp(-j u\Delta x) \exp(-j v\Delta y))^{m+1}
$$

finally, the displacements with gain could be estimated through

$$
r'(x, y) = IFT(CPS^{1 + m}) = \delta(x - (1 + m)\Delta x, y - (1 + m)\Delta y)
$$

then

$$
\Delta x', \Delta y' = \text{arg}\max_{x,y} r'(x,y)
$$

and the original displacement computed as

$$
\left(\Delta x, \Delta y\right) = \left(\frac{\Delta x'}{m + 1}, \frac{\Delta y'}{m + 1}\right)
$$



# References
[1] Foroosh, H., Zerubia, J. B., & Berthod, M. (2002). Extension of phase correlation to subpixel registration. IEEE transactions on image processing, 11(3), 188-200.

[2] Hoge, W. S. (2003). A subspace identification extension to the phase correlation method [MRI application]. IEEE transactions on medical imaging, 22(2), 277-280.

[3] Keller, Y., & Averbuch, A. (2007). A projection-based extension to phase correlation image alignment. Signal processing, 87(1), 124-133.

[4] Konstantinidis, D., Stathaki, T., & Argyriou, V. (2019). Phase amplified correlation for improved sub-pixel motion estimation. IEEE Transactions on Image Processing, 28(6), 3089-3101.
