USTH Advanced Programming with Python 2026
===============================================

* Your name: **Tran Ngoc Thanh Binh**
* Your id: **2540037**


Project
======================

* Goals: Implement the Non-local Means (NLM) denoising algorithm on an input image using a GPU.
    * Input: RGB image
    * Output: Denoised RGB image
    * Method: **Non-local means** [(Wikipedia refrence)](https://en.wikipedia.org/wiki/Non-local_means)
    * Platform: Numba CUDA on NVIDIA GPUs, or Numba HIP on AMD GPUs when that optional backend is installed.

* Requirements:
    * Make a new private repo on your Github account, invite me (my email @usth) to collaborate.
    * Put everything in a *single* .py source code file. No Jupyter notebook or Google Colab.
        * Takes input file name (an image in JPG format) from ```argv[1]```
        * Writes denoised output to ```output.jpg```
        * Static filter parameters (described in extras, below):
            * Search window size $R = 21$
            * Patch size $k = 7$
            * Smoothing strength $h = 10$
    * Python imports are limited to ```numpy```, ```numba```, and ```matplotlib``` (no ```cv2``` or other image-processing packages).
    * The AMD development environment also has the optional ```numba-hip``` backend installed. HIP imports are guarded so NVIDIA systems can use Numba CUDA without that package.
    * Don't copy paste from your friends. I have my own similarity checking tool. *I kill friendships*.
    * Responsible AI support. I will interview personally and directly, if I suspect abuse of AI usage.

* Deadline: 23:59, Sunday, October 11th 2026.
    * Hard deadline. No extension at all.
    * I will get the latest commit which is before the above deadline.
    

Run from the prepared Python environment:

```bash
python main.py input.jpg
```

The script reads one RGB JPG argument and writes the result to ```output.jpg``` in the current directory. It uses the HIP backend when an AMD HIP device is available, otherwise it selects CUDA. It reports an error if neither GPU backend has a usable device.

Non-local means extras:
==========================

To denoise a pixel $i$ from the input image $I$, the algorithm calculates a weighted average of all pixels $j$ within a large search window $S$ (size $R \times R$) around pixel $i$. Denote $I(i)$ as the intensity of the pixel $i$ in the input image $I$. Let $\Phi(i)$ be the output pixel intensity that corresponding to the input pixel $I(i)$. $\Phi(i)$ can *simply* calculated as:

$\Phi(i) = \frac{\sum _{j\in S}w(i,j) \cdot I(j)}{\sum _{j\in S} w(i,j)}$

The weight $w(i,j)$ evaluates how similar the local neighborhood around pixel $i$ is to the neighborhood around pixel $j$. Each neighborhood is called a comparison patch $P$ (size $k \times k$). The weight $w(i,j)$ is calculated as an exponential weight of the distance:

$w(i,j)=\exp \left(-\frac{d^{2}(i,j)}{h^{2}}\right)$

in which, $d(i,j)$ is the L2 Norm Distance and $h$ is a smoothing strength (higher $h$ leads to more blurry image). For two patches centered at $i$ and $j$ (called $P(i)$ and $P(j)$, respectively), compute the sum of squared differences for all corresponding pixels within the patch dimension:

$d^{2}(i,j)=\frac{1}{3k^{2}}\sum _{p\in P}\sum _{c\in\{R,G,B\}}\left(I_{c}(i+p)-I_{c}(j+p)\right)^{2}$

The distance is the mean squared error over all patch pixels and RGB channels. Input intensities stay on the 0–255 scale, so the fixed smoothing strength $h=10$ uses that scale. The image is reflect-padded by the search radius plus the patch radius, so every pixel uses complete patches and a full search window. If the patches are nearly identical, the distance is close to 0 and the weight is close to 1.


