# Stage 1.3: Classical Edge Detection (Canny & Sobel) from First Principles

**Dataset:** NEU-DET, `crazing` defect class (`crazing_1.jpg`)

## What this does

Implements the Sobel gradient operator entirely from scratch (no `cv2.filter2D`,
no `scipy.signal.convolve2d`) to prove the underlying math: 2D correlation via
zero-padding and a sliding 3x3 window. Then compares the result against
OpenCV's `cv2.Canny`, which runs the full 5-stage pipeline: Gaussian smoothing,
gradient computation, non-max suppression, double thresholding, and hysteresis.

## Key findings

- Raw Sobel produces thick, scattered, noisy edges, especially visible on this
  image, since "crazing" is a dense network of fine, high-frequency surface
  cracks that raw gradient thresholding can't distinguish from noise.
- Canny produces thin, continuous edge curves. The two biggest contributors
  are non-max suppression (thins edges to 1px by keeping only local maxima
  along the gradient direction) and hysteresis (a weak-gradient pixel only
  survives if connected to a strong one, which is what turns scattered dots
  into continuous curve fragments).
- A hand-rolled `apply_kernel` and `cv2.filter2D` compute the same underlying
  operation (correlation). Any visible difference between two runs comes from
  the threshold value used, not the computation method. Confirmed directly by
  re-running both with the same percentile-informed threshold.

## Run it

```
pip install opencv-python numpy
python edge_detection.py
```
