# Instructions for Running the Project
## 1. Image Sources

* All images are loaded directly from GitHub repository via their raw URLs.

* No local image files are required — the notebook will download them automatically at runtime.

## 2. Libraries

* All necessary libraries are imported at the beginning of the notebook:

```bash
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
```

### In Google Colab:

* Most libraries (NumPy, Matplotlib, OpenCV) are already pre-installed.

* If a library is missing, you can install it directly in a code cell using:

```bash
!pip install opencv-python numpy matplotlib
```

### In Jupyter Notebook (local PC):

* Python must be installed locally.

* Required libraries should be installed beforehand via terminal:

```bash
pip install opencv-python numpy matplotlib
```

## 3. How to Run
### Option 1: Google Colab

1. Open Google Colab in a browser.

2. Upload .ipynb notebook file.

3. Run all cells

### Option 2: Jupyter Notebook (Local)

1. Open the notebook in Jupyter Notebook installed on your computer.

2. Make sure required libraries are installed (see section 2).

3. Run all cells sequentially


## 4. Adjustable Parameters

| **Function**           | **Parameter**                        | **Effect**                                  |
| ---------------------- | ------------------------------------ | ------------------------------------------- |
| Binarization (`cv2.threshold()`)           | `threshold_value`                    | Higher → more black pixels; lower → more white pixels                     
| Gaussian Blur (`cv2.GaussianBlur`)          | Kernel size `(k x k)`                | Larger kernel → stronger smoothing, less detail                           
| Sharpening (Laplacian) | `alpha`                              | Higher → stronger edges, may amplify noise                                 
| Sobel Edge Detection (`cv2.Sobel`)  | `ksize`                              | Larger → thicker edges, smoother gradient                                   
| Cyclic Shift (`np.roll`)          | `dx`, `dy`                           | Number of pixels to shift; large shifts move content further, wrap around   
| Rotation (`cv2.getRotationMatrix2D`) | `angle`   | Rotation angle in degrees (positive = clockwise)                            | Rotation cell ()            |
| Rotation (`cv2.getRotationMatrix2D`)              | `center`                             | Pivot point for rotation, usually image center                              
| Rotation (`cv2.getRotationMatrix2D`)              | `scale`                              | Scaling factor (1.0 = original size)                                        
