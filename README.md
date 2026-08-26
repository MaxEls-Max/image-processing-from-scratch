# Computer Vision: Manual Image Processing & GUI 🧮📷

## Overview
This repository contains a Computer Vision project focused on processing and analyzing RGB images by applying mathematical formulas manually to image matrices. Featuring an interactive graphical user interface (GUI) built with Tkinter, this project demonstrates how core computer vision algorithms operate at the pixel level without relying on high-level automated processing libraries.

## 🛠️ Features & Mathematical Operations
This application allows users to load an image, view real-time RGB pixel coordinate data on hover, and apply the following manual operations:

1. **Manual Brightness:** Modifying pixel intensity by adding a constant value `C`.
2. **Manual Contrast:** Adjusting color variance using a gain coefficient `G` and a central point `P` with the formula: `G * (pixel - P) + P`.
3. **Geometric Mirroring:** Flipping image arrays horizontally, vertically, or both.
4. **Manual Rotation:** Rotating the image based on specific degrees using Trigonometry (Sine/Cosine) and **Inverse Mapping** to prevent empty pixels/holes.
5. **Robert Edge Detection:** Identifying boundaries within the image using the Robert Cross operator formula `|P(x,y) - P(x+1, y+1)| + |P(x+1, y) - P(x, y+1)|`.
6. **Median Noise Reduction:** Smoothing the image and removing noise by sorting neighboring pixels (3x3 grid) and extracting the median value.

## 💻 Tech Stack
- **Language:** Python
- **GUI:** Tkinter
- **Libraries:** NumPy (for core matrix/array calculations), Pillow/PIL (for basic image reading/saving)

## 🚀 How to Run
1. Ensure Python and required libraries are installed (`pip install numpy pillow`).
2. Place a sample image named `gambar.jpg` in the same directory as the script.
3. Run the script: `python nama_file.py`
