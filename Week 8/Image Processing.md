## 6. Image Processing for Classical ML

This section covers how to convert images into numerical feature vectors so you can feed them into standard sklearn classifiers (the same idea as text vectorization, just for pixels).

### The Pipeline: Load → Grayscale → Resize → Flatten → Normalise

```python
from PIL import Image
import numpy as np

def image_to_vector(path, size=(64,64), grayscale=True):
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')      # 1 channel
    else:
        img = img.convert('RGB')    # 3 channels
    img = img.resize(size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.flatten()
```

Let's walk through each step and *why* it's necessary:

- **`Image.open(path)`** — loads the image file.
- **`img.convert('L')`** — converts to grayscale (1 channel: just brightness, no color). `'RGB'` keeps 3 channels (Red, Green, Blue). Grayscale drastically reduces the amount of data (1/3 the values) — useful when color isn't essential to the task (e.g., recognizing handwritten digits).
- **`img.resize(size, Image.LANCZOS)`** — **this step is critical**. Every image in your dataset needs to be the *exact same dimensions*, because classical ML models require fixed-length input vectors. A 400×300 photo and a 1200×800 photo can't both become valid inputs unless you force them to a common size like 64×64. `LANCZOS` is a high-quality resampling algorithm used during the resize (better quality than simple nearest-neighbor).
- **`np.array(img, dtype=np.float32) / 255.0`** — converts the image to a NumPy array and normalizes pixel values from the raw range [0, 255] down to [0.0, 1.0]. This normalization helps most ML algorithms train better (avoids one feature's raw scale dominating others).
- **`arr.flatten()`** — this is the key "vectorization" step. A 64×64 grayscale image is a 2D array (64 rows × 64 columns = 4096 values). `flatten()` squashes it into a single 1D vector of length 4096, exactly like how BoW turns a sentence into a 1D vector of word counts. This is literally the same trick as text vectorization — turn structured data into a flat list of numbers.

### Building the full dataset

```python
X_images = np.array([image_to_vector(p) for p in image_paths])
# Shape: (n_images, size*size) or (n_images, size*size*3)
print(X_images.shape)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_images_train, y_train)
```

Once every image is flattened into a fixed-length vector, `X_images` becomes a standard 2D matrix (rows = images, columns = pixel features) — exactly the shape any sklearn classifier expects. From here, it's identical to any other classical ML problem: `.fit(X, y)`.

### Image Feature Extraction Options (comparison table)

| Method | Features | Best for |
|---|---|---|
| Pixel flatten | H×W or H×W×C | Small images, baselines |
| Histogram of pixels | Fixed bins | Lighting-invariant |
| HOG (Histogram of Oriented Gradients) | Edge patterns | Object detection |
| Pre-trained CNN features | 512–2048 dim | Transfer learning (Week 12) |
| PCA on pixels | Reduced dim | Visualisation, speed |

**Why not always just flatten raw pixels?** Raw pixel flattening is a fine *baseline*, but it has a big weakness: it's extremely sensitive to small shifts, rotations, or lighting changes. Two images of the same cat, one slightly brighter than the other, will look very different pixel-by-pixel — even though a human sees them as basically identical.

That's where **HOG** comes in.

### HOG (Histogram of Oriented Gradients)

```python
from skimage.feature import hog
from skimage import exposure

# HOG features (much better than raw pixels)
def extract_hog(img_array):
    fd, hog_image = hog(img_array, orientations=8, pixels_per_cell=(8,8),
                         cells_per_block=(2,2), visualize=True, channel_axis=-1)
    return fd

X_hog = np.array([extract_hog(img) for img in images])
```

**Intuition:** Instead of using raw brightness values, HOG looks at the *edges and gradients* (how brightness changes direction) in small local regions of the image. This captures shape and structure — which is far more robust to lighting changes than raw pixels, and is especially good for detecting objects by their outline/silhouette.

- **`orientations=8`** — bins gradient directions into 8 angle buckets (e.g., roughly every 45°).
- **`pixels_per_cell=(8,8)`** — divides the image into small 8×8 pixel cells; a mini-histogram of gradient directions is computed per cell.
- **`cells_per_block=(2,2)`** — groups cells into 2×2 blocks for local contrast normalization (helps further with lighting invariance).

The output `fd` (feature descriptor) is a 1D vector — same idea as flattening, but built from *edge information* rather than raw brightness, so it generally performs much better for classical ML on real-world images (which is why the table lists it as "much better than raw pixels").

---

Next up: **Section 7 — Named Entity Recognition & Simple NLP with spaCy** (the final section, plus the Key Takeaways summary and the pen-and-paper Bag of Words example). Continue?
