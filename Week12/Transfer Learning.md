## Topic 3: Transfer Learning

### The Core Concept

```
Pre-train a large model on a large dataset (e.g. ImageNet: 1.2M images, 1000 classes)
→ The model learns general features (edges, shapes, textures, patterns)

Fine-tune on your small task:
→ Replace final classification head
→ Freeze early layers (general features)
→ Unfreeze later layers (task-specific)
→ Train with small learning rate
```

**The intuition in plain English:** training a large neural network from scratch requires huge amounts of data and compute. But it turns out you rarely need to start from scratch — someone else has probably already trained a model on a massive dataset, and the early layers of that model learned things that are *useful for almost any visual task*, not just the original one. So instead of reinventing those layers, you borrow them and only retrain the part that's specific to your problem.

### Why Transfer Learning Works — the Layer Hierarchy

```
Layer 1: detects edges and colours
Layer 2: detects corners and textures
Layer 3: detects object parts        → These generalise across tasks
Layer N: detects specific classes    → This needs to be replaced
```

Think of it like education: early layers learn "reading and arithmetic" — universally useful skills. The final layers learn something like "how to answer questions on *this specific* exam." When you move to a new exam (a new task), you keep the reading/arithmetic skills but need to relearn the exam-specific part.

Concretely for images: layer 1 might learn to detect edges and blobs of color — that's true whether you're classifying cats, X-rays, or satellite photos. But the *very last* layer, which maps features to "this is a Golden Retriever" vs "this is a Persian cat," is specific to the original 1000 ImageNet classes — useless for your new task, so it gets thrown away and replaced.

### Using Pre-trained Features with sklearn

Since this course is sklearn-based (not a deep learning framework), the practical workflow is: use a deep learning framework (TensorFlow/Keras or PyTorch) *only* to extract features from a pre-trained CNN, then hand those features to a normal sklearn classifier.

```python
# Extract features from a pre-trained model
# (Without a deep learning framework, use saved feature vectors)

# Example: using MobileNet features from TensorFlow/Keras
# import tensorflow as tf
# base_model = tf.keras.applications.MobileNetV2(include_top=False, pooling='avg')
# features = base_model.predict(images)  # shape: (n_samples, 1280)

# Then treat features as tabular input to sklearn
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# features already extracted by CNN
clf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=10, probability=True))
])
clf.fit(features_train, y_train)
print(f"Accuracy: {clf.score(features_test, y_test):.4f}")
```

**What's happening here, step by step:**
1. `MobileNetV2(include_top=False, pooling='avg')` loads a model pre-trained on ImageNet, but **strips off** its final classification head (`include_top=False`) — you only want the "feature extractor" part.
2. `pooling='avg'` collapses the spatial feature maps down into a single flat vector per image (1280 numbers).
3. `base_model.predict(images)` runs your images through this frozen network — you're not training anything here, just extracting a rich numeric summary of each image.
4. Those 1280-dimensional vectors are now just tabular data — so you plug them straight into any normal sklearn classifier (here, an SVM), exactly like you would with regular tabular features from earlier in the course. This is called **feature extraction**, the simplest transfer learning strategy.

### Transfer Learning Strategies — When to Use Which

| Strategy | When | Description |
|---|---|---|
| Feature extraction | Small dataset | Freeze the entire pre-trained model, only train a new head on top |
| Fine-tuning (partial) | Medium dataset | Unfreeze just the last few layers, retrain those along with the head |
| Fine-tuning (full) | Large dataset | Unfreeze the whole network, retrain everything with a small learning rate |
| Domain adaptation | Very different domain (e.g. ImageNet photos → medical X-rays) | Needs specialized techniques, since the source and target look very different |

**The pattern to remember:** the more data you have for your specific task, the more of the pre-trained network you're allowed to retrain, because you have enough examples to safely adjust those weights without overfitting. With little data, you freeze almost everything and just learn a thin new layer on top.

---

That covers Transfer Learning. Next is **Topic 4: Complete Course Synthesis** — the "universal ML workflow" that ties together everything from all 12 weeks, plus the algorithm-selection cheat sheet. Want to continue there?
