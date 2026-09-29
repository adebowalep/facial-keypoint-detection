# Archive: original Udacity submission

This directory preserves the original, graded submission for Udacity's
"Facial Keypoint Detection" project (Computer Vision Nanodegree), unmodified,
for provenance.

Contents:

- `notebooks/` — the four original notebooks (`1. Load and Visualize Data`
  through `4. Fun with Keypoints`), with their original executed outputs
- `models.py` — the CNN architecture as submitted and graded
- `data_load.py` — the dataset/transform code as provided by the Udacity
  starter repository and used as-is

**What Udacity supplied:** the project skeleton and TODOs in all four
notebooks, `data_load.py`'s `FacialKeypointsDataset` and transform classes
(`Rescale`, `RandomCrop`, `Normalize`, `ToTensor`), the Haar cascade XML files
in `detector_architectures/`, the sample images in `images/`, and the
underlying Kaggle-derived facial keypoints dataset.

**What was completed independently for the graded submission:** the CNN
architecture in `models.py`, the `data_transform` composition and
loss/optimizer choice in Notebook 2, the training loop invocation and
hyperparameter choices, the feature-map visualization, and the
detect-preprocess-predict-display pipeline in Notebook 3. This passed Udacity
review against the project rubric.

**Why this is archived rather than deleted:** everything outside this
directory (the `facial_keypoints` package, the Gradio app, the test suite,
the bug fix described in the top-level README, and this repository's
structure) is a from-scratch portfolio refactor built on top of this
submission — reusing its trained weights (to avoid re-paying for GPU
training) and its core CNN design, while rebuilding the surrounding
engineering from the ground up. Keeping the original untouched here makes
that lineage checkable rather than asserted.
