# TryNewProject

Features of image patches are extracted:
. image patch of specific size containing the ground truth in the middle of the image (if the image patch is positive)
. generation of rectangles fitting in that patch
. features are histograms of orientation per rectangle
. histograms are made by partitioning the image gradients into 8 orientations
