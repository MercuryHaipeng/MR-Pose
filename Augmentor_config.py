# =====================================================================
# Degradation-Aware Data Augmentation Pipeline
# =====================================================================
# Objective:
#   1. Improve robustness to common image degradations.
#   2. Avoid excessive combinations of auxiliary degradations.
#   3. Preserve the geometric structure of human keypoints.
#   4. Keep the overall augmentation strength moderate so that the model
#      does not deviate from the original pose estimation task.
# =====================================================================

import albumentations as A
import cv2

# ---------------------------------------------------------------------
# Augmentation Pipeline
# ---------------------------------------------------------------------
transforms_list = [
    # -----------------------------------------------------------------
    # Common Accompanying Degradations
    # -----------------------------------------------------------------
    """
    Simulates secondary artifacts that may accompany image degradation, such as edge blurring and JPEG compression artifacts.
    OneOf ensures that Gaussian blur and JPEG compression are mutually exclusive, preventing multiple simultaneous degradations from excessively damaging the input image. 
    """
    A.OneOf(
        [
            A.GaussianBlur(sigma_limit=(0.5, 3.0), blur_limit=(0, 0), p=1.0),
            A.ImageCompression(compression_type="jpeg", quality_range=(40, 85), p=1.0),
        ],
        p=0.30,
    ),
	# -----------------------------------------------------------------
    # Spatial Structure Perturbations
    # -----------------------------------------------------------------
    """
    Introduces mild spatial perturbations to improve robustness to local information loss and geometric distortions.
    The two transformations are mutually exclusive through OneOf.
    This prevents the simultaneous application of multiple strong spatial perturbations that could severely disrupt human topology.
    """
    A.OneOf(
        [
            """
            # Simulates local loss of visual information around joints
            # or body parts. The small hole size prevents the task from
            # degenerating into an explicit human-part completion task.
            """
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.02, 0.05),
                hole_width_range=(0.02, 0.05),
                fill="random_uniform",
                p=1.0,
            ),
            """            
            Applies a mild nonlinear spatial deformation to break overly regular image structures while preserving the overall human skeletal geometry.
            """
            A.GridDistortion(
                num_steps=5,  
                distort_limit=(-0.08, 0.08),  
                interpolation=cv2.INTER_LINEAR,  
                normalized=True,
                keypoint_remapping_method="direct",  
                p=1.0,
            ),
        ],
        p=0.15,  
    ),
    # -----------------------------------------------------------------
    # Color Quantization
    # -----------------------------------------------------------------
    """
    Simulates reduced color precision and palette quantization, which may introduce abrupt color transitions and discrete intensity changes.
    This transformation is intentionally applied with a low probability because color quantization is an auxiliary degradation rather than the primary privacy transformation.
    """
    A.Posterize(
        num_bits=(4, 6),
        p=0.08,  
    ),
]

# ---------------------------------------------------------------------
# Keypoint Configuration
# ---------------------------------------------------------------------
"""
The keypoints are represented using the (x, y) coordinate format.
Invisible keypoints are retained so that the number and ordering of keypoints remain consistent across samples, which is important for pose estimation training.
"""
keypoint_params = A.KeypointParams(
    format="xy",  
    remove_invisible=False,  
)

# ---------------------------------------------------------------------
# Compose the Complete Augmentation Pipeline
# ---------------------------------------------------------------------
transform = A.Compose(
    transforms=transforms_list,
    keypoint_params=keypoint_params,
    strict=True
)