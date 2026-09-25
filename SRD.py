import cv2
import numpy as np
import random
from collections import Counter
import albumentations as A


class SRDForYOLOPose:
    """
    Spectral Radius Dropout (SRD):
    A frequency-domain dropout augmentation that performs stochastic spectral suppression based on the radial distance from the frequency origin. The cutoff radius is dynamically sampled according to dataset-specific spectral statistics.
    """
    def __init__(self, dataset_type: str = "coco"):
        """
        Initialize the Spectral Radius Dropout (SRD) augmenter.
        Args:
            dataset_type: Dataset type used to select the corresponding spectral statistics. Supported options are: 'coco', 'crowd', 'mpii', and 'hand'.
        """
        self.dataset_type = dataset_type

        # Define a fixed number of radial bins. This value must be strictly consistent with the configuration used when extracting the prior spectral statistics.
        self.NUM_BINS = 256

        # Load the complete dataset-specific spectral statistics.
        self.full_stats = self._load_full_stats()

        # Initialize a cache for the precomputed logarithmic radial distance matrices to reduce redundant computation.
        self.max_cache_size = 10
        self._init_distance_cache()

        # Construct the logarithmic radial-bin sampling pool and corresponding sampling weights.
        self._build_radius_pool()

    def _load_full_stats(self) -> Dict[str, Any]:
        """
        Load comprehensive statistical profiles using a bin resolution of 256.
        """
        stats = {
            "coco": {
                "scale_0": {"80": 25, "85": 51, "90": 78, "95": 123, "99": 201},
                "scale_4": {"80": 25, "85": 41, "90": 67, "95": 102, "99": 160},
                "scale_8": {"80": 25, "85": 41, "90": 60, "95": 92, "99": 146},
                "scale_12": {"80": 25, "85": 33, "90": 53, "95": 86, "99": 139},
                "scale_16": {"80": 0, "85": 25, "90": 51, "95": 79, "99": 134},
            },
            "mpii": {
                "scale_0": {"80": 25, "85": 50, "90": 77, "95": 118, "99": 183},
                "scale_4": {"80": 25, "85": 43, "90": 72, "95": 110, "99": 170},
                "scale_8": {"80": 25, "85": 41, "90": 67, "95": 103, "99": 159},
                "scale_12": {"80": 25, "85": 41, "90": 63, "95": 97, "99": 152},
                "scale_16": {"80": 25, "85": 41, "90": 60, "95": 93, "99": 147},
            },
            "crowd": {  
                "scale_0": {"80": 41, "85": 61, "90": 89, "95": 127, "99": 191},  
                "scale_4": {"80": 33, "85": 53, "90": 81, "95": 114, "99": 164},
                "scale_8": {"80": 25, "85": 51, "90": 74, "95": 105, "99": 155},
                "scale_12": {"80": 25, "85": 43, "90": 67, "95": 98, "99": 150},
                "scale_16": {"80": 25, "85": 41, "90": 67, "95": 93, "99": 146},
            },  # train
            "hand": {
                "scale_0": {"80": 0, "85": 0, "90": 41, "95": 67, "99": 124},
                "scale_4": {"80": 0, "85": 0, "90": 25, "95": 60, "99": 113},
                "scale_8": {"80": 0, "85": 0, "90": 25, "95": 53, "99": 103},
                "scale_12": {"80": 0, "85": 0, "90": 25, "95": 51, "99": 96},
                "scale_16": {"80": 0, "85": 0, "90": 25, "95": 43, "99": 93},
            },
        }
        return stats

    def _build_radius_pool(self):
        dataset_stats = self.full_stats[self.dataset_type]
        radii = []
        for scale in dataset_stats.values():
            radii.extend(scale.values())
        radii = [r for r in radii if r > 0]
        count = Counter(radii)
        self.unique_radii = sorted(count.keys())
        self.radius_weights = [count[r] for r in self.unique_radii]


    # ======= Sample cutoff radius =========================================
    def _get_radius(self) -> int:
        """
        Sample the spectral cutoff radius using a dataset-specific upper bound and a low-frequency-biased Beta distribution.
        Returns:
            Selected logarithmic radial-bin index.
        """
        # Set the upper bound using the 90% cumulative-energy radius estimated from the dataset-specific spectral prior.
        max_bin_limit = self.full_stats[self.dataset_type]["scale_0"]["90"]

        # Sample a low-frequency-biased ratio.
        # Beta(1, 3) concentrates samples toward smaller radii.
        beta_sample = random.betavariate(alpha=1.0, beta=3.0)

        # Map the sampled ratio to the valid radial-bin range.
        sampled_idx = int(beta_sample * max_bin_limit)

        # Preserve the lowest-frequency components to avoid excessive
        # spectral suppression.
        final_radius = max(3, sampled_idx)

        return final_radius

    def _init_distance_cache(self):
        self.distance_maps = {}

    def _get_distance_map(self, h, w):
        """Generate a cached logarithmic radial-bin map."""
        key = (h, w)

        if key not in self.distance_maps:
            # Evict the oldest entry when the cache is full.
            if len(self.distance_maps) >= self.max_cache_size:
                oldest_key = next(iter(self.distance_maps))
                del self.distance_maps[oldest_key]

            # Compute the Euclidean radial distance.
            y, x = np.indices((h, w), dtype=np.float32)
            r_physical = np.sqrt(x * x + y * y)
            max_r_physical = np.sqrt((h - 1) ** 2 + (w - 1) ** 2)

            # Map the radial distance to logarithmic spectral bins.
            log_r = np.log1p(r_physical)
            log_max_r = np.log1p(max_r_physical)
            r_bin_map = (log_r / log_max_r * (self.NUM_BINS - 1)).astype(np.int32)
            self.distance_maps[key] = r_bin_map

        return self.distance_maps[key]   

    def _apply_srd_to_channel(
        self, channel: np.ndarray, radius: int
    ) -> np.ndarray:
        """
        Apply Spectral Radius Dropout (SRD) to a single image channel.
        Args:
            channel: Single-channel image.
            radius: Selected cutoff logarithmic-bin index (0–255).
        Returns:
            Spatial-domain image after spectral suppression.
        """
        h, w = channel.shape[:2]

        # Obtain the logarithmic radial-bin index map.
        dist_map = self._get_distance_map(h, w)
        # Transform the image into the frequency domain.
        dct = cv2.dct(channel.astype(np.float32))
        # Suppress frequency components beyond the selected cutoff radius.
        dct[dist_map > radius] = 0
        # Transform back to the spatial domain.
        result = cv2.idct(dct)

        return result

    def apply_to_image(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply Spectral Radius Dropout (SRD) to an input image.
        Args:
            image: Input image in BGR format.
        Returns:
            result: SRD-processed image.
            info: Metadata of the applied spectral dropout.
        """
        # Sample the cutoff radius from the spectral prior.
        radius = self._get_radius()

        if len(image.shape) == 3:
            # Apply SRD to the luminance channel while preserving chromatic
            # information.
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y_channel = ycrcb[:, :, 0]
            y_processed = self._apply_srd_to_channel(y_channel, radius)
            # Restore the processed luminance channel and convert back to BGR.
            ycrcb[:, :, 0] = np.clip(y_processed, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            # Apply SRD directly to grayscale images.
            result = self._apply_srd_to_channel(image, radius)

        # Record the selected radial-bin index and augmentation settings.
        info = {
            "radius_bin_id": radius,
            "mode": self.mode,
            "dataset_type": self.dataset_type,
        }

        return result, info

# =================================================================== #
def create_srd_for_yolo(
    dataset_type: str = "coco",
) -> SRDForYOLOPose:
    return SRDForYOLOPose(dataset_type=dataset_type)


class SRDAlbumentations(A.ImageOnlyTransform):
    """
    Albumentations wrapper for Spectral Radius Dropout (SRD).
    """
    def __init__(self, dataset_type: str = "coco", p: float = 0.5):
        super().__init__(p=p)
        self.dataset_type = dataset_type
        self.srd = create_srd_for_yolo(dataset_type, mode="full")

    def apply(self, image, **params):
        """Apply SRD to the input image."""
        # The input image is already in BGR format.
        result, _ = self.srd.apply_to_image(image)

        return result

    @classmethod
    def get_transform_init_args_names(cls) -> Tuple[str, ...]:
        """Return parameters required for Albumentations serialization."""
        return ("dataset_type", "p")


# == Generate SRD data pipeline ======================================== #
# Select the corresponding dataset Spectrumal Radius
dataset_type = "mpii"	# "coco", "mpii", "crowd", "hand"
srd_prob = 0.5  		# The probality of DCT-SRD
srd_pipeline = A.Compose(
	[ SRDAlbumentations(dataset_type=dataset_type, p=srd_prob), ]
)