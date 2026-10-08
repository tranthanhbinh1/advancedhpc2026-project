import math
import sys

import matplotlib.image as mpimg
import numpy as np

_HIP_ACTIVE = False
_HIP_IMPORT_ERROR = None
try:
    from numba import hip as _hip

    if _hip.is_available():
        # For my AMD card
        _hip.pose_as_cuda()
        _HIP_ACTIVE = True
except Exception as exc:
    _HIP_IMPORT_ERROR = exc

from numba import cuda as _gpu


SEARCH_SIZE = 21
PATCH_SIZE = 7
SMOOTHING = 10.0
SEARCH_RADIUS = SEARCH_SIZE // 2
PATCH_RADIUS = PATCH_SIZE // 2
PADDING = SEARCH_RADIUS + PATCH_RADIUS
CHANNELS = 3
BLOCK_SIZE = 16
OUTPUT_PATH = "output.jpg"


def _select_gpu_backend():
    """Return the active Numba GPU module, or explain why none is usable."""
    try:
        if _gpu.is_available():
            return _gpu, "HIP" if _HIP_ACTIVE else "CUDA"
    except Exception as exc:
        raise RuntimeError(f"GPU runtime initialization failed: {exc}") from exc

    if _HIP_IMPORT_ERROR:
        detail = f"HIP import failed: {_HIP_IMPORT_ERROR}"
    else:
        detail = "no device found"
    raise RuntimeError(f"No usable AMD HIP or NVIDIA CUDA device ({detail})")


@_gpu.jit
def _denoise_kernel(image, output, width, height):
    x, y = _gpu.grid(2)
    if x >= width or y >= height:
        return

    center_x = x + PADDING
    center_y = y + PADDING
    weighted_red = 0.0
    weighted_green = 0.0
    weighted_blue = 0.0
    weight_sum = 0.0

    for search_y in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        candidate_y = center_y + search_y
        for search_x in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            candidate_x = center_x + search_x
            squared_error = 0.0

            for patch_y in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
                reference_y = center_y + patch_y
                comparison_y = candidate_y + patch_y
                for patch_x in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
                    reference_x = center_x + patch_x
                    comparison_x = candidate_x + patch_x
                    for channel in range(CHANNELS):
                        difference = (
                            image[reference_y, reference_x, channel]
                            - image[comparison_y, comparison_x, channel]
                        )
                        squared_error += difference * difference

            mean_squared_error = squared_error / (PATCH_SIZE * PATCH_SIZE * CHANNELS)
            weight = math.exp(-mean_squared_error / (SMOOTHING * SMOOTHING))
            weighted_red += weight * image[candidate_y, candidate_x, 0]
            weighted_green += weight * image[candidate_y, candidate_x, 1]
            weighted_blue += weight * image[candidate_y, candidate_x, 2]
            weight_sum += weight

    output[y, x, 0] = weighted_red / weight_sum
    output[y, x, 1] = weighted_green / weight_sum
    output[y, x, 2] = weighted_blue / weight_sum


def _load_rgb_image(path):
    image = np.asarray(mpimg.imread(path))
    if image.ndim != 3 or image.shape[2] != CHANNELS:
        raise ValueError("input image must be RGB with exactly three channels")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("input image must have nonzero width and height")

    if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
        image = image * 255.0
    return np.ascontiguousarray(image, dtype=np.float32)


def denoise(image):
    height, width, channels = image.shape
    if channels != CHANNELS:
        raise ValueError("input image must have exactly three RGB channels")

    gpu, backend_name = _select_gpu_backend()
    padded = np.pad(
        image,
        ((PADDING, PADDING), (PADDING, PADDING), (0, 0)),
        mode="reflect",
    )
    device_image = gpu.to_device(padded)
    device_output = gpu.device_array((height, width, CHANNELS), dtype=np.float32)
    blocks = (
        (width + BLOCK_SIZE - 1) // BLOCK_SIZE,
        (height + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )
    _denoise_kernel[blocks, (BLOCK_SIZE, BLOCK_SIZE)](
        device_image, device_output, width, height
    )
    result = device_output.copy_to_host()
    result = np.rint(result).clip(0, 255).astype(np.uint8)
    return result, backend_name


def main(argv=None):
    argv = sys.argv if argv is None else argv
    if len(argv) != 2:
        print(f"Usage: {argv[0]} INPUT.jpg", file=sys.stderr)
        return 2

    input_path = argv[1]
    try:
        image = _load_rgb_image(input_path)
    except Exception as exc:
        print(f"Could not load RGB image '{input_path}': {exc}", file=sys.stderr)
        return 1

    try:
        output, backend_name = denoise(image)
        mpimg.imsave(OUTPUT_PATH, output)
    except Exception as exc:
        print(f"Denoising failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {OUTPUT_PATH} using {backend_name}")
    return 0


if __name__ == "__main__":
    main()
