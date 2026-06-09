import gc
import argparse
import numpy as np
import cupy as cp

from warpfield.warp import warp_volume
from merfish3danalysis.utils.registration import compute_warpfield


def correct_deformation(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    tomove_image: np.ndarray,
    gpu_id: int = 0,
):
    """
    Estimate anisotropic deformation from:
        reference -> moving

    Apply same deformation to:
        moving + tomove
    """

    # Compute warp field
    moving_corrected, warp_field, block_size, block_stride = compute_warpfield(
        reference_image,
        moving_image,
        gpu_id=gpu_id,
    )

    # Apply deformation to second channel

    cp.cuda.Device(gpu_id).use()

    tomove_corrected_cp = warp_volume(
        tomove_image.astype(np.float32),
        warp_field.astype(np.float32),
        cp.asarray(block_stride, dtype=cp.float32),
        cp.asarray(-block_size / block_stride / 2, dtype=cp.float32),
        gpu_id=gpu_id,
    )

    tomove_corrected = cp.asnumpy(tomove_corrected_cp)


    del tomove_corrected_cp
    gc.collect()

    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return (
        moving_corrected.astype(np.float32),
        tomove_corrected.astype(np.float32),
        warp_field,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply GPU deformation correction (warpfield optical flow)."
    )

    parser.add_argument("--reference", required=True, help="Reference image (.npy)")
    parser.add_argument("--moving", required=True, help="Moving image (.npy)")
    parser.add_argument("--tomove", required=True, help="Image to apply same correction")
    parser.add_argument("--out_moving", required=True, help="Output corrected moving")
    parser.add_argument("--out_tomove", required=True, help="Output corrected tomove")
    parser.add_argument("--out_warp", required=True, help="Output warp field (.npy)")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()

    reference = np.load(args.reference).astype(np.float32)
    moving = np.load(args.moving).astype(np.float32)
    tomove = np.load(args.tomove).astype(np.float32)

    moving_corr, tomove_corr, warp_field = correct_deformation(
        reference,
        moving,
        tomove,
        gpu_id=args.gpu,
    )

    np.save(args.out_moving, moving_corr)
    np.save(args.out_tomove, tomove_corr)
    np.save(args.out_warp, warp_field)

    print("Deformation correction complete.")


if __name__ == "__main__":
    main()
