import gc
import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import tifffile as tiff
from numpy.typing import ArrayLike

from warpfield.warp import warp_volume
import gc
from collections.abc import Sequence


'''
#create environment without merfish install
conda create -n warpfield python=3.11
conda activate warpfield
conda install -c conda-forge cupy cuda-version=12 
pip install warpfield
python -m pip install matplotlib tifffile
'''


def compute_warpfield(
    img_ref: ArrayLike, img_trg: ArrayLike, gpu_id: int = 0
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """
    Compute the warpfield to warp a target image to a reference image.

    Parameters
    ----------
    img_ref: ArrayLike
        reference image
    img_trg: ArrayLike
        moving image
    gpu_id: int, default 0
        GPU ID to use for computation

    Returns
    -------
    warp_field: ArrayLike
        warpfield matrix
    """
    import cupy as cp

    cp.cuda.Device(gpu_id).use()

    from warpfield import Recipe, register_volumes

    recipe = (
        Recipe()
    )  # initialized with a translation level, followed by an affine registration level
    recipe.pre_filter.clip_thresh = 0  # clip DC background, if present
    recipe.pre_filter.soft_edge = [4, 32, 32]

    # affine level properties
    recipe.levels[-1].repeats = 0

    if max(img_ref.shape) > 2048:
        recipe.add_level(block_size=[11, 31, 31])
        recipe.levels[-1].block_stride = 0.85
        recipe.levels[-1].smooth.sigmas = [1.0, 3.0, 3.0]
        recipe.levels[-1].smooth.long_range_ratio = 0.1
        recipe.levels[-1].repeats = 2
        '''
        recipe.add_level(block_size=[5, 15, 15])
        recipe.levels[-1].block_stride = 0.75
        recipe.levels[-1].smooth.sigmas = [1.5, 5.0, 5.0]
        recipe.levels[-1].smooth.long_range_ratio = 0.1
        recipe.levels[-1].repeats = 2
        '''
    else:
        recipe.add_level(block_size=[11, 31, 31])
        recipe.levels[-1].block_stride = 0.75
        recipe.levels[-1].smooth.sigmas = [1.0, 3.0, 3.0]
        recipe.levels[-1].smooth.long_range_ratio = 0.1
        recipe.levels[-1].repeats = 2

        recipe.add_level(block_size=[5, 17, 17])
        recipe.levels[-1].block_stride = 0.75
        recipe.levels[-1].smooth.sigmas = [1.5, 5.0, 5.0]
        recipe.levels[-1].smooth.long_range_ratio = 0.1
        recipe.levels[-1].repeats = 2

    warped_image, warp_map, _ = register_volumes(
        ref=img_ref,
        vol=img_trg,
        recipe=recipe )
    warped_image = cp.asnumpy(warped_image).astype(np.float32)
    warp_field = cp.asnumpy(warp_map.warp_field).astype(np.float32)
    block_size = cp.asnumpy(warp_map.block_size).astype(np.float32)
    block_stride = cp.asnumpy(warp_map.block_stride).astype(np.float32)

    del warp_map
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return (warped_image, warp_field, block_size, block_stride)

def save_overlay_png(reference, moved, out_path, z_slice=None):
    """
    Save RGB overlay:
        - R	reference
        - G	moved
        - B	moved
    """

    def get_slice(img, z_slice=None, axis=0):
        if img.ndim == 3:
            z = img.shape[axis] // 2 if z_slice is None else z_slice
            return np.take(img, z, axis=axis)
        return img

    ref = get_slice(reference).astype(np.float32)
    mov = get_slice(moved).astype(np.float32)

    # Normalize for visualization
    def norm(x):
        x = x - x.min()
        return x / (x.max() + 1e-8)

    ref = norm(ref)
    mov = norm(mov)

    overlay = np.zeros((*ref.shape, 3), dtype=np.float32)
    overlay[..., 0] = ref
    overlay[..., 1] = mov
    overlay[..., 2] = mov

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Reference (red) vs Corrected (cyan)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
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
    cp.cuda.Device(gpu_id).use()

    # Compute warp field
    moving_corrected, warp_field, block_size, block_stride = compute_warpfield(
        reference_image,
        moving_image,
        gpu_id=gpu_id,
    )

    # Apply deformation to second channel
    block_size = cp.asarray(block_size, dtype=cp.float32)
    block_stride = cp.asarray(block_stride, dtype=cp.float32)
    offset = -(block_size / block_stride) / 2

    print('tomove:',type(tomove_image))
    print('warp_field:',type(warp_field))
    print(type('block_stride:',block_stride))
    print(type('offset:',offset))
    
    tomove_corrected_cp = warp_volume(
    cp.asarray(tomove_image, dtype=cp.float32),
    cp.asarray(warp_field, dtype=cp.float32),
    cp.asarray(block_stride, dtype=cp.float32),
    cp.asarray(offset, dtype=cp.float32))

    tomove_corrected = cp.asnumpy(tomove_corrected_cp)

    # CuPy / GPU 
    del tomove_corrected_cp
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    return ( moving_corrected.astype(np.float32), tomove_corrected.astype(np.float32), warp_field)


def main():
    parser = argparse.ArgumentParser( description="Apply GPU deformation correction (warpfield optical flow).")

    parser.add_argument("--reference", required=True, help="Reference image ")
    parser.add_argument("--moving", required=True, help="Moving image ")
    parser.add_argument("--tomove", required=True, help="Image to apply same correction")
    parser.add_argument("--out_moving", required=True, help="Output corrected moving")
    parser.add_argument("--out_tomove", required=True, help="Output corrected tomove")
    parser.add_argument("--out_warp", required=True, help="Output warp field (.npy)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_overlay", required=True, help="Output overlay PNG")

    args = parser.parse_args()

    # Load images
    reference = tiff.imread(args.reference).astype(np.float32)
    moving = tiff.imread(args.moving).astype(np.float32)
    tomove = tiff.imread(args.tomove).astype(np.float32)
    moving = moving[:, ::2, ::2]
    reference = reference[:, ::2, ::2]
    # Deformation correction
    moving_corr, tomove_corr, warp_field = correct_deformation(
        reference,
        moving,
        tomove,
        gpu_id=args.gpu,
    )

    # Saving outputs
    tiff.imwrite(args.out_moving, moving_corr.astype(np.float32))
    tiff.imwrite(args.out_tomove, tomove_corr.astype(np.float32))
    np.save(args.out_warp, warp_field)
    save_overlay_png(reference=reference, moved=moving_corr, out_path=args.out_overlay )
    print("Deformation correction complete.")


if __name__ == "__main__":
    main()
