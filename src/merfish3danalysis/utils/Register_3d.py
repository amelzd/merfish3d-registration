import gc
import argparse
import numpy as np
import cupy as cp
import os
import matplotlib.pyplot as plt
import tifffile as tiff

from numpy.typing import ArrayLike
from scipy.ndimage import shift as shift_image
from scipy.ndimage import zoom

from warpfield.warp import warp_volume
from collections.abc import Sequence


'''
#create environment without merfish install
conda create -n warpfield python=3.11
conda activate warpfield
conda install -c conda-forge cupy cuda-version=12 
pip install warpfield
python -m pip install matplotlib tifffile
'''

######### compute_warpfield()  from utils/registration.py ####################
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

def save_overlay_png(reference, moved, out_path, z_slice=None, axis=0):
    """
    Save red/green overlay for 3D microscopy:
        - Red   = reference
        - Green = moved
        - Yellow = overlap
    """

    def get_slice(img):
        if img.ndim != 3:
            return img.astype(np.float32)

        z = img.shape[axis] // 2 if z_slice is None else z_slice
        return np.take(img, z, axis=axis).astype(np.float32)

    ref = get_slice(reference)
    mov = get_slice(moved)

    # normalization 
    def normalize(a, vmin, vmax):
        denom = vmax - vmin
        if denom < 1e-8:
            return np.zeros_like(a, dtype=np.float32)
        return (a - vmin) / denom

    vmin = min(ref.min(), mov.min())
    vmax = max(ref.max(), mov.max())

    ref = normalize(ref, vmin, vmax)
    mov = normalize(mov, vmin, vmax)

    #  RGB overlay
    overlay = np.zeros((*ref.shape, 3), dtype=np.float32)
    overlay[..., 0] = ref  # Red
    overlay[..., 1] = mov  # Green
    
    plt.figure(figsize=(6, 6))
    plt.imshow(overlay, interpolation="nearest")
    plt.axis("off")
    plt.title("Reference (red) vs Moved (green)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

    
def correct_deformation(
    reference_image: np.ndarray,
    moving_image: np.ndarray,
    shift_global=[0,0,0],
    tomove_image: np.ndarray | None = None,
    gpu_id: int = 0):
        
    """
    Estimate anisotropic deformation from:
        reference -> moving

    Apply same deformation to:
        moving + tomove
    """
    cp.cuda.Device(gpu_id).use()

    # apply register global
    shift_global = np.asarray(shift_global, dtype=np.float32)
    shift_3d = shift_global[[2, 1, 0]]
    moving_image = shift_image(moving_image, shift_3d)

    # Compute warp field with gpu
    moving_corrected, warp_field, block_size, block_stride = compute_warpfield(
        reference_image,
        moving_image,
        gpu_id=gpu_id,
    )

    # Apply deformation to second channel
    block_size = cp.asarray(block_size, dtype=cp.float32)
    block_stride = cp.asarray(block_stride, dtype=cp.float32)
    offset = -(block_size / block_stride) / 2

    #### 3D correction applied (warpfield) to CHANGE
    tomove_corrected = None
    if tomove_image is not None:
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

    return (moving_corrected.astype(np.float32), None if tomove_image is None else tomove_corrected.astype(np.float32),warp_field)



#####register3D deeds overlay 
class BothImgRbgFile:
    def __init__(self, image1, image2, tag='', title=''):
        self.image1 = image1
        self.image2 = image2
        self.tag = tag
        if title is None:
            self.title = tag  # gets title from tag
        else:
            self.title = title  # New attribute to hold the title

    def save(self, folder_path, basename):
        self.folder_path = folder_path
        self.basename = f"{basename}_{self.tag}_overlay"
        self.path_name = os.path.join(self.folder_path, self.basename + ".png")
        
        # Normalize images and rescale intensity
        img_1 = self.image1 / self.image1.max()
        img_2 = self.image2 / self.image2.max()
        img_1 = exposure.rescale_intensity(img_1, out_range=(0, 1))
        img_2 = exposure.rescale_intensity(img_2, out_range=(0, 1))
        
        # Create the figure and axis
        fig, ax1 = plt.subplots()
        fig.set_size_inches((30, 30))
        
        # Create RGB overlay image
        null_image = np.zeros(img_1.shape)
        rgb = np.dstack([img_1, img_2, null_image])
        
        # Display the image and set the title
        ax1.imshow(rgb)
        ax1.axis("off")
        ax1.set_title(self.title)  # Set the title of the figure
        
        # Save the figure
        fig.savefig(self.path_name)
        plt.close(fig)


def plot_4_images(allimages, titles=None):
    if titles is None:
        titles = [
            "reference",
            "cycle <i>",
            "processed reference",
            "processed cycle <i>",
        ]

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches((10, 10))
    ax = axes.ravel()

    for axis, img, title in zip(ax, allimages, titles):
        axis.imshow(img, cmap="Greys")
        axis.set_title(title)
    fig.tight_layout()

    return fig

def main():
    parser = argparse.ArgumentParser( description="Apply GPU deformation correction (warpfield optical flow).")

    parser.add_argument("--reference", required=True, help="Reference image ")
    parser.add_argument("--moving", required=True, help="Moving image ")
    parser.add_argument("--tomove", default = None, help="Image to apply same correction")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bin", type=int, default=2)
    parser.add_argument("--shift_global", nargs=3, type=float, default=[0,0,0])

    args = parser.parse_args()

    # Load images
    print("Reading images.")    
    reference = tiff.imread(args.reference).astype(np.float32)
    moving = tiff.imread(args.moving).astype(np.float32)
    orig_dtype_mov = tiff.imread(args.moving).dtype
    
    tomove_dtype = None
    tomove = None
    if args.tomove is not None:
        orig_dtype = tiff.imread(args.tomove).dtype
        tomove = tiff.imread(args.tomove).astype(np.float32)
    
    #binning
    '''
    if args.bin > 1:
        moving = moving[:, ::args.bin, ::args.bin]
        reference = reference[:, ::args.bin, ::args.bin]
    '''

    if args.bin > 1:
        scale = 1.0 / args.bin
        moving = zoom(moving, (1, scale, scale), order=1)
        reference = zoom(reference, (1, scale, scale), order=1)
    
    shift_global=args.shift_global
    
    print("Correcting deformations.")
    # Deformation registration
    moving_corr, tomove_corr, warp_field = correct_deformation(
        reference,
        moving,
        shift_global,
        tomove,
        gpu_id=args.gpu )
    
    print(f"Saving images in: {args.output}")
    # Saving outputs
    
    tiff.imwrite(os.path.join(args.output, os.path.basename(args.moving)), np.clip(moving_corr, np.iinfo(orig_dtype_mov).min, np.iinfo(orig_dtype_mov).max).astype(orig_dtype_mov) )
    if tomove_corr is not None and args.tomove is not None:
        tiff.imwrite(os.path.join(args.output, os.path.basename(args.tomove)), np.clip(tomove_corr, np.iinfo(orig_dtype).min,  np.iinfo(orig_dtype).max,).astype(orig_dtype) )
    np.save(os.path.join(args.output, "warp_field.npy"), warp_field)

    # rgb overlay marcelo
    # usage :overlay = BothImgRbgFile(fixed_image_np.max(axis=0), moving_image_np.max(axis=0), tag='reference_original')
    overlay = BothImgRbgFile(reference.max(axis=0), moving.max(axis=0), tag='reference_original')
    overlay.save(os.path.join(args.output,args.moving))

    overlay = BothImgRbgFile(reference.max(axis=0), moving_corr.max(axis=0), tag='reference_aligned')
    overlay.save(os.path.join(args.output,args.moving))
    #save_overlay_png(reference=reference, moved=moving_corr, out_path=args.out_overlay )
    
    print("Deformation correction complete.")


if __name__ == "__main__":
    main()
