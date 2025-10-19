from skimage.morphology import label,binary_erosion, disk
from skimage.measure import regionprops
import numpy as np

def compute_metrics(img):
    """
    Compute metrics for the given image.

    Parameters:
    img (ndarray): Input image.

    Returns:
    tuple: A tuple containing the centroid coordinates, area, major axis length, and minor axis length.
    """
    # Label the image
    label_img = label(img)
    regions=regionprops(label_img)

    props = max(regions, key=lambda r: r.area)
    #props = regions[0]
    y0, x0 = props.centroid # in coordinate immagine
    area = props.area
    max_diam = props.major_axis_length # The length of the major axis of the ellipse that has the same normalized second central moments as the region.
    min_diam = props.minor_axis_length
    orientation = props.orientation # The angle of the major axis of the ellipse that has the same normalized second central moments as the region.
    return (x0, y0), area, max_diam, min_diam, orientation
    
def erode_mask(mask_data):
    """
    Erode the binary mask using a disk-shaped structuring element.

    Parameters:
    mask (ndarray): Input binary mask (uint8).

    Returns:
    ndarray: Eroded binary mask (uint8).
    """
    # Convert to binary: True for 255, False for 0
    binary_mask = mask_data >= 128  # thresholding

    # Apply binary erosion with a disk-shaped structuring element
    selem = disk(5)  # you can change the radius
    eroded_mask = binary_erosion(binary_mask, selem)

    # Convert back to uint8 with 0 and 255
    return  (eroded_mask * 255).astype(np.uint8)