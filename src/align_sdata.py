import numpy as np
import os
import re
from numpy.linalg import inv
from pathlib import Path
import math
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import scale

import anndata as ad
from spatialdata import SpatialData
from spatialdata.transformations import (
    Affine,
    BaseTransformation,
    Sequence,
    get_transformation,
    set_transformation,
    get_transformation_between_landmarks
)

ALIGNMENT_MATRICES_PATH = Path('/media/Lynn/alignment/codex_columns')


def postpone_codex_transformation(
    sdata: SpatialData,
    transformation: BaseTransformation,
    source_coordinate_system: str,
    target_coordinate_system: str,
):
    """
    Apply a transformation to all elements in CODEX SpatialData that have an existing
    transformation from the source coordinate system.
    
    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing CODEX data.
    transformation : BaseTransformation
        The transformation to apply.
    source_coordinate_system : str
        The source coordinate system to check for existing transformations.
    target_coordinate_system : str
        The target coordinate system to set the new transformation to.
    """
    for element_type, element_name, element in sdata._gen_elements():
        old_transformations = get_transformation(element, get_all=True)
        if source_coordinate_system in old_transformations:
            old_transformation = old_transformations[source_coordinate_system]
            sequence = Sequence([old_transformation, transformation])
            set_transformation(element, sequence, target_coordinate_system)


def postpone_xenium_transformation(
    sdata: SpatialData,
    transformation: BaseTransformation,
    target_coordinate_system: str,
):
    """
    Apply a transformation to all elements in Xenium SpatialData.
    
    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing Xenium data.
    transformation : BaseTransformation
        The transformation to apply.
    target_coordinate_system : str
        The target coordinate system to set the transformation to.
    """
    for element_type, element_name, element in sdata._gen_elements():
        set_transformation(element, transformation, target_coordinate_system)


def align_images_using_napari_landmarks(
    codex_sdata, 
    xenium_sdata, 
    codex_landmarks, 
    xenium_landmarks, 
    aligned_coordinate_system_name
):
    """
    Align CODEX and Xenium images using landmark points.
    
    This function performs the following steps:
    1. Copy the CODEX image from 'pixels' to 'global' coordinate system
    2. Add landmark point coordinates to the SpatialData objects
    3. Calculate the transformation matrix between landmarks
    4. Apply transformations to align images
    5. Align all other elements in the SpatialData objects
    
    Parameters
    ----------
    codex_sdata : SpatialData
        SpatialData object containing CODEX data.
    xenium_sdata : SpatialData
        SpatialData object containing Xenium data.
    codex_landmarks : DataFrame or array-like
        Landmark coordinates in CODEX image.
    xenium_landmarks : DataFrame or array-like
        Landmark coordinates in Xenium image.
    aligned_coordinate_system_name : str
        Name to give the aligned coordinate system.
    """
    # STEP 1: Copy the CODEX image from the 'pixels' coordinate system to 'global'
    identity_matrix = np.eye(3)  
    identity_affine = Affine(
        identity_matrix, 
        input_axes=("x", "y"),
        output_axes=("x", "y")
    )
    for image_name in codex_sdata.images.keys():
        set_transformation(
            codex_sdata.images[image_name], 
            identity_affine, 
            to_coordinate_system="global"
        )
    
    # STEP 2: Add landmark point coordinates to the sdata objects
    xenium_sdata["xenium_landmarks"] = xenium_landmarks
    codex_sdata["codex_landmarks"] = codex_landmarks
    
    # STEP 3: Get transformation matrix, swap tx and ty coordinates
    affine_matrix = get_transformation_between_landmarks(
        references_coords=xenium_sdata["xenium_landmarks"], 
        moving_coords=codex_sdata["codex_landmarks"]
    )
    affine_array = affine_matrix.matrix.copy()
    affine_array[0, 2], affine_array[1, 2] = affine_array[1, 2], affine_array[0, 2]
    affine_swapped = Affine(
        affine_array, 
        input_axes=("x", "y"),
        output_axes=("x", "y")
    )
    
    # STEP 4: Align Images
    for image_name in codex_sdata.images.keys():
        set_transformation(
            codex_sdata.images[image_name], 
            affine_swapped, 
            to_coordinate_system=aligned_coordinate_system_name
        )
    for image_name in xenium_sdata.images.keys():
        set_transformation(
            xenium_sdata.images[image_name], 
            identity_affine, 
            to_coordinate_system=aligned_coordinate_system_name
        )
    
    # STEP 5: Align all other elements
    postpone_codex_transformation(
        sdata=codex_sdata,
        transformation=affine_swapped,
        source_coordinate_system="global",
        target_coordinate_system=aligned_coordinate_system_name,
    )
    postpone_xenium_transformation(
        sdata=xenium_sdata,
        transformation=identity_affine,
        target_coordinate_system=aligned_coordinate_system_name,
    )


def load_alignment_matrix_from_csv(csv_path):
    """
    Load a 3x3 alignment matrix from a CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the alignment matrix.
        
    Returns
    -------
    numpy.ndarray
        3x3 alignment matrix loaded from the CSV file.
    """
    import numpy as np
    
    # Load the matrix from CSV
    alignment_matrix = np.loadtxt(csv_path, delimiter=',')
    
    # Ensure it's a 3x3 matrix
    if alignment_matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got {alignment_matrix.shape}")
        
    return alignment_matrix


def align_images_using_alignment_matrix(
    codex_sdata, 
    xenium_sdata, 
    alignment_matrix_csv_path
):
    """
    Align CODEX and Xenium images using a pre-computed alignment matrix.
    
    This function performs the following steps:
    1. Copy the CODEX image from 'pixels' to 'global' coordinate system
    2. Create transformation objects from the provided alignment matrix
    3. Apply transformations to align images
    4. Align all other elements in the SpatialData objects
    
    Parameters
    ----------
    codex_sdata : SpatialData
        SpatialData object containing CODEX data.
    xenium_sdata : SpatialData
        SpatialData object containing Xenium data.
    alignment_matrix_csv_path : str
        Path to the CSV file containing the alignment matrix.
        Name to give the aligned coordinate system.
    """
    # STEP 1: Load alignment matrix and create transformation
    alignment_matrix = load_alignment_matrix_from_csv(alignment_matrix_csv_path)
    
    affine = Affine(
        alignment_matrix, 
        input_axes=("x", "y"),
        output_axes=("x", "y")
    )
    
    # STEP 2: Align CODEX image to Xenium
    for image_name in codex_sdata.images.keys():
        set_transformation(
            codex_sdata.images[image_name], 
            affine, 
            to_coordinate_system='global'
        )
    
    # STEP 3: Merge the two SpatialData objects
    # Add CODEX images to Xenium SpatialData
    for image_name, image in codex_sdata.images.items():
        xenium_sdata.images[image_name] = image
    
    # Add CODEX labels, points, shapes, and tables if they exist
    if hasattr(codex_sdata, 'labels') and codex_sdata.labels:
        if not hasattr(xenium_sdata, 'labels'):
            xenium_sdata.labels = {}
        xenium_sdata.labels.update(codex_sdata.labels)
    
    if hasattr(codex_sdata, 'points') and codex_sdata.points:
        if not hasattr(xenium_sdata, 'points'):
            xenium_sdata.points = {}
        xenium_sdata.points.update(codex_sdata.points)
    
    if hasattr(codex_sdata, 'shapes') and codex_sdata.shapes:
        if not hasattr(xenium_sdata, 'shapes'):
            xenium_sdata.shapes = {}
        xenium_sdata.shapes.update(codex_sdata.shapes)
    
    if hasattr(codex_sdata, 'tables') and codex_sdata.tables:
        if not hasattr(xenium_sdata, 'tables'):
            xenium_sdata.tables = {}
        xenium_sdata.tables.update(codex_sdata.tables)
    
    # Store the merged results
    results = xenium_sdata

    return results


def align_slide_columns(
    codex_sdata_dict, 
    xenium_sdata_dict, 
    slide_id,
    alignment_matrices_base_dir = ALIGNMENT_MATRICES_PATH,
):
    """
    Align all columns of CODEX images to corresponding Xenium images for a specific slide 
    and merge the SpatialData objects.
    
    Parameters
    ----------
    codex_sdata_dict : dict
        Dictionary of SpatialData objects containing CODEX data, with keys like "column_1"
    xenium_sdata_dict : dict
        Dictionary of SpatialData objects containing Xenium data, with keys like "column_1"
    alignment_matrices_base_dir : str
        Base directory containing alignment matrices organized in folders by slide and column
    slide_id : str
        ID of the slide to process
    aligned_coordinate_system_name : str, optional
        Name to give the aligned coordinate system, by default "aligned"
    
    Returns
    -------
    dict
        Dictionary with column_ids as keys and merged SpatialData objects as values
    """
    
    results = {}
    
    # Process each column
    for column_id in codex_sdata_dict.keys():
        if column_id not in xenium_sdata_dict:
            print(f"Warning: {column_id} found in CODEX data but not in Xenium data. Skipping.")
            continue
        
        print(f"Processing slide {slide_id}, {column_id}...")
        
        # Get the corresponding SpatialData objects
        codex_sdata = codex_sdata_dict[column_id]
        xenium_sdata = xenium_sdata_dict[column_id]
        
        # Find the alignment matrix file using the pattern ID_slideid_column_x
        folder_pattern = f"ID_{slide_id}_{column_id}"
        
        # Look for a folder matching the pattern
        matching_folders = [f for f in os.listdir(alignment_matrices_base_dir) 
                           if folder_pattern in f]
        
        if not matching_folders:
            print(f"Warning: No alignment matrix folder found for slide {slide_id}, {column_id}. Skipping.")
            continue
            
        matrix_folder = matching_folders[0]  # Take the first match
        matrix_path = os.path.join(alignment_matrices_base_dir, matrix_folder, "matrix.csv")
        
        if not os.path.exists(matrix_path):
            print(f"Warning: Alignment matrix not found at {matrix_path}. Skipping.")
            continue
        
        # STEP 1: Load alignment matrix and create transformation
        alignment_matrix = load_alignment_matrix_from_csv(matrix_path)
        
        affine = Affine(
            alignment_matrix, 
            input_axes=("x", "y"),
            output_axes=("x", "y")
        )
        
        # STEP 2: Align CODEX image to Xenium
        for image_name in codex_sdata.images.keys():
            set_transformation(
                codex_sdata.images[image_name], 
                affine, 
                to_coordinate_system='global'
            )
        
        # STEP 3: Merge the two SpatialData objects
        # Add CODEX images to Xenium SpatialData
        for image_name, image in codex_sdata.images.items():
            xenium_sdata.images[image_name] = image
        
        # Add CODEX labels, points, shapes, and tables if they exist
        if hasattr(codex_sdata, 'labels') and codex_sdata.labels:
            if not hasattr(xenium_sdata, 'labels'):
                xenium_sdata.labels = {}
            xenium_sdata.labels.update(codex_sdata.labels)
        
        if hasattr(codex_sdata, 'points') and codex_sdata.points:
            if not hasattr(xenium_sdata, 'points'):
                xenium_sdata.points = {}
            xenium_sdata.points.update(codex_sdata.points)
        
        if hasattr(codex_sdata, 'shapes') and codex_sdata.shapes:
            if not hasattr(xenium_sdata, 'shapes'):
                xenium_sdata.shapes = {}
            xenium_sdata.shapes.update(codex_sdata.shapes)
        
        if hasattr(codex_sdata, 'tables') and codex_sdata.tables:
            if not hasattr(xenium_sdata, 'tables'):
                xenium_sdata.tables = {}
            xenium_sdata.tables.update(codex_sdata.tables)
        
        # Store the merged results
        results[column_id] = xenium_sdata
        
        print(f"Successfully aligned and merged slide {slide_id}, {column_id}")
    
    return results


def shrink_cell_boundaries_in_sdata(
    sdata, 
    reduction_percentage=0.1, 
    new_shape_name='shrunken_cell_boundaries'
):
    """
    Shrink cell boundaries in a SpatialData object while maintaining their original centroid.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing cell boundaries
    reduction_percentage : float, optional
        Percentage of area reduction (default is 0.1 or 10%)
    new_shape_name : str, optional
        Name to give to the new shrunken cell boundaries shape (default is 'shrunken_cell_boundaries')
    
    Returns
    -------
    SpatialData
        SpatialData object with added shrunken cell boundaries
    """
    def shrink_polygon(polygon):
        """
        Shrink a single polygon while maintaining its centroid
        
        Parameters
        ----------
        polygon : Shapely Polygon
            Input polygon to be shrunk
        
        Returns
        -------
        Shapely Polygon
            Shrunken polygon
        """
        # If the polygon is invalid or too small, return the original
        if not polygon.is_valid or polygon.area == 0:
            return polygon
        
        # Calculate the linear scaling factor
        # Area scales with the square of linear dimensions
        current_area = polygon.area
        target_area = current_area * (1 - reduction_percentage)
        scale_factor = np.sqrt(target_area / current_area)
        
        # Create a scaled polygon
        try:
            # The scaling is done relative to the centroid
            shrunken_polygon = scale(polygon, xfact=scale_factor, yfact=scale_factor, origin='centroid')
            
            return shrunken_polygon
        except Exception as e:
            print(f"Error shrinking polygon: {e}")
            return polygon

    # Check if cell_boundaries exist in shapes
    if 'cell_boundaries' not in sdata.shapes:
        raise ValueError("No 'cell_boundaries' found in SpatialData object shapes")
    
    original_cell_boundaries = sdata.shapes['cell_boundaries']
    
    # Create a copy of the GeoDataFrame to avoid modifying the original
    shrunken_cell_boundaries = original_cell_boundaries.copy()
    
    # Apply the shrinking function to each geometry
    shrunken_cell_boundaries.geometry = shrunken_cell_boundaries.geometry.apply(shrink_polygon)
    
    # Add the new shrunken cell boundaries to the SpatialData object
    sdata.shapes[new_shape_name] = shrunken_cell_boundaries
    
    return sdata
    

def transform_table_coordinates(adata, transformation_matrix):
    """
    Transform spatial coordinates in an AnnData object.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The AnnData object containing spatial coordinates
    transformation_matrix : numpy.ndarray
        3x3 transformation matrix to apply
        
    Returns
    -------
    anndata.AnnData
        New AnnData object with transformed coordinates
    """
    # Create a deep copy of the AnnData object
    new_adata = ad.AnnData(
        X=adata.X.copy(),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
        uns=adata.uns.copy(),
        obsm={k: v.copy() for k, v in adata.obsm.items()},
        varm={k: v.copy() for k, v in adata.varm.items()} if hasattr(adata, 'varm') else None
    )

    # Transform spatial coordinates if they exist
    if 'spatial' in new_adata.obsm.keys():
        # Get coordinates
        coords = new_adata.obsm['spatial']

        # Convert to homogeneous coordinates
        homog_coords = np.hstack([coords, np.ones((coords.shape[0], 1))])
        
        # Apply transformation
        transformed_coords = np.dot(homog_coords, transformation_matrix.T)[:, :2]

        # Store transformed coordinates
        new_adata.obsm['spatial'] = transformed_coords
    else:
        print('spatial not in adata.obsm keys')
        
    return new_adata

def align_xenium_columns_to_codex(
    codex_sdata,
    xenium_sdata,
    alignment_matrices_folder,
    id_prefix,
    aligned_coordinate_system_name
):
    """
    Align multiple Xenium images to a CODEX slide image using column-specific alignment matrices.
    Apply 180-degree rotation to account for the CODEX image orientation.
    Create new tables with transformed spatial coordinates.
    
    Parameters
    ----------
    codex_sdata : SpatialData
        SpatialData object containing CODEX data
    xenium_sdata : SpatialData
        SpatialData object containing Xenium data
    alignment_matrices_folder : str or Path
        Path to folder containing alignment matrices
    id_prefix : str
        Prefix for alignment matrix files
    aligned_coordinate_system_name : str
        Name for the aligned coordinate system
    """
    # Create identity matrix and affine transformation
    identity_matrix = np.eye(3)
    identity_affine = Affine(
        identity_matrix,
        input_axes=("x", "y"),
        output_axes=("x", "y")
    )
    
    # Create 180 degree rotation matrix and affine transformation
    theta = math.pi  # 180 degrees in radians
    rotation_matrix = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])
    rotation_affine = Affine(
        rotation_matrix,
        input_axes=("x", "y"),
        output_axes=("x", "y")
    )
    
    # Create combined identity + rotation sequence
    identity_rotation_sequence = Sequence([identity_affine, rotation_affine])
    
    # Apply transformations to CODEX images
    for image_name in codex_sdata.images.keys():
        # Set global coordinate system
        set_transformation(
            codex_sdata.images[image_name], 
            identity_affine, 
            to_coordinate_system="global"
        )
        # Set aligned coordinate system with rotation
        set_transformation(
            codex_sdata.images[image_name],
            identity_rotation_sequence,
            to_coordinate_system=aligned_coordinate_system_name
        )
    
    # Store transformation matrices for each column
    transformation_matrices = {}
    
    # Process Xenium images
    for image_name in xenium_sdata.images.keys():
        # Extract column number from image name
        match = re.search(r'column_(\d+)', image_name)
        if not match:
            print(f"Skipping {image_name}: No column number found.")
            continue
        
        column_number = match.group(1)
        column_id = f"column_{column_number}"
        matrix_column_id = f"col{column_number}"
        
        # Construct path to alignment matrix
        matrix_path = os.path.join(
            alignment_matrices_folder,
            f"ID_{id_prefix}_{matrix_column_id}_iF_alignment_files/matrix.csv"
        )
        
        if not os.path.exists(matrix_path):
            print(f"Skipping {image_name}: Matrix file not found at {matrix_path}")
            continue
        
        # Load and invert transformation matrix
        codex_to_xenium_matrix = np.loadtxt(matrix_path, delimiter=',')
        xenium_to_codex_matrix = inv(codex_to_xenium_matrix)
        
        # Create affine transformation
        affine = Affine(
            xenium_to_codex_matrix,
            input_axes=("x", "y"),
            output_axes=("x", "y")
        )
        
        # Combine with rotation
        affine_rotation_sequence = Sequence([affine, rotation_affine])
        
        # Store transformation for this column
        transformation_matrices[column_id] = {
            'sequence': affine_rotation_sequence,
            'matrix': np.dot(rotation_matrix, xenium_to_codex_matrix)  # Combined matrix for AnnData
        }
        
        # Apply transformation to image
        set_transformation(
            xenium_sdata.images[image_name],
            affine_rotation_sequence,
            to_coordinate_system=aligned_coordinate_system_name
        )
        print(f"Aligned {image_name} using matrix from {matrix_path}")
    
    # Set up postponed CODEX transformation
    postpone_codex_transformation(
        sdata=codex_sdata,
        transformation=identity_rotation_sequence,
        source_coordinate_system="global",
        target_coordinate_system=aligned_coordinate_system_name,
    )
    
    # Process all non-image elements in Xenium data
    for element_type, element_name, element in xenium_sdata._gen_elements():
        if element_type == "images":
            continue
        
        # Extract column number
        match = re.search(r'column_(\d+)', element_name)
        if not match:
            print(f"Warning: Could not extract column info from {element_name}")
            # Apply only rotation for elements without column info
            set_transformation(
                element,
                identity_rotation_sequence,
                to_coordinate_system=aligned_coordinate_system_name
            )
            continue
        
        column_id = match.group(0)  # This gets "column_X"
        
        if column_id in transformation_matrices:
            # Apply transformation to the original element
            set_transformation(
                element,
                transformation_matrices[column_id]['sequence'],
                to_coordinate_system=aligned_coordinate_system_name
            )
            print(f"Aligned {element_type}/{element_name} using {column_id} transformation")
            
    # Special handling for AnnData tables
    for element_name, element in list(xenium_sdata.tables.items()):
        # Extract column number
        match = re.search(r'column_(\d+)', element_name)
        column_id = match.group(0)
        
        # Transform coordinates in the AnnData object
        transformed_table = transform_table_coordinates(
            element, 
            transformation_matrices[column_id]['matrix']
        )
        
        # Add the transformed table to the SpatialData object
        aligned_table_name = f"{element_name}_aligned"
        xenium_sdata.tables[aligned_table_name] = transformed_table
        
        print(f"Created aligned table {aligned_table_name} with transformed spatial coordinates")
        
    
    print(f"All Xenium elements aligned to '{aligned_coordinate_system_name}' coordinate system.")


