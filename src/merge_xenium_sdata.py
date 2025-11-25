import numpy as np
import pandas as pd
from spatialdata import SpatialData
from anndata import AnnData
import geopandas as gpd
from typing import Dict
import xarray as xr
import anndata as ad
import spatialdata as sd
from copy import deepcopy


def combine_xenium_columns(sdata_dict):
    """
    Combine multiple SpatialData objects from a dictionary into a single SpatialData object
    without translating or repositioning any elements.
    
    Parameters
    ----------
    sdata_dict : dict
        Dictionary where keys are column identifiers (e.g., 'column_1', 'column_2')
        and values are SpatialData objects.
        
    Returns
    -------
    SpatialData
        A combined SpatialData object containing data from all regions.
    """
    if not sdata_dict:
        raise ValueError("Empty dictionary provided. Need at least one SpatialData object.")
    
    # Initialize containers for combined data
    combined_images = {}
    combined_labels = {}
    combined_points = {}
    combined_shapes = {}
    combined_tables = {}
    
    # Process each column's SpatialData object
    for column_id, sdata in sdata_dict.items():
        # Use the original column identifier as prefix
        prefix = column_id
        
        # Add Images
        for image_key, image in sdata.images.items():
            new_key = f"{prefix}_{image_key}"
            combined_images[new_key] = deepcopy(image)
        
        # Add Labels
        for label_key, label in sdata.labels.items():
            new_key = f"{prefix}_{label_key}"
            combined_labels[new_key] = deepcopy(label)
        
        # Add Points (no coordinate adjustments)
        for points_key, points_df in sdata.points.items():
            new_key = f"{prefix}_{points_key}"
            combined_points[new_key] = points_df.copy()
        
        # Add Shapes (no geometry adjustments)
        for shapes_key, shapes_gdf in sdata.shapes.items():
            new_key = f"{prefix}_{shapes_key}"
            combined_shapes[new_key] = shapes_gdf.copy()
        
        # Add Tables
        for table_key, anndata in sdata.tables.items():
            new_key = f"{prefix}_{table_key}"
            adata_copy = anndata.copy()
            
            # Add a column to identify the column/region
            adata_copy.obs['column'] = column_id
            
            combined_tables[new_key] = adata_copy
    
    # Create a new SpatialData object with combined data
    combined_sdata = sd.SpatialData(
        images=combined_images if combined_images else None,
        labels=combined_labels if combined_labels else None,
        points=combined_points if combined_points else None,
        shapes=combined_shapes if combined_shapes else None,
        tables=combined_tables if combined_tables else None
    )
    
    return combined_sdata

def concatenate_tables(sdata, label='column'):
    """
    Concatenate all AnnData tables in a SpatialData object into one.

    Parameters:
    - sdata: SpatialData object containing Tables
    - label (str): Column name to store original table name (default: 'column')

    Returns:
    - AnnData: Concatenated AnnData table with added label column
    """
    # Filter only the table keys
    table_keys = list(sdata.tables.keys())
    
    # Get all AnnData tables
    tables = [sdata.tables[k] for k in table_keys]
    
    # Use anndata.concat with keys for tracking origin
    merged = ad.concat(tables, join='outer', label=label, keys=table_keys)
    
    return merged


def merge_xenium_slides_tables(adata1, adata2, slide1_id="0022110", slide2_id="0022111"):
    """
    Merge two AnnData objects from Xenium experiments with slide identification.
    
    Parameters:
    -----------
    adata1 : AnnData
        First AnnData object from Xenium
    adata2 : AnnData
        Second AnnData object from Xenium
    slide1_id : str, default="slide1"
        Identifier for the first slide
    slide2_id : str, default="slide2"
        Identifier for the second slide
        
    Returns:
    --------
    AnnData
        Merged AnnData object with slide identifiers
    """
    # Make copies to avoid modifying the originals
    adata1_copy = adata1.copy()
    adata2_copy = adata2.copy()
    
    # Add slide ID to obs dataframe
    adata1_copy.obs['slide_id'] = slide1_id
    adata2_copy.obs['slide_id'] = slide2_id
    
    # Concatenate the AnnData objects
    merged_adata = ad.concat(
        [adata1_copy, adata2_copy],
        join='outer',  # Use outer join to keep all genes from both slides
        label='slide_id',  # Use the slide_id as the batch key
        keys=[slide1_id, slide2_id]  # Keys for the batches
    )
    
    return merged_adata