"""
Module for loading spatial data from Xenium and CODEX sources.
"""
import spatialdata as sd
import spatialdata_plot
from spatialdata_io import xenium, codex
from spatialdata.models import Image2DModel, Image3DModel
from pathlib import Path
import os
import xarray as xr
import re

XENIUM_BASE = Path('/media/Lynn/data/SpatialData/Xenium')
CODEX_BASE = Path('/media/Lynn/data/SpatialData/CODEX')
CODEX_CROPPED_BASE = Path('/media/Lynn/data/SpatialData/CODEX_cropped')
CODEX_CROPPED_UPDATED = Path('/media/Lynn/data/SpatialData/CODEX_cropped_updated')

codex_channels = [
    'DAPI', 'FoxP3', 'aSMA', 'CD4', 'CD8', 'CD31', 'CD11c', 'IFNG', 'Pan-Cytokeratin',
    'CD68', 'CD20', 'CD66b', 'TNFa', 'CD45RO', 'CD14', 'CD11b', 'Vimentin', 'CD163',
    'IL10', 'CD45', 'CCR7', 'CD38', 'CD69', 'Podoplanin', 'PNAd', 'CD16', 'CXCL13'
]

def get_xenium_slide_data(slide_id, input_path=XENIUM_BASE):
    """Load Xenium data for a specific slide ID."""
    xenium_files = list(input_path.iterdir())
    xenium_paths = [file for file in xenium_files if file.is_dir() and file.name.startswith(f'output-XETG00404__{slide_id}')]

    column_data = {}
    
    try:
        for path in xenium_paths:
            filename = path.name
            region_match = re.search(r'Region_(\d+)', filename)
            
            if region_match:
                region_num = int(region_match.group(1))
                sdata = sd.read_zarr(str(path))
                column_data[f'column_{region_num}'] = sdata
        
        return column_data
    
    except Exception as e:
        print(f"Error loading Xenium data for slide {slide_id}: {e}")
        return None

def get_codex_slide_data(slide_id, input_path = CODEX_BASE):
    """Load full CODEX slide data."""
    try:
        return sd.read_zarr(f'{input_path}/ID_{slide_id}_Scan1.er.zarr')
    except Exception as e:
        print(f"Error loading CODEX data for slide {slide_id}: {e}")
        return None

def get_codex_columns_data(slide_id, input_path=CODEX_CROPPED_BASE):
    """Load cropped CODEX data by columns for a specific slide ID."""
    codex_files = list(input_path.iterdir())
    codex_paths = [file for file in codex_files if file.is_dir() and file.name.startswith(f'ID_{slide_id}')]
    
    column_data = {}
    
    try:
        for path in codex_paths:
            filename = path.name
            column_match = re.search(r'column_(\d+)', filename)
            
            if column_match:
                column_num = int(column_match.group(1))
                sdata = sd.read_zarr(str(path))
                column_data[f'column_{column_num}'] = sdata
        
        return column_data
    
    except Exception as e:
        print(f"Error loading CODEX column data for slide {slide_id}: {e}")
        return None

def get_codex_updated_columns_data(slide_id, input_path = CODEX_CROPPED_UPDATED):
    """Load cropped CODEX data by columns for a specific slide ID."""
    codex_files = list(input_path.iterdir())
    codex_paths = [file for file in codex_files if file.is_dir() and file.name.startswith(f'updated_codex_sdata_{slide_id}')]
    
    column_data = {}
    
    try:
        for path in codex_paths:
            filename = path.name
            column_match = re.search(r'column_(\d+)', filename)
            
            if column_match:
                column_num = int(column_match.group(1))
                sdata = sd.read_zarr(str(path))
                column_data[f'column_{column_num}'] = sdata
        
        return column_data
    
    except Exception as e:
        print(f"Error loading CODEX column data for slide {slide_id}: {e}")
        return None
        

def rename_channels(sdata, output_path):
    """Reassign the correct channel names to the cropped codex images"""
    for image_name in sdata.images.keys():
        image_data = sdata.images[image_name]  # Extract image data

        # Iterate through all scales (assuming scales are named scale0, scale1, etc.)
        for scale_name in image_data.groups:
            scale_data = image_data[scale_name]

            if "c" in scale_data.dims:
                num_channels = len(scale_data.coords["c"])
                if num_channels != len(codex_channels):
                    raise ValueError(f"Mismatch in '{scale_name}' of image '{image_name}': Found {num_channels} channels, expected {len(codex_channels)}")

                # Assign new coordinates (channel names)
                scale_data.coords['c'] = codex_channels  # Assign directly to the coords dictionary
                print(scale_data.coords['c'])
                image_data[scale_name] = scale_data  # Update scale data

        # Update the image data in sdata
        sdata.images[image_name] = image_data

    # Save the updated spatial data object
    sdata.write(output_path)
    return sdata
