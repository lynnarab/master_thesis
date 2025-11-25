import pandas as pd
import numpy as np
import anndata
import xarray as xr
from typing import Dict, Any, Union
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def count_xenium_unassigned_transcripts(sdata) -> int:
    """
    Count the total number of transcripts not assigned to any cell in Xenium spatial data.
    
    Parameters
    ----------
    sdata : xr.Dataset
        Xenium spatial data containing transcript information with cell assignments.
        Expected to have a 'points' attribute containing transcript coordinates.
        
    Returns
    -------
    int
        Total number of transcripts labeled as "UNASSIGNED".
        
    Notes
    -----
    This function iterates through all xenium columns in the spatial data
    and counts transcripts that are not assigned to any cell (labeled as "UNASSIGNED").
    """
    unassigned_transcripts = 0
    
    for column_i_transcripts in sdata.points.keys():
        total_transcripts = sdata[column_i_transcripts].compute()
        if "cell_id" not in total_transcripts:
            logger.warning(f"Column {column_i_transcripts} does not contain 'cell_id' field")
            continue
        unassigned_transcripts += (total_transcripts["cell_id"] == "UNASSIGNED").sum()
    
    return unassigned_transcripts


def compute_xenium_metrics(adata, sdata) -> Dict[str, Any]:
    """
    Compute quality control metrics for Xenium spatial transcriptomics data.
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing cell-gene expression data.
        Expected to have 'n_genes_by_counts' and 'transcript_counts' in obs.
    sdata : xr.Dataset
        Xenium spatial data containing transcript information with cell assignments.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing various QC metrics:
        - total_cells: Number of cells detected
        - average_unique_genes_per_cell: Average number of unique genes expressed per cell
        - total_transcripts: Total number of transcripts in the dataset
        - percent_transcripts_in_cells: Percentage of transcripts inside cells
        - percent_transcripts_outside_of_cells: Percentage of transcripts outside cells
    """
    summary = {}
    
    # Check for required columns
    required_cols = ['n_genes_by_counts', 'transcript_counts']
    for col in required_cols:
        if col not in adata.obs:
            raise ValueError(f"Required column '{col}' not found in adata.obs")
    
    # Total number of cells
    summary['total_cells'] = adata.n_obs
    
    # Average genes per cell 
    summary['average_unique_genes_per_cell'] = adata.obs['n_genes_by_counts'].mean()
    
    # Transcript localization (inside vs outside cell)
    outside_transcripts = count_xenium_unassigned_transcripts(sdata)
    
    # Total number of transcripts 
    summary['total_transcripts'] = adata.obs['transcript_counts'].sum()

    inside_transcripts = summary['total_transcripts'] - outside_transcripts
    
    # Avoid division by zero
    if summary['total_transcripts'] > 0:
        summary['percent_transcripts_in_cells'] = 100 * inside_transcripts / summary['total_transcripts']
        summary['percent_transcripts_outside_of_cells'] = 100 * outside_transcripts / summary['total_transcripts']
    else:
        summary['percent_transcripts_in_cells'] = 0
        summary['percent_transcripts_outside_of_cells'] = 0
    
    return summary

def compute_codex_channel_means(image_data, 
                              downsample_factor: int = 10) -> Dict[str, float]:
    """
    Compute the mean intensity for each channel in CODEX imaging data.
    
    Parameters
    ----------
    image_data : xr.DataArray
        CODEX image data with dimensions including 'c' for channels.
    downsample_factor : int, default=10
        Factor by which to downsample the image to reduce memory usage.
        Higher values result in faster computation but less precise means.
        
    Returns
    -------
    Dict[str, float]
        Dictionary mapping channel names to their mean intensity values.
        
    Notes
    -----
    This function calculates mean intensity values for each channel in CODEX data,
    using downsampling to manage memory usage when processing large images.
    """
    means = {}
        
    if 'c' not in image_data.coords:
        raise ValueError("Input image_data missing 'c' coordinate for channels")
    
    for c in image_data.coords['c'].values:
        # Sample pixels to avoid loading entire slide
        channel_data = image_data.sel(c=c)
        img = channel_data.data[::downsample_factor, ::downsample_factor]
        means[str(c)] = float(img.mean().compute().item())
        logger.info(f"Processed channel {c}, mean intensity: {means[str(c)]:.4f}")
        
    return means

def export_metrics_to_csv(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Export computed metrics to a CSV file.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary containing metrics to export.
    output_path : str
        Path where the CSV file will be saved.
        
    Returns
    -------
    None
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Metrics exported to {output_path}")

