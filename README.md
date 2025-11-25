# Master's Thesis

**Integration of Multi-Omics Single-Cell Spatial Data for the Discovery of Biomarkers**

This repository contains the full codebase, notebooks, and environment files developed for my Master’s thesis. The goal of the project is to integrate multiple spatial omics technologies—primarily **10x Xenium** and **CODEX multiplexed immunofluorescence**—to explore cellular interactions and molecular signatures in Crohn's disease ileum and colon patient biopsies.

## Thesis Overview

Single-cell spatial omics platforms provide high-resolution maps of gene and protein expression directly within tissue context. However, integrating these complementary modalities poses computational and methodological challenges.

My thesis focuses on:

* **Preprocessing and quality control** of Xenium and CODEX datasets
* **Spatial alignment and cell segmentation** of protein-level and transcript-level data
* **Integration, clustering and annotation** of multi-omics data 
* **Identification of spatially resolved biomarkers** and cell–cell interaction signatures

The integration framework aims to uncover robust biomarkers by leveraging the strengths of each platform: rich protein panels from CODEX and high-throughput gene expression from Xenium.

## Repository Structure

### **notebooks/**

Jupyter notebooks for exploratory data analysis, QC, visualizations, and integration workflows.

### **src/**

Python source code containing reusable functions.

### **yaml_envs/**

Environment specifications (micromamba) used for reproducible computational environments.

## Technologies & Tools

* **Spatial omics:** 10x Xenium, CODEX
* **Analysis libraries:** `spatialdata`, `scvi-tools`, `scanpy`, `squidpy`, `napari`, `sopa`
* **Segmentation:** Cellpose, Xenium Explorer
* **Alignment:** custom transformation matrix pipeline
