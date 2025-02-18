# dissMapR

[![License](https://img.shields.io/badge/license-PENDING-orange.svg)](LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/macSands/dissMapR/badge.svg?branch=main)](https://coveralls.io/github/macSands/dissMapR?branch=main)

> **A Novel Framework for Automated Compositional Dissimilarity and Biodiversity Turnover Analysis**

`dissMapR` is an R package designed to streamline the assessment, mapping, and prediction of compositional dissimilarity and biodiversity turnover. It integrates multi-site dissimilarity metrics—particularly **zeta diversity**—and advanced modeling approaches, such as Multi-Site Generalized Dissimilarity Modelling (**MS-GDM**), to evaluate how biodiversity patterns respond to environmental drivers at multiple spatial scales.

This project is part of the [B3 (B-Cubed) project](https://b-cubed.eu/), a Europe-wide collaboration dedicated to advancing robust biodiversity indicators for more effective policy and decision-making. In addition to this repository, `dissMapR` will be hosted on the [B-Cubed GitHub organization](https://github.com/b-cubed-eu).

---

## Features

- **Automated Data Integration**  
  - Retrieve species occurrence data from [GBIF](https://www.gbif.org/)  
  - Access environmental variables from sources like [WorldClim](https://worldclim.org/) and [CHELSA](https://chelsa-climate.org/)

- **Compositional Turnover Metrics**  
  - Calculate pairwise and multi-site dissimilarity (zeta diversity, Bray–Curtis, etc.)  
  - Model turnover using MS-GDM to link species composition with spatial and environmental gradients  

- **Predictive Modeling & Visualization**  
  - Project changes in community composition or “bioregions” under different environmental scenarios  
  - Generate interactive or static maps of turnover “hotspots” and emergent bioregions  

---

## Why Use `dissMapR`?

1. **Reproducible Workflows**  
   Automates data retrieval, cleaning, and modeling to reduce error and save time.

2. **Beyond Alpha and Beta Diversity**  
   Emphasizes **multi-site connectivity** (zeta diversity), capturing compositional patterns often overlooked by simpler metrics.

3. **Scalable and Flexible**  
   Adaptable to diverse datasets and spatial extents, from local to continental scales.

4. **Decision-Support Ready**  
   Maps of predicted turnover and delineated bioregions can inform conservation priorities, climate adaptation plans, and biodiversity policies.

---

## Example Application

Below is a brief overview of how `dissMapR` can be applied to butterfly occurrence data in southern Africa:

1. **Data Access**  
   - Automated retrieval of occurrence records via GBIF  
   - Environmental data (e.g., temperature, precipitation) from WorldClim  

2. **Multi-Site Metrics**  
   - Computation of zeta diversity to assess the proportion of shared species across multiple sites  
   - Calculation of Bray–Curtis dissimilarity and others for broader context  

3. **MS-GDM**  
   - Linking environmental gradients (e.g., climate variables) with biodiversity turnover  
   - Predicting compositional changes under future environmental scenarios  

4. **Visualization**  
   - Mapping of high-turnover “hotspots” and distinct bioregional clusters  
   - Identifying potential zones of rapid community shift under climate change  

---

## Installation

> **Note**: `dissMapR` is under active development and will be released via the [B-Cubed GitHub organization](https://github.com/b-cubed-eu). Installation from source will follow once the repository is public.

```r
# Coming soon:
# install.packages("devtools")
# devtools::install_github("b-cubed-eu/dissMapR")
