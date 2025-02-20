# dissMapR Tutorial

This document outlines a step-by-step workflow for analyzing compositional dissimilarity and bioregionalization using `dissMapR`.

---

## 1. User-Defined Area of Interest and Grid Resolution

1. **Define the AOI**: Use a shapefile (e.g., `rsa.shp`) or a bounding box to specify the region of interest.  
2. **Grid Resolution**: Choose a grid size (e.g., 0.5°).  
3. **Generate Spatial Grids**:  
   - Create a site centroid data frame `xy`, with each row representing a grid cell’s centroid.  
   - Produce a map visualizing the Area of Interest (AOI), overlaid with the 0.5° grid.  
4. **Outcome**:  
   - A **map** of the AOI with grid cells  
   - A **data frame** `xy` containing centroid coordinates for each site (grid cell)

---

## 2. Site-by-Species Matrix and Sampling Effort

1. **Occurrence Data**:  
   - Retrieve occurrence records for a user-specified taxon or list of species (e.g., via GBIF).  
   - Assign each record to the corresponding grid cell based on its coordinates.  
2. **Sampling Effort**:  
   - Sum the number of occurrences in each grid cell.  
   - Create a **raster** of occurrence counts (referred to as “sampling effort”).  
3. **Site-by-Species Matrix**:  
   - Generate a **binary** data frame `sbs`, where 1 = presence and 0 = absence for each site-species combination.  
   - Optionally calculate the row sums (species richness) to produce a **richness map**.  
   - Extract the single variable of sampling effort into a data frame `sam.eff`, mapped as a **raster**.  
4. **Outcome**:  
   - **sbs** (binary presence–absence data)  
   - **sam.eff** (numerical sampling effort for each site)  
   - **Rasters** for occurrence count (sampling effort) and species richness  
   - **Coordinates** are **not** needed beyond grid assignment

---

## 3. Site-by-Environment Matrix

1. **Extract Environmental Variables**:  
   - Sample climate/environmental rasters (e.g., temperature, precipitation) at each site centroid in `xy`.  
2. **Build `sbe`**:  
   - Combine extracted environmental variables into a matrix or data frame (`sbe`).  
   - Append the sampling effort column (`sam.eff`) to both the raster stack and `sbe`.  
3. **Outcome**:  
   - **sbe** (site-by-environment matrix) with additional `sam.eff`  
   - Updated raster stack that includes environmental layers + sampling effort

---

## 4. Zeta Decline and Zeta Decay

1. **Zeta Decline**:  
   - Use the **site-by-species matrix** (`sbs`) to evaluate how the proportion of shared species changes with higher-order site combinations (orders 2 to 15).  
   - Produce **statistics** (e.g., zeta values for each order) and any relevant **figures** (e.g., a plot of zeta decline vs. order).  
2. **Zeta Decay**:  
   - Combine `sbs` and **site coordinates** `xy` to assess how zeta diversity changes with geographic distance.  
   - Generate summary **statistics** and **figures** (scatter or curve plots), no spatial map.  
3. **Outcome**:  
   - Key metrics describing zeta decline (multi-site turnover)  
   - Figures illustrating multi-site dissimilarity vs. order and/or distance

---

## 5. MS-GDM with `Zeta.msgdm(sbs, sbe, xy)`

1. **Modeling**:  
   - Fit multi-site Generalized Dissimilarity Models (MS-GDM) using `Zeta.msgdm` for orders 2, 3, 5, and 10.  
   - Inputs: `sbs` (species data), `sbe` (environment and sampling effort), and `xy` (coordinates).  
2. **Results**:  
   - Model summaries and **statistics** for each order.  
   - **Figures** describing model fit or partial response curves if needed (no mapping at this stage).  
   - **Save** the fitted MS-GDM model for order 2 as `zeta2`.  
3. **Outcome**:  
   - **zeta2** (fitted order 2 model)  
   - Plots or tables summarizing the other fitted orders

---

## 6. Prediction with `zeta2` (Present Scenario)

1. **Update `sam.eff`**:  
   - Replace sampling effort in `sbe` with a constant maximum value (`sam.max` = `max(sam.eff)`), simulating uniform sampling.  
2. **Predict**:  
   - Use `predict(zeta2, newdata = sbe_updated)` to produce a **site-by-site** matrix of predicted zeta diversity (`zeta.now`).  
3. **Visualize**:  
   - **NMDS** on `zeta.now` to reduce dimensionality.  
   - Plot an **RGB** composite from the top 3 NMDS axes, generating a **dissimilarity map**.  
   - **Clustering** on `zeta.now` to define “bioregions.”  
   - Produce maps of the **RGB** plot (gradient in compositional dissimilarity) and a **cluster map** (bioregions).  
4. **Outcome**:  
   - **zeta.now** (site-by-site predicted dissimilarity matrix)  
   - NMDS-based dissimilarity map (RGB)  
   - Bioregional clusters and their spatial visualization

---

## 7. Prediction with `zeta2` (Future Scenarios)

1. **Append Future Environmental Variables**:  
   - Add or replace environment columns in `sbe` with corresponding future scenario data.  
   - Maintain `sam.max` for sampling effort.  
   - For *m* scenarios + present scenario, the new data frame (`sbe.future`) has `(m+1)*n` rows, merging all sites and scenarios.  
2. **Predict**:  
   - Use `predict(zeta2, newdata = sbe.future)` to produce a `(k x k)` predicted zeta matrix (`zeta.future`), where `k = (m+1)*n`.  
   - Separate or subset the results to map each scenario individually.  
3. **Visualize**:  
   - Perform NMDS or clustering on `zeta.future` to examine temporal changes in dissimilarity.  
   - Map each scenario’s **sub-matrix** or produce an integrated “future shift” map illustrating predicted changes.  
4. **Outcome**:  
   - **zeta.future** (predicted site-by-site dissimilarities for all scenarios + present)  
   - NMDS or cluster results across scenarios, showing potential **bioregion shifts** or changes in dissimilarity

> **Note**: Step 6 is effectively included within step 7 if you incorporate the present scenario as one of the `(m+1)` scenarios. However, be aware of the computational load when dealing with multiple future layers.

---

## 8. Data Publication to Zenodo

1. **Compile Outputs**:  
   - Data frames (`sbs`, `xy`, `sbe`, `zeta.now`, `zeta.future`, etc.)  
   - All raster/maps (sampling effort, zeta decline figures, NMDS RGB maps)  
   - Model summaries or any relevant metadata (e.g., scripts, parameter settings)  
2. **Prepare a Zenodo Archive**:  
   - Provide standard metadata: title, authors, date, version, digital object identifier (DOI).  
   - Include references and a brief readme describing each dataset and figure.  
3. **Publish**:  
   - Deposit all relevant outputs for reproducibility and future reference.

---

### Final Remarks

- This workflow demonstrates the **core** steps for dissecting compositional turnover and bioregional patterns under present or future scenarios using `dissMapR`.  
- Each stage can be customized to meet specific research or management needs (e.g., selecting different environment layers or adjusting the sampling resolution).  
- Always confirm the validity of spatial and environmental data sources, particularly when scaling up to large regions or high-resolution scenarios.
