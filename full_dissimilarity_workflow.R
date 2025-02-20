# dissMapR Tutorial
# This document outlines a step-by-step workflow for analyzing compositional dissimilarity and bioregionalization using **dissMapR**.
# All code chunks below illustrate how one might implement each step in R, using various commonly used packages (e.g., `sf`, `terra`, `dplyr`, `ggplot2`, etc.).
# Please note that paths to data, package names, or exact functions may need adjustment depending on your local setup.


# //////////////////////////////////////////////////////////////////////////////
# 1. User-Defined Area of Interest and Grid Resolution
# //////////////////////////////////////////////////////////////////////////////

# ---- 1a_Libraries ----
# Load necessary libraries
library(sf)      # for vector spatial data
library(terra)   # for raster/grid operations
library(dplyr)   # for data manipulation
library(ggplot2) # for plotting
library(zetadiv)

# ---- 1b_Define_AOI ----
# 1. Define the AOI (e.g., using a shapefile 'rsa.shp')
# Adjust the path and layer name as needed
provinces <- st_read("D:/Data/South_Africa/southafrica_provinces_lesotho_swaziland.shp")
aoi <- st_read("D:/Data/South_Africa/southafrica_lesotho_swaziland_dissolved.shp")
plot(aoi)

# 2. Grid Resolution (e.g., 0.5 degrees)
grid_res <- 0.5

# # ---- 1c_Generate_Grid ----
# # Create a blank raster template with the desired resolution covering the AOI
# # We'll use terra::rast() and the extent (bounding box) from 'aoi'
# r_template <- rast(ext(aoi), resolution = grid_res, crs = st_crs(aoi)$wkt)
#
# # Convert raster cell centroids to a data frame (x = longitude, y = latitude)
# xy <- as.data.frame(crds(r_template))
# colnames(xy) <- c("x", "y")
#
# # Optional: clip out grid cells that fall outside the AOI boundary
# # Convert xy -> sf -> spatial filter
# xy_sf <- st_as_sf(xy, coords = c("x", "y"), crs = st_crs(aoi), remove = FALSE)
# xy_aoi <- xy_sf[aoi, ]  # intersection
# xy <- st_drop_geometry(xy_aoi)

# Suppose 'xy' is a data frame with columns "x" and "y", in the same CRS as 'aoi'.
# and 'aoi' is an sf polygon.

# ---- 1c_Generate_Grid ----
# 1. Determine bounding box from AOI, add a small buffer, snap to grid_res
bb <- st_bbox(aoi)

# We'll buffer by exactly 1 grid cell on each side. Increase if needed.
bb["xmin"] <- floor(bb["xmin"] / grid_res) * grid_res - grid_res
bb["ymin"] <- floor(bb["ymin"] / grid_res) * grid_res - grid_res
bb["xmax"] <- ceiling(bb["xmax"] / grid_res) * grid_res + grid_res
bb["ymax"] <- ceiling(bb["ymax"] / grid_res) * grid_res + grid_res

# 2. Create an sf polygon grid over this bounding box
grid_polygons <- st_make_grid(
  st_as_sfc(bb),
  cellsize = grid_res,
  what = "polygons",
  square = TRUE
)
grid_sf <- st_sf(geometry = grid_polygons, crs = st_crs(aoi))

# 3. Calculate centroids for each grid cell
centroids <- st_centroid(grid_sf)
coords <- st_coordinates(centroids)
grid_sf$centroid_x <- coords[, 1]
grid_sf$centroid_y <- coords[, 2]

# 4. Optionally assign a “mapsheet” code
#    (commonly used if you have a geographic CRS in degrees)
#    A simple format might look like: E018N33, W001S12, etc.
lon_int <- floor(grid_sf$centroid_x)
lat_int <- floor(grid_sf$centroid_y)
lon_dir <- ifelse(grid_sf$centroid_x >= 0, "E", "W")
lat_dir <- ifelse(grid_sf$centroid_y >= 0, "N", "S")
grid_sf$mapsheet <- sprintf("%s%03d%s%02d", lon_dir, abs(lon_int), lat_dir, abs(lat_int))

# 5. Add a unique grid ID
grid_sf$grid_id <- as.character(seq_len(nrow(grid_sf)))

# 6. (Optional) Convert to a terra raster template if you need a raster base
r_template <- rast(
  ext(grid_sf),
  resolution = grid_res,
  crs = st_crs(aoi)$wkt
)
# Assign cell IDs
r_template[] <- seq_len(ncell(r_template))

# 7. Create a data frame of centroid coordinates
xy <- data.frame(
  grid_id = grid_sf$grid_id,
  x       = grid_sf$centroid_x,
  y       = grid_sf$centroid_y,
  mapsheet = grid_sf$mapsheet
)
dim(xy)
str(xy)

# 8. Quick plot of the AOI and centroids
ggplot() +
  geom_point(data = xy, aes(x = x, y = y), color = "blue", size = 1) +
  geom_sf(data = grid_sf, fill = NA, color = "gray70") +
  geom_sf(data = aoi, fill = NA, color = "black") +
  labs(title = "Area of Interest with Buffered Grid",
       subtitle = paste("Grid size:", grid_res, "degrees")) +
  theme_minimal()

# # Optional: clip out grid cells that fall outside the AOI boundary
# # Convert xy -> sf -> spatial filter
# xy_sf <- st_as_sf(xy, coords = c("x", "y"), crs = st_crs(aoi), remove = FALSE)
# xy_aoi <- xy_sf[aoi, ]  # intersection
# xy_clip <- st_drop_geometry(xy_aoi)
# dim(xy_clip)
# str(xy_clip)

# Convert xy to sf
xy_sf <- st_as_sf(xy, coords = c("x", "y"), crs = st_crs(aoi), remove = FALSE)

# 1. Buffer the polygon by grid_res
#    If your AOI is in degrees (EPSG:4326), this buffer is also in degrees.
#    If you want a meter buffer, you should project to a suitable CRS first.
# aoi_simple = st_simplify(aoi['PROV_CODE'], dTolerance = 5e3) # 5000m
# aoi_buffered <- st_buffer(aoi, dist = grid_res*2)

# # 1) Choose a projected CRS that covers South Africa (e.g., Africa Albers Equal Area, EPSG:9822)
# aoi_projected <- st_transform(aoi, crs=9822)
#
# # 2) Buffer by 50 km, for instance
# aoi_buffered_proj <- st_buffer(aoi_projected, dist = 50000)  # 50,000 meters
#
# # 3) Transform back to lat/lon if you need it in EPSG:4326
# aoi_buffered <- st_transform(aoi_buffered_proj, 4326)
#
# # 4) Compare visually
# plot(st_geometry(aoi_buffered), col = "lightblue")
# plot(st_geometry(aoi), add = TRUE, border = "red")

# # 1. Buffer the polygon by grid_res
# aoi_simple = st_simplify(aoi['NAME_01'], dTolerance = 100) # 5000m
# plot(aoi_simple)
# # aoi_buffered <- st_buffer(aoi, dist = grid_res*2)
# aoi_buffered <- st_buffer(aoi_simple, dist = 5)
# plot(st_geometry(aoi_buffered), col = "lightblue")
# plot(st_geometry(aoi), add = TRUE, border = "red")
#
#
# # 2. Clip points by the buffered polygon
# xy_aoi_sf <- xy_sf[aoi_buffered, ]  # intersection: only points within or on boundary
#
# # 3. Drop geometry if you only need a data frame
# xy_clip <- st_drop_geometry(xy_aoi_sf)
# head(xy_clip)
# # Quick check
# dim(xy_clip)   # should have more points than a strict, no-buffer intersection
#
# ggplot() +
#   geom_sf(data = aoi, fill = "lightgray", color = "black") +
#   geom_point(data = xy_clip, aes(x = x, y = y), color = "blue", size = 1) +
#   ggtitle("Area of Interest with 0.5° Grid") +
#   theme_minimal()

# ---- 1d_Map_AOI_Grid ----
# Quick visualization: AOI + centroid points
ggplot() +
  geom_sf(data = aoi, fill = "lightgray", color = "black") +
  geom_point(data = xy, aes(x = x, y = y), color = "blue", size = 1) +
  ggtitle("Area of Interest with 0.5° Grid") +
  theme_minimal()

# Outcome:
# A map of the AOI with 0.5° grid cells
# A data frame xy containing centroid coordinates for each site (grid cell)

# //////////////////////////////////////////////////////////////////////////////
# 2. Site-by-Species Matrix and Sampling Effort
# //////////////////////////////////////////////////////////////////////////////

# ---- 2a_Occurrence_Data ----
# Suppose you have occurrence data with fields: species, longitude, latitude
# (e.g., from GBIF, local CSV, etc.). We'll simulate for demonstration.

# occ_data <- data.frame(
#   species = sample(c("SpeciesA", "SpeciesB", "SpeciesC"), 1000, replace = TRUE),
#   decimalLongitude = runif(1000, min(st_bbox(aoi)["xmin"]), max(st_bbox(aoi)["xmax"])),
#   decimalLatitude  = runif(1000, min(st_bbox(aoi)["ymin"]), max(st_bbox(aoi)["ymax"]))
# )

occ_data = get_occurrence_data(data = 'D:/Courses_Workshops/Dissimilarity/rData/0006880-241024112534372.csv',
                               source_type = 'local_csv', # options: 'local_csv', 'data_frame', 'gbif'),
                               sep = '\t', # Default: ','
                               gbif_zip_url = NULL, # e.g. 'https://api.gbif.org/v1/occurrence/download/request/0038969-240906103802322.zip'
                               download_dir = tempdir())
head(occ_data)

# Assign each record to a grid cell based on coordinates:
# Convert 'xy' to a raster to facilitate cell lookups
r_id <- r_template
values(r_id) <- 1:ncell(r_id)  # each cell gets a unique ID

# Convert occurrence data to SpatVector (terra) for cell lookups
# occ_vect <- vect(occ_data, geom = c("decimalLongitude", "decimalLatitude"), crs = crs(r_template))
# occ_vect <- vect(occ_data, geom = c("x", "y"), crs = crs(r_template))
#
# # Identify cell ID for each occurrence
# cell_ids <- cellFromXY(r_id, geom(occ_vect)[, c("x", "y")])
# occ_data$cell_id <- cell_ids

# 1. Assign each occurrence to a cell
cell_ids <- cellFromXY(r_id, geom(occ_vect)[, c("x", "y")])
occ_data$cell_id <- cell_ids

# 2. Extract the center (x,y) of each cell in r_id
centers_df <- data.frame(
  cell_id = 1:ncell(r_id),
  xyFromCell(r_id, 1:ncell(r_id))  # returns a matrix of [x, y] cell centers
)
colnames(centers_df)[2:3] <- c("x_center", "y_center")  # rename for clarity

# 3. Join cell-center coordinates back into occ_data
occ_data <- left_join(occ_data, centers_df, by = "cell_id")

# Now occ_data has the columns: cell_id, x_center, y_center for each record
head(occ_data)

# Filter out occurrences that fall outside the defined AOI grid
occ_data <- subset(occ_data, !is.na(cell_id))
head(occ_data)

# ---- 2b_Site coordinates ----
# Count number of occurrences per cell as a measure of sampling effort
sites_xy <- occ_data %>%
  group_by(cell_id,x_center,y_center) %>%
  summarise(effort = n(),
            richness = n_distinct(verbatimScientificName))
names(sites_xy) = c("cell_id","x","y","effort","richness")
dim(sites_xy)
head(sites_xy)


# ---- 2c_Sampling_Effort ----
# Count number of occurrences per cell as a measure of sampling effort
effort_df <- occ_data %>%
  group_by(cell_id) %>%
  summarise(n_occurrences = n())
dim(effort_df)
head(effort_df)
min(effort_df$n_occurrences);max(effort_df$n_occurrences)

# Rasterize sampling effort (optional for mapping)
sam_eff_rast <- r_id
values(sam_eff_rast) <- 0
# values(sam_eff_rast)[effort_df$cell_id] <- effort_df$n_occurrences
values(sam_eff_rast) <- effort_df$n_occurrences

# ---- 2c_Site_by_Species_Matrix ----
# For presence-absence, pivot occurrence data to wide format:
# sbs_long <- occ_data %>%
#   distinct(cell_id, species) %>%
#   mutate(presence = 1)
sbs_long <- occ_data %>%
  group_by(cell_id, verbatimScientificName) %>%
  summarise(presence = sum(pa, na.rm = TRUE), .groups = "drop")
head(sbs_long)
dim(sbs_long)
View(sbs_long)

# Ensure we have a row for every cell (site) in 'xy'
all_cells <- data.frame(cell_id = 1:ncell(r_id))
# sbs_wide <- all_cells %>%
#   left_join(sbs_long, by = "cell_id") %>%
#   pivot_wider(id_cols = cell_id, names_from = species, values_from = presence, values_fill = 0)

# 1. Filter out empty or NA species names from sbs_long
sbs_long_clean <- sbs_long %>%
  filter(!is.na(verbatimScientificName) & verbatimScientificName != "")
dim(sbs_long_clean)

# 2. Proceed with the join and pivot
sbs_wide <- all_cells %>%
  left_join(sbs_long_clean, by = "cell_id") %>%
  # 1) Remove any rows with NA species, which would create an NA column
  filter(!is.na(verbatimScientificName) & verbatimScientificName != "") %>%
  # 2) Pivot while filling non-matching species with 0
  pivot_wider(
    id_cols = cell_id,
    names_from = verbatimScientificName,
    values_from = presence,
    values_fill = 0
  )

head(sbs_wide)
dim(sbs_wide)

# sbs - site-by-species matrix
# sbs <- as.data.frame(sbs_wide)
sbs <- as.data.frame(sbs_wide[ , -1])  # remove cell_id column
# sbs <- as.data.frame(sbs_wide[ , -2])  # remove NA column
row.names(sbs) <- sbs_wide$cell_id
head(sbs)
dim(sbs)
View(sbs)

# Summarize sampling effort in a vector aligned with sbs rows
dim(effort_df)
sam.eff <- numeric(nrow(sbs))
# sam.eff[effort_df$cell_id] <- effort_df$n_occurrences
sam.eff <- effort_df$n_occurrences
str(sam.eff)

# ---- 2d_Optional_Maps ----
# Species richness map (row sums of sbs)
# richness <- rowSums(sbs[,-1])
# richness <- rowSums(sbs)
# Only count columns with values > 0 for each row
richness <- rowSums(sbs > 0)
min(richness);max(richness)
richness_rast <- r_id
values(richness_rast) <- 0
values(richness_rast)[as.numeric(names(richness))] <- richness
richness_rast

# Quick plot of sampling effort + species richness
plot(sam_eff_rast, main = "Sampling Effort")
plot(richness_rast, main = "Species Richness")

# Mask the areas outside of RSA
# 1. Convert your sf polygon (aoi) into a SpatVector
aoi_vect <- vect(aoi)
plot(aoi_vect)

# aoi_buffered <- vect(st_buffer(aoi, dist = grid_res*2))
# plot(aoi_buffered)

# 2. Crop the raster to the bounding extent of aoi (optional but efficient)
sam_eff_cropped <- crop(sam_eff_rast, aoi_vect, touches=TRUE)
plot(sam_eff_cropped)

# 3. Mask the cropped raster by aoi so cells outside the polygon become NA
sam_eff_masked <- mask(sam_eff_cropped, aoi_vect)

# 4. Repeat for richness_rast
richness_cropped <- crop(richness_rast, aoi_vect)
richness_masked  <- mask(richness_cropped, aoi_vect)

# 5. Plot the masked results
plot(sam_eff_masked, main = "Sampling Effort (Masked)")
plot(richness_masked, main = "Species Richness (Masked)")

# Outcome:
# sbs: binary presence–absence (site-by-species)
# sam.eff: numeric sampling effort per site
# Optional raster maps for sampling effort and species richness

# //////////////////////////////////////////////////////////////////////////////
# 3. Site-by-Environment Matrix
# //////////////////////////////////////////////////////////////////////////////

# ---- 3a_Environmental_Data ----
# Assume we have environmental rasters (temp, precip, etc.) with same CRS and extent
# Here we just simulate them:
# env1 <- r_template; values(env1) <- runif(ncell(r_template), min = 10, max = 30)  # e.g., temperature
# env2 <- r_template; values(env2) <- runif(ncell(r_template), min = 100, max = 3000) # e.g., precipitation
#
# # Combine into a SpatRaster
# env_stack <- c(env1, env2)
# names(env_stack) <- c("temp", "precip")

output_path = 'D:/Courses_Workshops/Dissimilarity/rData'
head(sites_xy)
enviro_list1 = get_enviro_data(
  data = sites_xy,
  buffer_km = 10,
  source = 'geodata',
  var = "bio",  # Specify WorldClim variable
  res = 2.5,
  path = output_path
)
str(enviro_list1, max.level = 1)
enviro_list1$env_rast
names(enviro_list1$env_rast) = paste0('bio_',names(enviro_list1$env_rast))

# ---- 3b_Extract_Env_Values ----
# Extract environmental values at each site centroid
# env_vals <- terra::extract(env_stack, cbind(xy$x, xy$y))
head(sites_xy)
env_vals <- terra::extract(enviro_list1$env_rast, sites_xy[,2:3])
# env_vals <- env_vals[ , -1]  # remove ID column returned by extract()
head(env_vals)

# 2. Combine (cbind) the columns you want to keep (cols 1, 45) with the extracted values
env_vals <- cbind(sites_xy[, c(1,4,5)], env_vals)

# ---- 3c_Build_sbe ----
# Combine environment variables + sampling effort into one data frame
str(sam.eff)
head(env_vals)
sbe <- cbind(env_vals[,-4], sam.eff)  # site-by-environment, includes sampling effort
# colnames(sbe) <- c("site_id","temp", "precip", "sam.eff")
head(sbe)

# We'll also update the env_stack to include sampling effort if desired
sam_eff_raster <- sam_eff_rast
names(sam_eff_raster) <- "sam.eff"

rich_raster <- richness_rast
names(rich_raster) <- "richness"

plot(enviro_list1$env_rast$bio_1)
env_rast_crop = crop(enviro_list1$env_rast, aoi_vect)
env_rast_crop  <- mask(env_rast_crop, aoi_vect)
plot(env_rast_crop$bio_1)
plot(sam_eff_masked)

plot(sam_eff_masked)
plot(richness_masked)

richness_resample = resample(richness_masked, env_rast_crop$bio_1)
names(richness_resample) = 'richness'
effort_resample = resample(sam_eff_masked, env_rast_crop$bio_1)
names(effort_resample) = 'effort'

env_stack_all <- c(effort_resample, richness_resample, env_rast_crop)
names(env_stack_all)
plot(env_stack_all)

# Outcome:
# sbe: site-by-environment matrix (including sampling effort)
# Updated raster stack (env_stack_all) with environment layers + sampling effort

# //////////////////////////////////////////////////////////////////////////////
# 4. Zeta Decline and Zeta Decay
# //////////////////////////////////////////////////////////////////////////////

# ---- 4a_Zeta_Decline ----
# We'll assume dissMapR or zetadiv provides functions for multi-site zeta.
# For example, using the 'zetadiv' package:
library(zetadiv)

# sbs must be a site-by-species matrix (rows=sites, columns=species)
zeta_orders <- 1:5
# View(sbs)
zeta.decline <- Zeta.decline.ex(sbs,
                                orders = zeta_orders,
                                plot = FALSE)

# Plot zeta decline
plot(zeta_orders,
     zeta.decline$zeta.val,
     type = "b",
     xlab = "Order", ylab = "Mean Zeta Diversity",
     main = "Zeta Decline")

# ---- 4b_Zeta_Decay ----
# Evaluate how zeta diversity changes with distance
# We need site coordinates (xy) + species (sbs)
zeta.decay <- Zeta.ddecay(
  data.spec   = sbs,          # species presence-absence matrix
  # xy          = cbind(xy_sf$x,xy_sf$y),# site coordinates
  xy          = cbind(sites_xy$x,sites_xy$y),# site coordinates
  order       = 2,            # zeta order of interest (e.g., 2)
  method      = "mean",       # or "sum"
  sam         = 100,           # number of distance bins (or set distances manually)
  plot        = FALSE         # set to TRUE to auto-plot
)

# The returned object contains distance bins (or pairs of sites, etc.) and
# average zeta diversity for each distance bin.

# Plot the result:
Plot.zeta.ddecay(zeta.decay)

# Multiple-Order Distance Decay Example
# If you want to see how zeta distance-decay behaves across multiple orders,
# you can use Zeta.ddecays():

zeta.decays <- Zeta.ddecays(
  data.spec   = sbs,
  xy          = cbind(sites_xy$x,sites_xy$y),
  orders      = 2:10,       # or pick any vector of orders
  method      = "mean",
  sam         = 1000,
  plot        = FALSE
)

# Then visualize:
Plot.zeta.ddecays(zeta.decays)

# Outcome:
# Measures of zeta decline across orders
# Measures of zeta decay with distance
# Plots illustrating these relationships

# //////////////////////////////////////////////////////////////////////////////
# 5. MS-GDM with Zeta.msgdm(sbs, sbe, xy)
# //////////////////////////////////////////////////////////////////////////////

# ---- 5a_Fit_MultiSite_GDM ----
# We now assume dissMapR has a function Zeta.msgdm() for multi-site GDM
# (this is hypothetical; adapt based on actual package usage)

library(dissMapR)  # hypothetical package

# Fit MS-GDM for orders = 2, 3, 5, 10
orders_to_fit <- c(2, 3, 5, 7)
msgdm_list <- list()

dim(sbs)
dim(sbe[,-1])
sbe$effort = as.numeric(sbe$effort)
sbe$richness = as.numeric(sbe$richness)
head(sbe[,c(-1,-23)])
dim(sites_xy)

nrow(sbs)
nrow(sbe[, c(-1,-23)])
nrow(sites_xy)

# Check if any columns in sbs are all zero:
colSums(sbs)
zero_rows <- which(rowSums(sbs) == 0)
length(zero_rows)

# Check if any columns in sbe are constant or NA:
sapply(sbe, function(x) length(unique(x[!is.na(x)])))

sbs_sub <- sbs[rowSums(sbs) > 0, ]
sbe_sub <- sbe[rowSums(sbs) > 0, ]
sites_xy_sub <- sites_xy[rowSums(sbs) > 0, ]

# Suppose sbs row names are numeric cell IDs
all( row.names(sbs) == sbe$cell_id )
all( row.names(sbs) == sites_xy$cell_id )

View(sbs)
View(sbe[,c(-1,-23)])
View(sites_xy[,2:3])

fit <- Zeta.msgdm(sbs,
                  sbe[,c(-1,-23)],
                  sites_xy[,2:3],
                  order = 2,
                  reg.type = 'ispline',
                  sam = 10000)

# Summaries or model stats
summary(fit$model)

dev.new()
graphics::plot(fit$model)

# Loop instead?

for (ord in orders_to_fit) {
  fit <- Zeta.msgdm(sbs,
                    sbe[,c(-1,-23)],
                    sites_xy[,2:3],
                    order = ord,
                    sam = 825)
  msgdm_list[[paste0("order", ord)]] <- fit
}

# Summaries or model stats
lapply(msgdm_list, summary)

# ---- 5b_Save_Model_Order2 ----
zeta2 <- msgdm_list[["order2"]]  # store the fitted order 2 model
summary(zeta2$model)

summary(msgdm_list[["order2"]]$model)
summary(msgdm_list[["order5"]]$model)

# Outcome:
# zeta2: fitted MS-GDM for order 2
# Additional model objects for orders 3, 5, and 10

# //////////////////////////////////////////////////////////////////////////////
# 6. Prediction with zeta2 (Present Scenario)
# //////////////////////////////////////////////////////////////////////////////

# ---- 6a_Uniform_Sampling ----
# Replace sampling effort in 'sbe' with its maximum value
sam.max <- max(sbe$sam.eff)
sbe_now <- sbe[,-1]
sbe_now$sam.eff <- sam.max
head(sbe_now)

# ---- 6b_Predict_zeta2 ----
head(zeta2$model$data)
zeta.now <- predict(zeta2$model, newdata = zeta2$model$data)
# zeta.now should be a site-by-site matrix of predicted dissimilarities (order=2)

# ---- 6c_Visualize_zeta_now ----
# (i) NMDS
library(vegan)
nmds_now <- metaMDS(zeta.now, k = 3, try = 20)

# Extract NMDS coordinates
nmds_coords <- as.data.frame(scores(nmds_now))
colnames(nmds_coords) <- c("NMDS1", "NMDS2", "NMDS3")

# (ii) RGB composite plot from NMDS axes
# Combine NMDS coords with site centroids 'xy'
# nmds_plot_df <- cbind(xy, nmds_coords)
nmds_plot_df <- cbind(sites.xy, nmds_coords)

ggplot(nmds_plot_df, aes(x = x, y = y)) +
  geom_point(aes(color = rgb(
    scales::rescale(NMDS1),
    scales::rescale(NMDS2),
    scales::rescale(NMDS3)
  )), size = 2) +
  scale_color_identity() +
  ggtitle("Present Scenario: NMDS RGB Composite") +
  theme_minimal()

# (iii) Clustering + Bioregions
# Example: hierarchical clustering
dim(as.dist(zeta.now))
head(zeta.now)

zeta.mat <- as.matrix(as.dist(zeta.now))
dim(zeta.mat)
hc_now <- hclust(as.dist(zeta.mat), method = "ward.D2")

# Choose number of clusters
k <- 5
bioregions_now <- cutree(hc_now, k = k)

# Outcome:
# zeta.now: predicted site-by-site dissimilarity matrix under uniform sampling
# NMDS-based RGB map
# Bioregional clusters (e.g., 5 clusters)

# //////////////////////////////////////////////////////////////////////////////
# 7. Prediction with zeta2 (Future Scenarios)
# //////////////////////////////////////////////////////////////////////////////

# ---- 7a_Prepare_Future_Env ----
# Suppose we have m future scenarios for environment (temp, precip).
# We'll combine them so that all scenarios + present are in one large sbe data frame.

# Create a placeholder example for 2 future scenarios
env1_future1 <- env1; values(env1_future1) <- values(env1_future1) + 2
env2_future1 <- env2; values(env2_future1) <- values(env2_future1) + 100

env1_future2 <- env1; values(env1_future2) <- values(env1_future2) + 4
env2_future2 <- env2; values(env2_future2) <- values(env2_future2) + 200

# Extract for each scenario
env_vals_future1 <- extract(c(env1_future1, env2_future1), cbind(xy$x, xy$y))[, -1]
env_vals_future2 <- extract(c(env1_future2, env2_future2), cbind(xy$x, xy$y))[, -1]

# Combine into data frames, reusing sam.max
sbe_future1 <- cbind(env_vals_future1, sam.eff = sam.max)
colnames(sbe_future1) <- c("temp", "precip", "sam.eff")

sbe_future2 <- cbind(env_vals_future2, sam.eff = sam.max)
colnames(sbe_future2) <- c("temp", "precip", "sam.eff")

# Combine present (sbe_now) + 2 future scenarios:
sbe_all <- rbind(sbe_now, sbe_future1, sbe_future2)

# ---- 7b_Predict_Future ----
zeta.future <- predict(zeta2, newdata = sbe_all)
# This will be a ((m+1)*n x (m+1)*n) dissimilarity matrix
# where m=2 (two future scenarios), n=number of sites

# ---- 7c_Visualize_Future ----
# We can subset 'zeta.future' to map each scenario individually
# For example, the first 'n' rows/cols = present, next 'n' = future1, last 'n' = future2
n_sites <- nrow(sbe_now)
present_idx   <- 1:n_sites
future1_idx   <- (n_sites+1):(2*n_sites)
future2_idx   <- (2*n_sites+1):(3*n_sites)

zeta.present  <- zeta.future[present_idx, present_idx]
zeta.fut1     <- zeta.future[future1_idx, future1_idx]
zeta.fut2     <- zeta.future[future2_idx, future2_idx]

# Optionally run NMDS or clustering for each scenario
# Here, just an example with future scenario 1
nmds_fut1 <- metaMDS(zeta.fut1, k = 3, try = 20)
nmds_coords_fut1 <- as.data.frame(scores(nmds_fut1))
# (Plot similarly as above, but note the site order matches rows in future1_idx)

# Outcome:
# zeta.future: site-by-site predicted dissimilarities for present + multiple future scenarios
# Tools for NMDS/clustering to map future shifts in bioregions

# //////////////////////////////////////////////////////////////////////////////
# 8. Data Publication to Zenodo
# //////////////////////////////////////////////////////////////////////////////

# ---- 8a_Prepare_Outputs ----
# Collect your final data frames and raster objects:
#   - sbs, xy, sbe, zeta.now, zeta.future, etc.
#   - Maps (sampling effort, zeta plots, NMDS RGB maps, cluster results)
#   - Model objects (zeta2, etc.)
#
# Save them to disk as CSV, RDS, GeoTIFF, etc.
saveRDS(sbs, "sbs.rds")
saveRDS(xy, "xy.rds")
saveRDS(sbe, "sbe.rds")
saveRDS(zeta.now, "zeta_now.rds")
saveRDS(zeta.future, "zeta_future.rds")
# ... etc.

# ---- 8b_Zenodo_Upload ----
# Use zen4R or manual upload:
# install.packages("zen4R")
library(zen4R)

# Provide your Zenodo token, metadata, etc.
zenodo <- ZenodoManager$new(token = "YOUR_ZENODO_TOKEN")

# Create a new deposition, set metadata
my_deposition <- zenodo$createEmptyRecord()
my_deposition <- zenodo$setMetadata(
  my_deposition,
  title = "Multi-Site Dissimilarity Data & Models",
  upload_type = "dataset",
  description = "Data and model outputs from the dissMapR workflow.",
  creators = list(
    list(name = "Your Name", affiliation = "Your Institution")
  )
)

# Then upload files:
# zenodo$uploadFile("sbs.rds", my_deposition$id)
# zenodo$uploadFile("sbe.rds", my_deposition$id)
# etc.

# Finally, publish:
# zenodo$publishRecord(my_deposition$id)

# Outcome:
# All necessary data (e.g., sbs, xy, sbe, zeta.now, zeta.future, figures)
# are archived on Zenodo for reproducibility.

# //////////////////////////////////////////////////////////////////////////////
# Final Remarks
# //////////////////////////////////////////////////////////////////////////////

# This script demonstrates the core steps for dissecting compositional turnover and
# bioregional patterns under present or future scenarios using dissMapR
# (and related packages).
# Each step can be adapted to use your specific data sources and packages.
# Always verify spatial extents, coordinate reference systems, and the validity of
# environmental data before scaling up analyses.
