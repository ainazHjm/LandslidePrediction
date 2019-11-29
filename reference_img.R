library(raster)
library(rgdal)

reference_image <- function(unref_img_path, ref_img_path, wrt_to)
{
  crf_img <- raster(ref_img_path)
  img <- raster(unref_img_path)
  
  values(crf_img) <- values(img)
  writeRaster(crf_img, wrt_to, format = "GTiff") 
}