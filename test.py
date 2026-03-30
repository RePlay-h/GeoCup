from osgeo import gdal

gdal.VectorTranslate(
    "client/public/data/buildings.pmtiles",
    "data/processed/frontend.geojson",
    format="PMTiles"
)