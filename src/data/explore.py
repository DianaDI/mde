from osgeo import gdal
from src.data.las2xyz import LasReader

pc_path = "/mnt/data/davletshina/datasets/Bera_MDE/KirbyLeafOff2017PointCloudEntireSite.las"
tif_path = "/mnt/data/davletshina/datasets/Bera_MDE/KirbyLeafOff2017RGBNEntireSitePCCrop.tif"

if __name__ == "__main__":
    print("EXPLORE IMG:")
    raster = gdal.Open(tif_path)
    width = raster.RasterXSize
    height = raster.RasterYSize
    print(f'GIVEN IMG: {width}x{height}')

    print("GEO INFO FROM TIFF: ")
    print(raster.GetProjection())
    print(raster.GetMetadata())

    gt = raster.GetGeoTransform()
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]

    print("min x: ", minx)
    print("min y: ", miny)
    print("max x: ", maxx)
    print("max y: ", maxy)

    print("EXPLORE PC:")
    pc_file = LasReader.read(pc_path)
    print("GEO INFO FROM PC:")
    # geokey_directorytag = pc_file._header.vlrs[0].parsed_body
    # lasf_proj = pc_file._header.vlrs[2].parsed_body[0].decode("utf-8")
    full_proj = pc_file._header.vlrs[3].VLR_body.decode("utf-8")
    print(full_proj)
    xyz = LasReader.get_scaled_dimensions(pc_file)

    pc_minx = min(xyz[:, 0])
    pc_maxx = max(xyz[:, 0])
    pc_miny = min(xyz[:, 1])
    pc_maxy = max(xyz[:, 1])
    pc_minz = min(xyz[:, 2])
    pc_maxz = max(xyz[:, 2])
    print("min x: ", pc_minx)
    print("min y: ", pc_miny)
    print("max x: ", pc_maxx)
    print("max y: ", pc_maxy)
    print("min z: ", pc_minz)
    print("max z: ", pc_maxz)
