import numpy as np
from laspy.file import File


class LasReader:

    @staticmethod
    def read(file):
        """
        :param file: path to file
        :return: laspy object containing point cloud
        """
        return File(file, mode="r")

    @staticmethod
    def scaled_dimension(las_file, dim):
        scale = las_file.header.scale[0]
        offset = las_file.header.offset[0]
        return dim * scale + offset

    @staticmethod
    def get_scaled_dimensions(las_file):
        x_dim = LasReader.scaled_dimension(las_file, las_file.X)
        y_dim = LasReader.scaled_dimension(las_file, las_file.Y)
        z_dim = LasReader.scaled_dimension(las_file, las_file.Z)
        return np.c_[x_dim, y_dim, z_dim]

    @staticmethod
    def get_dimensions(las_file):
        return np.c_[las_file.x, las_file.y, las_file.z]

    @staticmethod
    def get_xyz(path):
        infile = LasReader.read(path)
        xyz = LasReader.get_dimensions(infile)
        infile.close()
        return xyz
