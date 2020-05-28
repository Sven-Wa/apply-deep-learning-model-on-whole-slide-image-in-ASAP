import xml.etree.cElementTree as et
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import large_image
import numpy
from PIL import Image, ImageDraw
from PIL import ImagePath
path = "/home/sven/Segmentation/data/WSI/SS09.24550_2A_HE.xml"

def xml_to_coordinates(path):
    """
    Take XML annotations from ASAP and return coordinates
    :param path: path of xml file
    :return: coordinates as list of tuples
    """
    xtree = et.parse(path)
    xroot = xtree.getroot()
    xml_coordinates = {}
    for node in xroot:
        for annotation in node:
            name = annotation.attrib.get("Name")
            name+= " "
            name+= annotation.attrib.get("Type")
            for all_coords in annotation:
                tuples_of_coordinates=[]
                for coordinates in all_coords:
                    x = round(float(coordinates.attrib.get("X")))
                    y = round(float(coordinates.attrib.get("Y")))
                    tuples_of_coordinates.append((x,y))
                xml_coordinates[name] = tuples_of_coordinates
    return xml_coordinates
################################################################################l






