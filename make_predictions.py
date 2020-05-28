
#from read_xml import xml_to_coordinates


import large_image

# My imports
#from read_xml import xml_to_coordinates
from make_predictions_functions import xml_to_coordinates, load_ROI, plot_ROI, cut_ROI, transform_patches, load_model, predict_class_probability_for_each_patch, \
    concatenate_predicted_patches, argmax, erosion_followed_by_dilation, remove_small_objects, fill_holes, find_contours, save_contours_as_coordinates_in_xml_file

from UNet import Unet
# from post_process import softmax, crf
# import time
# import torch
# import cv2
# import math
# import numpy as np
# import matplotlib.pyplot as plt


# what do you wnat to have as an object?
    # Each ROI should be an object and not, a list of  ROI
        # makes it easyer to understand
        # maybe it is also easyer to extract single ROI, and its itermediate products


#instance variables ar
class MakePrediction:

    # Class variables
    model_input_size = 256
    overlay = 0.5
    model_architecture = Unet()


    def __init__(self, WSI_path, xml_path, weights_path, model_architecture):
        self.WSI_path = WSI_path
        self.xml_path = xml_path
        # self.target_path = target_path
        # self.name_prediction_file = name_prediction_file
        self.weights_path = weights_path
        self.model_architecture = model_architecture

        # self.xml_coordinates = None
        # self.ROI_names = None
        # self.ROI_image = None
        # self.ROI_patches = None

        self.xml_coordinates = xml_to_coordinates(self.xml_path)
        self.ROI_names = list(self.xml_coordinates.keys())
        self.ROI_images = load_ROI(self.WSI_path, self.xml_path, self.model_input_size)
        self.ROI_patches = cut_ROI(self.ROI_images, self.model_input_size, self.overlay)


        self.patch_class_probability_prediction = None
        self.ROI_class_probability_prediction = None
        self.ROI_binary_prediction = None
        self.ROI_erosion_and_dilation_POST_PROCESS_1 = None
        self.ROI_small_object_removed_POST_PROCESS_2 = None
        self.ROI_holes_filled_POST_PROCESS_3 = None
        self.ROI_contours = None
        self.ROI_contour_image = None
        self.output_xml = None
        # self.ROI_normalize = None
        # self.load_model = load_model(self.weights_path, self.model_architecture)

    # I want to be able to extract in between steps of  the pipline
    #
    # def create_xml_coordinates(self):
    #     self.xml_coordinates = xml_to_coordinates(self.xml_path)
    #     return self.xml_coordinates
    #
    # def create_ROI_names(self):
    #     if self.xml_coordinates is None:
    #         self.create_xml_to_coordinates()
    #         self.ROI_names = list(self.xml_coordinates.keys())
    #     else:
    #         self.ROI_names = list(self.xml_coordinates.keys())
    #     return self.ROI_names
    #
    # def create_ROI_image(self):
    #     self.ROI_image = load_ROI(self.WSI_path, self.xml_path, self.model_input_size)
    #     return self.ROI_image



    def load_model(self):
        return load_model(self.weights_path, self.model_architecture)

    def predict_class_probability_for_each_patch(self):
        self.patch_class_probability_prediction = predict_class_probability_for_each_patch(self.weights_path, self.model_architecture, self.ROI_patches)
        return self.patch_class_probability_prediction


    def predict_class_probability_for_ROI(self):
        if self.patch_class_probability_prediction is None:
            self.predict_class_probability_for_each_patch
            self.ROI_class_probability_prediction = concatenate_predicted_patches(self.weights_path, self.model_architecture,self.ROI_patches, self.ROI_images, self.model_input_size, self.overlay)
            return self.ROI_class_probability_prediction
        else:
            self.ROI_class_probability_prediction = concatenate_predicted_patches(self.weights_path, self.model_architecture,self.ROI_patches, self.ROI_images, self.model_input_size, self.overlay)
            return self.ROI_class_probability_prediction


    def binary_classifiaction_ROI(self):
        if self.ROI_class_probability_prediction is None:
            self.predict_class_probability_for_ROI()
            self.ROI_binary_prediction = argmax(self.ROI_class_probability_prediction)
            return self.ROI_binary_prediction
        else:
            self.ROI_binary_prediction = argmax(self.ROI_class_probability_prediction)
            return self.ROI_binary_prediction

    def erosion_followed_by_dilation(self):
        if self.ROI_binary_prediction is None:
            self.binary_classifiaction_ROI()
            self.ROI_erosion_and_dilation_POST_PROCESS_1 = erosion_followed_by_dilation(self.ROI_binary_prediction)
            return self.ROI_erosion_and_dilation_POST_PROCESS_1
        else:
            self.ROI_erosion_and_dilation_POST_PROCESS_1 = erosion_followed_by_dilation(self.ROI_binary_prediction)
            return self.ROI_erosion_and_dilation_POST_PROCESS_1

    def remove_small_objects(self):
        if self.ROI_erosion_and_dilation_POST_PROCESS_1 is None:
            self.erosion_followed_by_dilation()
            self.ROI_small_object_removed_POST_PROCESS_2 = remove_small_objects(self.erosion_followed_by_dilation())
            return self.ROI_small_object_removed_POST_PROCESS_2
        else:
            self.ROI_small_object_removed_POST_PROCESS_2 = remove_small_objects(self.erosion_followed_by_dilation())
            return self.ROI_small_object_removed_POST_PROCESS_2

    def fill_holes(self):
        if self.ROI_small_object_removed_POST_PROCESS_2 is None:
            self.remove_small_objects()
            self.ROI_holes_filled_POST_PROCESS_3 = fill_holes(self.ROI_small_object_removed_POST_PROCESS_2)
            return self.ROI_holes_filled_POST_PROCESS_3
        else:
            self.ROI_holes_filled_POST_PROCESS_3 = fill_holes(self.ROI_small_object_removed_POST_PROCESS_2)
            return self.ROI_holes_filled_POST_PROCESS_3

    def find_contours(self):
        if self.ROI_holes_filled_POST_PROCESS_3 is None:
            self.fill_holes()
            self.ROI_contours, self.ROI_contour_image= find_contours(self.ROI_holes_filled_POST_PROCESS_3)
            return self.ROI_contours, self.ROI_contour_image
        else:
            self.ROI_contours = find_contours(self.ROI_holes_filled_POST_PROCESS_3)
            return self.ROI_contours, self.ROI_contour_image

    def save_coordinates_of_prediction_in_xml_file(self, target_path, name_prediction_file):
        if self.ROI_contours is None:
            self.find_contours()
            self.output_xml = save_contours_as_coordinates_in_xml_file(self.ROI_contours, self.xml_coordinates, target_path, name_prediction_file)
            return self.output_xml
        else:
            self.output_xml = save_contours_as_coordinates_in_xml_file(self.ROI_contours, self.xml_coordinates, target_path, name_prediction_file)
            return self.output_xml



#
if __name__== '__main__':
    import os

    # save the rectangle inside the MakePrediction folder
    # only one file should be inside this folder
    # for fast demonstration of the programm
    folder_path = "/home/sven/Desktop/data/WSI/MakePrediction"
    target_path = "/home/sven/Desktop/data/WSI/MakePrediction_Result"
    name_prediction_file = "prediction"
    model_architecture = Unet()

    file = os.listdir(folder_path)[0]
    xml_file_path = os.path.join(folder_path, file)


    #WSI_path = "/home/sven/Desktop/data/WSI/02.11715_1E_HE.mrxs"
    WSI_path = "/home/sven/Desktop/data/WSI/SS11.17124_2E1_HE.mrxs"
    weights_path = "/home/sven/Desktop/results/Abstract_results/GlaS_pT1_SMALL/pT1_SMALL/model_name=unet/batch_size=32/epochs=60/lr=0.01/decay_lr=10/momentum=0.9/workers=5/imgs_in_memory=5/crop_size=256/crops_per_image=10/19-01-20-16h-57m-13s/model_best.pth.tar"

    instance_1 = MakePrediction(WSI_path, xml_file_path, weights_path=weights_path, model_architecture=model_architecture)




    c = instance_1.save_coordinates_of_prediction_in_xml_file(target_path, name_prediction_file)
    print(type(c))
    print(c)

    # for key in c:
    #     img = c[key]
    #     plt.imshwo(img)
    #     plt.show()














