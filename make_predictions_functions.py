from read_xml import xml_to_coordinates
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as F
import cv2


import xml.etree.cElementTree as et
import large_image
import math
import torch
from scipy import ndimage
from lxml import etree



def xml_to_coordinates(xml_path):

    """
    Tim gave me this peace of code
    Take XML annotations from ASAP and return coordinates
    :param path: path of xml file
    :return: returns a Dictionary.
    key --> 'Annotation 1 Rectangle'
    values --> list of tuple coordinates --> [(15682, 105852), (19210, 105852), (19210, 107848), (15682, 107848)]
    """
    xtree = et.parse(xml_path)
    xroot = xtree.getroot()
    xml_coordinates = {}
    for node in xroot:
        for annotation in node:
            name = annotation.attrib.get("Name")
            name += " "
            name += annotation.attrib.get("Type")
            for all_coords in annotation:
                tuples_of_coordinates = []
                for coordinates in all_coords:
                    x = round(float(coordinates.attrib.get("X")))
                    y = round(float(coordinates.attrib.get("Y")))
                    tuples_of_coordinates.append((x, y))
                xml_coordinates[name] = tuples_of_coordinates
    return xml_coordinates



def load_ROI(WSI_path, xml_path, model_input_size):
    """
    This function extracts Regions Of Interest (ROI) from WSI,
    Each entry in the xml_file which is denoted as "Rectangel" will be used as a ROI
    This function makes the ROI a little bit bigger, so that when we divide the ROI into patches
    all patches will have the required model input size. --> all patches should fit on the ROI
    :return:
    returns a list of arrays (each array is one ROI)
    array shape: (H, W, 4)
    """

    ROI_images = {}
    magnification = 20

    # necessary for loading the ROI with the large image library
    ts = large_image.getTileSource(WSI_path)

    # get all ROI coordinates
    dictionary_of_all_coordinates = xml_to_coordinates(xml_path)
    for key in dictionary_of_all_coordinates:

        # get one ROI coordinate
        if key.endswith('Rectangle'):
            ROI = dictionary_of_all_coordinates[key]
            left = ROI[0][0]
            top = ROI[0][1]
            ROI_height = ROI[2][1] - ROI[0][1]
            ROI_width = ROI[1][0] - ROI[0][0]

            # round up to the next integer (the size of the ROI should be divisible by the model_input_size)
            # makes the ROI a little bit bigger --> so that the sliding window matches exactly inside the ROI
            num_patches_in_y = math.ceil(ROI_height / model_input_size)
            num_patches_in_x = math.ceil(ROI_width / model_input_size)

            # the new ROI size is a little bit bigger
            new_ROI_height = model_input_size * num_patches_in_y
            new_ROI_width = model_input_size * num_patches_in_x

            # uses large image library to load the ROI
            ROI, _ = ts.getRegionAtAnotherScale(
                sourceRegion=dict(left=left, top=top, width=new_ROI_width, height=new_ROI_height,
                                  units='base_pixels'), targetScale=dict(magnification=magnification),
                format=large_image.tilesource.TILE_FORMAT_NUMPY)

            ROI_images[key] = ROI


    return ROI_images


def plot_ROI(ROI_images):
    for i in ROI_images:
        img = ROI_images[i]

        # each ROI has to be converted from RGBA to RGB (maybe incoporated into Load_ROI_for_predictiom
        img_RGB = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        plt.figure(figsize=(20, 10))
        plt.imshow(img_RGB)
        plt.show()


def cut_ROI(ROI_images, model_input_size, overlay):
    """
    cuts the ROI into patches, matching the model input size
    :return:
    returns a dictionary
    keys are the ROI names --> ROI_0, ROI_1, ROI_2
    each value is a list of patches. Each patch is an array of shape (model_input_size, model_input_size)
    """
    ### Cut ROI with specific patch size and overlay ###

    patch_size = model_input_size
    ROI_patches = {}
    step_size = overlay * patch_size

    for c, key in enumerate(ROI_images):
        ROI = ROI_images[key]

        ROI_height = np.shape(ROI)[0]
        ROI_width = np.shape(ROI)[1]

        # check if overlay is feasible with ROI size and patch_size
        # the Length: (ROI - patch_size),  must be divisble by the length of an non-overlapping
        # end of an overlay
        if not (ROI_height - patch_size) % ((1 - overlay) * patch_size) == 0:
            print("overlay does not match")
            print("overlay: ", overlay)
            print("patch_size", patch_size)
            print("ROI height: ", np.shape(ROI)[0])
            print((ROI_height - patch_size) % ((1 - overlay) * patch_size))
            break
        if not (ROI_width - patch_size) % ((1 - overlay) * patch_size) == 0:
            print("overlay does not match")
            print("overlay: ", overlay)
            print("patch_size", patch_size)
            print("ROI width: ", np.shape(ROI)[1])
            print((ROI_height - patch_size) % ((1 - overlay) * patch_size))
            break

        # # the number of iteration
        # is equal to the Length: (ROI - patch_size), divided by the length of an non-overlapping
        # end of an overlay (non-overlapping end= stepsize) + 1
        iterations_in_y_direction = int((ROI_height - patch_size) / (patch_size * (1 - overlay)) + 1)
        iterations_in_x_direction = int((ROI_width - patch_size) / (patch_size * (1 - overlay)) + 1)



        patch_list = []

        start_y = int(-patch_size + (patch_size * (overlay)))
        end_y = int(patch_size * overlay)

        for row in range(iterations_in_y_direction):

            # next step
            start_y += int(patch_size * (1 - overlay))
            end_y += int(patch_size * (1 - overlay))

            start_x = int(-patch_size + (patch_size * (overlay)))
            end_x = int(patch_size * overlay)

            for col in range(iterations_in_x_direction):
                start_x += int(patch_size * (1 - overlay))
                end_x += int(patch_size * (1 - overlay))


                patch = ROI[start_y:end_y, start_x:end_x, :]
                patch_list.append(patch)

                # TODO: apply same transforms like in deepDIVA so that patches fit the model
            #print(len(patch_list))
        ROI_patches[key] = patch_list

    return ROI_patches

def transform_patches(ROI_patches):
    """
    Input: dictionary --> {ROI_0: array_list_0, ROI_1: array_list_1,......}
    transforms each single patch, to a tensor, so that it can be used as an input, for a pytorch model
    :return:
    dictionary  --> {ROI_0: tensor_list_0, ROI_1: tensor_list_1,......} with
    """
    transformed_ROI_patches = {}
    for key in ROI_patches:
        new_patch_list = []
        for patch in ROI_patches[key]:
            patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
            patch = np.transpose(patch, (2, 0, 1))
            # maybe we can deleat this line
            patch = torch.tensor(patch, dtype=float)
            patch = patch.unsqueeze(0)
            patch = torch.DoubleTensor(patch)
            patch = patch.type(torch.cuda.FloatTensor)
            new_patch_list.append(patch)

        transformed_ROI_patches[key] = new_patch_list
    return transformed_ROI_patches


def load_model(weights_path, model_architecture):
    model = model_architecture
    model = torch.nn.DataParallel(model).cuda()
    model_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(model_dict['state_dict'], strict=False)
    return model

###########################################################################################################

def predict_class_probability_for_each_patch(weights_path, model_architecture, ROI_patches):
    model = load_model(weights_path, model_architecture)
    transformed_ROI_patches = transform_patches(ROI_patches)
    predicted_class_probabilities_for_each_patch = {}
    for key in transformed_ROI_patches:
        predictions = []
        for Input in transformed_ROI_patches[key]:
            output = model(Input)
            output = output.detach()
            # because of: TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
            output = output.cpu()
            output = np.array(output)
            predictions.append(output)

            output = output[0]
        predicted_class_probabilities_for_each_patch[key] = predictions

    return predicted_class_probabilities_for_each_patch


def concatenate_predicted_patches(weights_path, model_architecture, ROI_patches, ROI_images, model_input_size, overlay):
    patch_size = model_input_size
    full_output_dict = {}
    for key in ROI_images:
        ROI = ROI_images[key]
        ROI_height = np.shape(ROI)[0]
        ROI_width = np.shape(ROI)[1]

        all_predictions = predict_class_probability_for_each_patch(weights_path, model_architecture, ROI_patches)
        prediction_list = all_predictions[key]

        # check if overlay is feasible with ROI size and patch_size
        # the Length: (ROI - patch_size),  must be divisble by the length of an non-overlapping
        # end of an overlay
        if not (ROI_height - patch_size) % ((1 - overlay) * patch_size) == 0:
            print("overlay does not match")
            print("overlay: ", overlay)
            print("patch_size", patch_size)
            print("ROI height: ", np.shape(ROI)[0])
            print((ROI_height - patch_size) % ((1 - overlay) * patch_size))
            break

        if not (ROI_width - patch_size) % ((1 - overlay) * patch_size) == 0:
            print("overlay does not match")
            print("overlay: ", overlay)
            print("patch_size", patch_size)
            print("ROI width: ", np.shape(ROI)[1])
            print((ROI_height - patch_size) % ((1 - overlay) * patch_size))
            break


        # the number of iterastion
        # is equal to the Length: (ROI - patch_size), divided by the length of an non-overlapping
        # end of an overlay (non-overlaping end= stepsize) + 1
        iterations_in_y_direction = int((ROI_height - patch_size) / (patch_size * (1 - overlay)) + 1)
        iterations_in_x_direction = int((ROI_width - patch_size) / (patch_size * (1 - overlay)) + 1)

        # creat empthy image
        full_output = np.empty((2, np.shape(ROI_images[key])[0], np.shape(ROI_images[key])[1]))
        full_output.fill(np.nan)

        start_y = int(-patch_size + (patch_size * (overlay)))
        # end_y = int(patch_size * (1 - overlay))
        end_y = int(patch_size * overlay)

        counter = -1
        for row in range(iterations_in_y_direction):

            # next step
            start_y += int(patch_size * (1 - overlay))
            end_y += int(patch_size * (1 - overlay))

            start_x = int(-patch_size + (patch_size * (overlay)))
            # end_x = int(patch_size * (1-overlay))
            end_x = int(patch_size * overlay)

            for col in range(iterations_in_x_direction):
                start_x += int(patch_size * (1 - overlay))
                end_x += int(patch_size * (1 - overlay))

                counter += 1

                output = prediction_list[counter]

                mask = np.isnan(full_output[:, start_y:end_y, start_x:end_x])

                #
                # inserts predicted patches into target array
                # ....
                # if still NaN in full_output just insert value from crop (output), else (there is a value) then take max
                full_output[:, start_y:end_y, start_x:end_x] = np.where(mask, output, np.maximum(output,
                                                                                                 full_output[:,
                                                                                                 start_y:end_y,
                                                                                                 start_x:end_x]))


        a = np.array(full_output)
        a = np.transpose(a, (1, 2, 0))
        full_output_dict[key] = a

    return full_output_dict

def argmax(ROI_class_probability_prediction):

    ROI_binary_prediciton = {}
    for key in ROI_class_probability_prediction:
        ROI_pred = ROI_class_probability_prediction[key]
        ROI_binary = np.argmax(ROI_pred, axis=2)
        ROI_binary_prediciton[key] = ROI_binary
    return ROI_binary_prediciton


# TODO check the effect of the arguments in getstructuringELement(cv2.MORPH_ELLIPSE, (11,11)
# TODO define parameters inside the functions as class-varaibles,

def erosion_followed_by_dilation(ROI_binary_prediction):

    ROI_erosion_and_dilation_POST_PROCESS_1 = {}
    for key in ROI_binary_prediction:
        ROI = ROI_binary_prediction[key]
        ROI = ROI.astype(np.uint8)

        # makes ROI to a 3-channel image by copying the single channel three times
        ROI_rgb = np.zeros((*np.shape(ROI), 3))
        for i in range(3):
            ROI_rgb[:,:,i] = ROI[:,:]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        (thresh, binRed) = cv2.threshold(ROI_rgb, 10000, 20000, cv2.THRESH_BINARY)
        ROI_ero_dil = cv2.morphologyEx(ROI_rgb, cv2.MORPH_OPEN, kernel, iterations=3)
        ROI_erosion_and_dilation_POST_PROCESS_1[key] = ROI_ero_dil

    return ROI_erosion_and_dilation_POST_PROCESS_1


def remove_small_objects(ROI_erosion_and_dilation):
    """

    :param ROI_erosion_and_dilation:
    :return:
    """
    ROI_small_object_removed_POST_PROCESS_2 = {}
    for key in ROI_erosion_and_dilation:
        ROI = ROI_erosion_and_dilation[key]
        Zlabeled, Nlabels = ndimage.measurements.label(ROI)
        label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
        # for label, size in enumerate(label_size):
        #     print("label %s is %s pixels in size" % (label, size))

        # now remove the labels
        for label, size in enumerate(label_size):
            if size < 50000:
                ROI[Zlabeled == label] = 0

        ROI_small_object_removed_POST_PROCESS_2[key] = ROI
    return ROI_small_object_removed_POST_PROCESS_2


def fill_holes(ROI_small_object_removed_POST_PROCESS_2):
    """

    :param ROI_small_object_removed_POST_PROCESS_2:
    :return:
    """
    ROI_holes_filled_POST_PROCESS_3 = {}
    # invert thevalues of the ROI
    for key in ROI_small_object_removed_POST_PROCESS_2:
        ROI = ROI_small_object_removed_POST_PROCESS_2[key]
        ROI = ROI.astype(np.uint8)
        ROI[ROI == 1] = 255
        ROI = np.invert(ROI)
        Zlabeled, Nlabels = ndimage.measurements.label(ROI)
        label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
        # for label, size in enumerate(label_size):
        #     print("label %s is %s pixels in size" % (label, size))

        # now remove the labels
        for label, size in enumerate(label_size):
            if size < 100000:
                ROI[Zlabeled == label] = 0
                ROI_holes_filled_POST_PROCESS_3[key] = ROI


    return ROI_holes_filled_POST_PROCESS_3
#TODO at the end the right end of the ROI should be cut of, same amount that we have made the image bigger

def find_contours(ROI_holes_filled):
    """

    :param ROI_holes_filled:
    :return:
    """
    # https://stackoverflow.com/questions/30757273/opencv-findcontours-complains-if-used-with-black-white-image
    ROI_contours = {}
    ROI_contour_images = {}

    for key in ROI_holes_filled:
        ROI = ROI_holes_filled[key]

        imgray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 1)
        # the contours variable will have a channel for each shape, each channel contains all coordinates of the shape
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ROI_contours[key] = contours
        ###############################################################################

        # create an empthy image with same size like original image
        out = np.zeros_like(ROI)

        # Draw contours in an empty image
        contour_image = cv2.drawContours(out, contours, -1, (0, 0, 254), 5)
        ROI_contour_images[key] = contour_image

    return ROI_contours, ROI_contour_images

# TODO the Rectangle coordinates are not included in the xml file. When I tryed that ASAP crashed when loading the xml file in ASAP
# TODO apply a function which reduces the number of coordinates but preserves the shape of the prediction

def save_contours_as_coordinates_in_xml_file(ROI_contours, xml_coordinates, target_path, name_prediction_file):
    # calculates image coordinates relative to WSI coordinates
    ROI_annotation_coordinates = {}
    for key in ROI_contours:
        contours = ROI_contours[key]
        x_coordinate = xml_coordinates[key][0][1]   # takes coordinates from the input Rectangle
        y_coordinate = xml_coordinates[key][0][0]

        Annotation_coords_lst = []
        for contour in contours:
            # we have to make a copy, otherways we also altering the original list
            annotation = np.copy(contour)
            annotation[:, :, 0] = annotation[:, :, 0] + y_coordinate
            annotation[:, :, 1] = annotation[:, :, 1] + x_coordinate
            #print(annotation)
            Annotation_coords_lst.append(annotation)
        ROI_annotation_coordinates[key] = Annotation_coords_lst



    # Create the root element
    page = etree.Element('ASAP_Annotations')
    # Make a new document tree
    doc = etree.ElementTree(page)
    Annotations = etree.SubElement(page, 'Annotations')

    # Coordinates for Polygons
    for key in ROI_annotation_coordinates:
        Annotation_coords_lst = ROI_annotation_coordinates[key]

        for i in range(1, len(Annotation_coords_lst)):
            Annotation = etree.SubElement(Annotations, 'Annotation',
                                          Name="Annotation {}".format(i),
                                          Type='Polygon',
                                          PartOfGroup='None',
                                          Color="#F4FA58")
            Coordinates = etree.SubElement(Annotation, 'Coordinates')

            counter = -1
            for x, y, in np.squeeze(Annotation_coords_lst[i], axis=1):
                counter += 1
                Coordinate = etree.SubElement(Coordinates, 'Coordinate', Order=str(counter), X=str(x), Y=str(y))

        AnnotationGroups = etree.SubElement(page, "AnnotationGroups")

        # Save to XML file
        outFile = open(target_path + '/' + name_prediction_file + '.xml', 'w')
        doc.write(target_path + '/' + name_prediction_file + '.xml', xml_declaration=True, encoding='utf-16')

        # load the file and save it with new lines --> pretty print
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(target_path + '/' + name_prediction_file + '.xml', parser)
        tree.write(target_path + '/' + name_prediction_file + '_pretty' + '.xml', pretty_print=True, xml_declaration=True)

    return None
















