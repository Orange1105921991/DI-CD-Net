import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from utils.utils1 import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

coordinates = {}

def get_coordinates(image_name):
    if image_name in coordinates:
        return coordinates[image_name]
    else:
        return None


if __name__ == "__main__":
    map_mode = 0
    classes_path = 'model_data/AOI_class.txt'
    MINOVERLAP = 0.5
    confidence = 0.001
    nms_iou = 0.5
    score_threhold = 0.5
    map_vis = False
    map_out_path = f'mapout_0326_0228AOI_bs8_ep300_orange2_decode2_o_ciou_N4_True_FClTrue_hsvTrue_11/IOU{MINOVERLAP}_conf{confidence}_nmsiou{nms_iou}_nocut'
    val_file_path = '/home/WQL/datasets/0228AOI/test/file.txt'
    val_img_path = '/home/WQL/datasets/0228AOI/test'

    with open(val_file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split(' ')
        image_name = parts[0]
        boxes = parts[1:]

        coordinates[image_name] = []

        for box in boxes:
            left, top, right, bottom, _ = map(int, box.split(','))
            coordinates[image_name].append((left, top, right, bottom))

    with open(val_file_path, 'r') as file:
        lines = file.readlines()
    image_ids = []
    for line in lines:
        file_name = line.split(' ')[0].strip()
        image_ids.append(file_name)

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image1_path = os.path.join(val_img_path, "im1/" + image_id)
            image2_path = os.path.join(val_img_path, "im2/" + image_id)
            image1 = Image.open(image1_path)
            image2 = Image.open(image2_path)
            if map_vis:
                image1.save(os.path.join(map_out_path, "images-optional/" + image_id))
                image2.save(os.path.join(map_out_path, "images-optional/" + image_id))
            yolo.get_map_txt(image_id, image1, image2, class_names, map_out_path)
        print("Get predict result done.")

    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/" + image_id.split('.')[0] + ".txt"), "w") as new_f:
                found_coordinates = get_coordinates(image_id)

                for obj in found_coordinates:
                    difficult_flag = False
                    obj_name = 'change'

                    left = obj[0]
                    top = obj[1]
                    right = obj[2]
                    bottom = obj[3]

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")
    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=map_out_path)
        print("Get map done.")
