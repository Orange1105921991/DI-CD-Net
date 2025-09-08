import time

import cv2
import numpy as np
from PIL import Image
import os
from yolo import YOLO, YOLO_ONNX

if __name__ == "__main__":
    mode = "fps"
    crop = False
    count = False
    root = '/home/WQL/datasets/0228AOI/test/'
    dir_origin_path1 = root + "im1/"
    dir_origin_path2 = root + "im2/"
    dir_origin_path3 = root + "im2/"
    dir_save_path = "/home/WQL/project/yolo4-orange-insprion/res/0228AOI/Ours"
    heatmap_save_path = "/home/WQL/project/yolo4-orange-insprion/heatmap/DI-CD/"
    simplify = True
    onnx_save_path = "model_data/models.onnx"
    test_interval = 100
    fps_path1 = "/home/WQL/datasets/0210AOI/test/im1/A_10_12.jpg"
    fps_path2 = "/home/WQL/datasets/0210AOI/test/im2/A_10_12.jpg"
    if mode != "predict_onnx":
        yolo = YOLO()
    else:
        yolo = YOLO_ONNX()

    if mode == "predict":
        img1 = "/wql/yolo4-orange/3.jpg"
        img2 = "/wql/yolo4-orange/4.jpg"
        try:
            image1 = Image.open(img1)
            image2 = Image.open(img2)
        except:
            print('Open Error! Try again!')

        else:
            r_image = yolo.detect_image(image1, image2, crop=crop, count=count)
            r_image.save("res.jpg")

    elif mode == "dir_predict":

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path1)
        all_t = 0
        count = 0.0
        for img_name in tqdm(img_names):
            count += 1
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path1 = os.path.join(dir_origin_path1, img_name)
                image_path2 = os.path.join(dir_origin_path2, img_name)
                image_path3 = os.path.join(dir_origin_path3, img_name)
                image1 = Image.open(image_path1)
                image2 = Image.open(image_path2)
                image3 = Image.open(image_path3)
                image3 = image3.convert("RGB")
                start = time.time()
                r_image1, r_image2, r_image3 = yolo.detect_image(image1, image2, image3, img_name)
                end = time.time()
                all_t += end - start
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path, exist_ok=True)
                im1_path = os.path.join(dir_save_path, 'im1')
                im2_path = os.path.join(dir_save_path, 'im2')
                im3_path = os.path.join(dir_save_path, 'label')
                if not os.path.exists(im1_path):
                    os.makedirs(im1_path)

                if not os.path.exists(im2_path):
                    os.makedirs(im2_path)
                if not os.path.exists(im3_path):
                    os.makedirs(im3_path)
                r_image1.save(os.path.join(im1_path, img_name), quality=95, subsampling=0)
                r_image2.save(os.path.join(im2_path, img_name), quality=95, subsampling=0)
                r_image3.save(os.path.join(im3_path, img_name), quality=95, subsampling=0)
        print(all_t / count)
    elif mode == "fps":
        img1 = Image.open(fps_path1)
        img2 = Image.open(fps_path2)
        tact_time = yolo.get_FPS(img1, img2, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
    elif mode == "heatmap":
        if not os.path.exists(heatmap_save_path):
            os.makedirs(heatmap_save_path, exist_ok=True)
        while True:
            root = '/home/WQL/datasets/0228AOI/test/'
            name = "AA_3_13.jpg"
            heatmap_save_path1 = heatmap_save_path + "im1_ex" + name
            heatmap_save_path2 = heatmap_save_path + "im2_change" + name
            img1 = root + "im1/" + name
            img2 = root + "im2/" + name
            try:
                image1 = Image.open(img1)
                image2 = Image.open(img2)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image2, image1, heatmap_save_path1, heatmap_save_path2)
                break


    elif mode == "export_onnx":
        yolo.convert_to_onnx(simplify, onnx_save_path)

    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError(
            "Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
