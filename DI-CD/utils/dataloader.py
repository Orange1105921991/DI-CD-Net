import random
from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils1 import cvtColor, preprocess_input


# -------------------------------------------------#
#   build a datasets loader
# -------------------------------------------------#
class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, Root, epoch_length, \
                 mosaic, exchange, exchange_prob, mixup, mosaic_prob, mixup_prob, train, hsv_aug=False,
                 special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.root = Root
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.hsv_aug = hsv_aug
        self.exchange = exchange
        self.exchange_prob = exchange_prob

        self.epoch_now = -1
        self.length = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image_A, image_B, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            # if self.mixup and self.rand() < self.mixup_prob:
            #     lines           = sample(self.annotation_lines, 1)
            #     image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
            #     image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)

        else:
            image_A, image_B, box = self.get_random_data(self.annotation_lines[index], self.input_shape,
                                                         Random=self.train)
        image_A = np.transpose(preprocess_input(np.array(image_A, dtype=np.float32)), (2, 0, 1))
        image_B = np.transpose(preprocess_input(np.array(image_B, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image_A, image_B, box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def apply_hsv_augmentation(self, image_data, hue_factor, sat_factor, val_factor):
        r = np.random.uniform(-1, 1, 3) * [hue_factor, sat_factor, val_factor] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data

    def get_random_data(self, annotation_line, input_shape, jitter=0, hue=.1, sat=0.7, val=0.4, Random=True):
        line = annotation_line.split()

        line_A = self.root + 'im1/' + line[0]
        line_B = self.root + 'im2/' + line[0]
        if self.exchange:
            if random.random() < self.exchange_prob:
                line_A, line_B = line_B, line_A
        image_A = Image.open(line_A)
        image_B = Image.open(line_B)
        image_A = cvtColor(image_A)
        image_B = cvtColor(image_B)

        iw, ih = image_A.size
        h, w = input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not Random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image_A = image_A.resize((nw, nh), Image.BICUBIC)
            image_B = image_B.resize((nw, nh), Image.BICUBIC)
            new_image_A = Image.new('RGB', (w, h), (128, 128, 128))
            new_image_B = Image.new('RGB', (w, h), (128, 128, 128))
            new_image_A.paste(image_A, (dx, dy))
            new_image_B.paste(image_B, (dx, dy))
            image_data_A = np.array(new_image_A, np.float32)
            image_data_B = np.array(new_image_B, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data_A, image_data_B, box

        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = 1
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image_A = image_A.resize((nw, nh), Image.BICUBIC)
        image_B = image_B.resize((nw, nh), Image.BICUBIC)

        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image_A = Image.new('RGB', (w, h), (128, 128, 128))
        new_image_B = Image.new('RGB', (w, h), (128, 128, 128))
        new_image_A.paste(image_A, (dx, dy))
        new_image_B.paste(image_B, (dx, dy))
        image_A = new_image_A
        image_B = new_image_B

        flip = self.rand() < .5
        # rotate = np.random.choice([0, 90, 180, 270])  # 随机选择旋转角度
        if flip:
            image_A = image_A.transpose(Image.FLIP_LEFT_RIGHT)
            image_B = image_B.transpose(Image.FLIP_LEFT_RIGHT)
            # box[:, [0,2]] = w - box[:, [2,0]]

        image_data_A = np.array(image_A, np.uint8)
        image_data_B = np.array(image_B, np.uint8)

        if self.hsv_aug:
            image_data_A = self.apply_hsv_augmentation(image_data_A, hue, sat, val)
            image_data_B = self.apply_hsv_augmentation(image_data_B, hue, sat, val)

        '''
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
 '''
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data_A, image_data_B, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0., hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        scale = 0.5

        image_datasA = []
        image_datasB = []
        box_datas = []
        index = 0

        for line in annotation_line:
            line_content = line.split()
            line_content_A = self.root + 'im1/' + line_content[0]
            line_content_B = self.root + 'im2/' + line_content[0]

            if self.exchange:
                if random.random() < self.exchange_prob:
                    line_content_A, line_content_B = line_content_B, line_content_A

            image_A = Image.open(line_content_A)
            image_B = Image.open(line_content_B)
            image_A = cvtColor(image_A)
            image_B = cvtColor(image_B)

            iw, ih = image_A.size
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image_A = image_A.transpose(Image.FLIP_LEFT_RIGHT)
                image_B = image_B.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            nw, nh = int(iw * scale), int(ih * scale)
            image_A = image_A.resize((nw, nh), Image.BICUBIC)
            image_B = image_B.resize((nw, nh), Image.BICUBIC)

            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih

            if index == 0:
                dx, dy = 0, 0
            elif index == 1:
                dx, dy = w // 2, 0
            elif index == 2:
                dx, dy = 0, h // 2
            elif index == 3:
                dx, dy = w // 2, h // 2

            new_imageA = Image.new('RGB', (w, h), (128, 128, 128))
            new_imageA.paste(image_A, (dx, dy))
            new_imageB = Image.new('RGB', (w, h), (128, 128, 128))
            new_imageB.paste(image_B, (dx, dy))

            image_dataA = np.array(new_imageA)
            image_dataB = np.array(new_imageB)

            index += 1
            box_data = []
            if len(box) > 0:
                box[:, [0, 2]] += dx
                box[:, [1, 3]] += dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datasA.append(image_dataA)
            image_datasB.append(image_dataB)
            box_datas.append(box_data)

        new_imageA = np.zeros([h, w, 3])
        new_imageA[:h // 2, :w // 2, :] = image_datasA[0][:h // 2, :w // 2, :]
        new_imageA[h // 2:, :w // 2, :] = image_datasA[2][h // 2:, :w // 2, :]
        new_imageA[h // 2:, w // 2:, :] = image_datasA[3][h // 2:, w // 2:, :]
        new_imageA[:h // 2, w // 2:, :] = image_datasA[1][:h // 2, w // 2:, :]

        new_imageB = np.zeros([h, w, 3])
        new_imageB[:h // 2, :w // 2, :] = image_datasB[0][:h // 2, :w // 2, :]
        new_imageB[h // 2:, :w // 2, :] = image_datasB[2][h // 2:, :w // 2, :]
        new_imageB[h // 2:, w // 2:, :] = image_datasB[3][h // 2:, w // 2:, :]
        new_imageB[:h // 2, w // 2:, :] = image_datasB[1][:h // 2, w // 2:, :]

        new_imageA = np.array(new_imageA, np.uint8)
        new_imageB = np.array(new_imageB, np.uint8)
        if self.hsv_aug:
            new_imageA = self.apply_hsv_augmentation(new_imageA, hue, sat, val)
            new_imageB = self.apply_hsv_augmentation(new_imageB, hue, sat, val)

        new_boxes = self.merge_bboxes(box_datas, w // 2, h // 2)

        return new_imageA, new_imageB, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes

def yolo_dataset_collate(batch):
    images_A = []
    images_B = []
    bboxes = []
    for img_A, img_B, box in batch:
        images_A.append(img_A)
        images_B.append(img_B)
        bboxes.append(box)
    images_A = torch.from_numpy(np.array(images_A)).type(torch.FloatTensor)
    images_B = torch.from_numpy(np.array(images_B)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images_A, images_B, bboxes
