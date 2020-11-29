from PIL import Image, ImageDraw #version 6.1.0
import PIL #version 1.2.0
import torch
import torchvision.transforms.functional as F
import numpy as np
import random
import csv
import math

voc_labels = ['people', 'empty_box']
label_map = {k: v for v, k in enumerate(voc_labels)}
rev_label_map = {v: k for k, v in label_map.items()}

CLASSES = len(voc_labels)
distinct_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                   for i in range(CLASSES)]
label_color_map  = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def yolo_to_normal(box):
    xmin = box[0] - (box[2]/2)
    ymin = box[1] - (box[3]/2)
    xmax = box[0] + (box[2]/2)
    ymax = box[1] + (box[3]/2)
    return [xmin, ymin, xmax, ymax]

def normal_to_yolo(box, WIDTH=1920, HEIGHT=1080):
    print(box)
    label = box[0]
    xmin = box[1]
    ymin = box[2]
    xmax = box[3]
    ymax = box[4]
    width_difference = math.fabs(xmin - xmax)
    height_difference = math.fabs(ymin - ymax)

    xcenter = float((xmin+(width_difference/2))/WIDTH)
    ycenter = float((ymin+(height_difference/2))/HEIGHT)

    return [label, xcenter, ycenter, float(width_difference/WIDTH), float(height_difference/HEIGHT)]

def parse_annot(annotation_path, WIDTH=1920, HEIGHT=1080):
    boxes = list()
    labels = list()
    difficulties = list()
    with open(annotation_path, "r") as annotation_file:
        reader = csv.reader(annotation_file, delimiter=" ")
        for row in reader:
            label = int(row[0])
            xcenter = int(float(row[1])*WIDTH)
            ycenter = int(float(row[2])*HEIGHT)
            b_width = int(float(row[3])*WIDTH)
            b_height = int(float(row[4])*HEIGHT)

            boxes.append(yolo_to_normal([xcenter, ycenter, b_width, b_height]))
            labels.append(label)
            difficulties.append(0)
    return {"boxes": boxes, "labels": labels, "difficulties": difficulties}

def draw_PIL_image(image, boxes, labels):
    '''
        Draw PIL image
        image: A PIL image
        labels: A tensor of dimensions (#objects,)
        boxes: A tensor of dimensions (#objects, 4)
    '''
    if type(image) != PIL.Image.Image:
        image = F.to_pil_image(image)
    new_image = image.copy()
    #labels = labels.tolist()
    draw = ImageDraw.Draw(new_image)
    #boxes = boxes.tolist()
    for i in range(len(boxes)):
        print(boxes[i])
        draw.rectangle(xy=boxes[i], outline=label_color_map[rev_label_map[labels[i]]])
    new_image.show()