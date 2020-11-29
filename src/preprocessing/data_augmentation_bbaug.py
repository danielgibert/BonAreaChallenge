from bbaug import policies
import sys
import os
sys.path.append("../../")
from src.preprocessing.utils import parse_annot
from src.preprocessing.utils import draw_PIL_image
from src.preprocessing.utils import normal_to_yolo
from matplotlib import image
from numpy import asarray
import argparse


def augment_image(image_filepath, annotations_filepath, policy_container):
    random_policy = policy_container.select_random_policy()
    normal_image = asarray(image.imread(image_filepath))
    objects = parse_annot(annotations_filepath)
    img_aug, bbs_aug = policy_container.apply_augmentation(random_policy, normal_image, objects['boxes'], objects['labels'])
    return img_aug, bbs_aug


def generate_augmented_img(img_filepath, annotation_filepath, policy_version, output_img_filepath, output_annotations_filepath, display=True):
    if policy_version == 0:
        aug_policy = policies\
            .policies_v0()
    elif policy_version == 1:
        aug_policy = policies.policies_v1()
    elif policy_version == 2:
        aug_policy = policies.policies_v2()
    else:
        aug_policy = policies.policies_v3()
    policy_container = policies.PolicyContainer(aug_policy)

    img_aug, bbs_aug = augment_image(img_filepath, annotation_filepath, policy_container)
    if display:
        bbs = []
        labels = []
        for bbx in bbs_aug:
            labels.append(bbx[0])
            bbs.append([bbx[1], bbx[2], bbx[3], bbx[4]])
        draw_PIL_image(img_aug, bbs, labels)
    image.imsave(output_img_filepath, img_aug)

    with open(output_annotations_filepath, "w") as output_annotations_file:
        for box_aug in bbs_aug:
            box = normal_to_yolo(box_aug)
            output_annotations_file.write("{} {} {} {} {}\n".format(box[0],
                                                                    box[1],
                                                                    box[2],
                                                                    box[3],
                                                                    box[4]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data augmentation script')
    parser.add_argument("img_filepath",
                        type=str,
                        help="Image filepath")
    parser.add_argument("boxes_filepath",
                        type=str,
                        help="Annotations filepath")
    parser.add_argument("policy_version",
                        type=int,
                        help="Policy version")
    parser.add_argument("output_img_filepath",
                        type=str,
                        help="Augmented image filepath")
    parser.add_argument("output_boxes_filepath",
                        type=str,
                        help="Augmented boxes filepath")

    args = parser.parse_args()

    generate_augmented_img(args.img_filepath,
                           args.boxes_filepath,
                           args.policy_version,
                           args.output_img_filepath,
                           args.output_boxes_filepath)



