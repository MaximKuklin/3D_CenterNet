import argparse
import os
from os.path import join

import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import albumentations as A
from PIL import Image

from Objectron.objectron.dataset.box import Box
from Objectron.objectron.schema import annotation_data_pb2 as annotation_protocol

CLASSES = ['shoe']

TEST = False
FRAME_RATE = 30  # get only every n-th frame from video


def convert(root_obj, image_dir, index_objectron='Objectron/index', mode='train'):
    annotation_files = [join(index_objectron, "_".join((cls, 'annotations', mode))) for cls in CLASSES]

    anno_to_video = parse_index(annotation_files, root_obj)

    info = {
        "description": "Objectron 2020 Dataset",
        "url": "https://github.com/google-research-datasets/Objectron",
        "version": "1.0",
        "year": 2020,
        "contributor": "Google",
        "date_created": "2021/03/13"
        }
    licences = [{
        "url": "",
        "id": 1,
        "name": ""
    }]

    categories = [{
        "supercategory": "shoe",
        "id": 1,
        "name": "shoe"
    }]

    image_info, annotations = get_video_annotation(anno_to_video, image_dir)

    result = {
        "info": info,
        "licences": licences,
        "images": image_info,
        "annotations": annotations,
        "categories": categories,
    }

    return result


def get_frame_annotation(sequence, frame_id, image_id, anno_counter, image=None):

    annotations_list = []
    additional = {}

    data = sequence.frame_annotations[frame_id]
    projection_matrix = np.array(data.camera.projection_matrix).reshape(4, 4).tolist()
    additional['projection_matrix'] = projection_matrix

    object_keypoints_2d = []
    object_keypoints_3d = []
    obj_num = len(data.annotations)

    for obj_idx in range(obj_num):
        annotations = data.annotations[obj_idx]
        num_keypoints = len(annotations.keypoints)

        for keypoint_id in range(num_keypoints):
            keypoint = annotations.keypoints[keypoint_id]
            # pixel coords
            object_keypoints_2d.append(
                (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
            # 3d object keypoints in **camera** coords
            object_keypoints_3d.append(
                (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))

    object_keypoints_2d = np.array(object_keypoints_2d).reshape((-1, num_keypoints, 3))
    object_keypoints_3d = np.array(object_keypoints_3d).reshape((-1, num_keypoints, 3))

    for i in range(obj_num):
        box = Box(vertices=object_keypoints_3d[i])
        scale = box.scale.tolist()
        translation = box.translation.tolist()
        rot_mat = box.rotation
        rot_euler = R.from_matrix(rot_mat).as_euler('zyx').tolist()
        bbox = np.array([
                min(object_keypoints_2d[i].T[:2][0]), min(object_keypoints_2d[i].T[:2][1]),
                max(object_keypoints_2d[i].T[:2][0]), max(object_keypoints_2d[i].T[:2][1])
            ]).tolist()

        if TEST:
            box = Box().from_transformation(
                R.from_euler('zyx', rot_euler).as_matrix(),
                np.array(translation),
                np.array(scale)
            ).vertices

            assert np.allclose(box, object_keypoints_3d[i], atol=1e-3)

            points_2d = project_points(box, np.array(projection_matrix))

            assert np.allclose(points_2d[:, :2], object_keypoints_2d[i][:, :2], atol=1e-2)


        anno = {
            "segmentation": [[]],
            "area": 0,
            "iscrowd": 0,
            "image_id": image_id,
            "bbox": bbox,
            "keypoints_2d": object_keypoints_2d[i].tolist(),
            "keypoints_3d": object_keypoints_3d[i].tolist(),
            "category_id": 1,
            "id": anno_counter,
            "scale": scale,
            "translation": translation,
            "rot": rot_euler
        }
        annotations_list.append(anno)
        anno_counter += 1
    return annotations_list, anno_counter, additional


def project_points(p_3d_cam, projection_matrix):
    p_3d_cam = np.concatenate((p_3d_cam, np.ones_like(p_3d_cam[:, :1])), axis=-1).T
    p_2d_proj = np.matmul(projection_matrix, p_3d_cam)
    # Project the points
    p_2d_ndc = p_2d_proj[:-1, :] / p_2d_proj[-1, :]
    p_2d_ndc = p_2d_ndc.T

    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    x = p_2d_ndc[:, 1]
    y = p_2d_ndc[:, 0]
    pixels = np.copy(p_2d_ndc)
    pixels[:, 0] = ((1 + x) * 0.5)
    pixels[:, 1] = ((1 + y) * 0.5)
    # pixels = pixels.astype(int)
    return pixels


def get_image_name(video_path, frame, save_folder):
    video_path = video_path.split('/')
    batch = video_path[-3].split('-')[-1]
    video_id = video_path[-2]
    save_path = f"{batch}_{video_id}_{frame}.png"
    save_path = join(save_folder, save_path)
    return save_path


def save_image(img, path, size=640):
    img = A.smallest_max_size(img, size, cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img)
    im.save(path)
    w, h = img.shape[1], img.shape[0]

    return img, (w, h)


def get_video_annotation(anno_video_map, image_dir):
    image_counter = 1
    anno_counter = 1
    images = []
    annotations = []
    for ann, video in tqdm(anno_video_map.items()):
        with open(ann, 'rb') as pb:
            sequence = annotation_protocol.Sequence()
            sequence.ParseFromString(pb.read())
        cap = cv2.VideoCapture(video)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frames):
            if i % FRAME_RATE != 0:
                continue
            cap.set(1, i)
            ret, image = cap.read()
            image_path = get_image_name(video, i, image_dir)
            if not ret:
                print(f"cant get frame {i} from {video}")
                continue

            _, (w, h) = save_image(image, image_path)

            data, anno_counter, aux = get_frame_annotation(sequence, i, image_counter, anno_counter)
            annotations.extend(data)

            filename = os.path.basename(image_path)

            image_info = {
                "license": 1,
                "file_name": filename,
                "coco_url": "",
                "height": int(h),
                "width": int(w),
                "date_captured": "",
                "flickr_url": "",
                "id": image_counter,
                **aux
            }

            images.append(image_info)
            image_counter += 1

    return images, annotations


def parse_index(files, dataset_path):
    anno_video_map = {}

    for file in files:
        with open(file) as f:
            data = [line[:-1] for line in f.readlines()]
        # data = ['shoe/batch-4/44']
        for d in data:
            item_ann = join(dataset_path, d + '.pbdata')
            item_video = join(dataset_path, d, 'video.MOV')
            anno_video_map[item_ann] = item_video
            assert os.path.exists(item_video), f"No video found {item_video}"
            assert os.path.exists(item_ann), f"No annotation found {item_ann}"
        return anno_video_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to Objectron dataset')
    parser.add_argument('-i', '--index', type=str, required=True, help='Path to Objectron index (train test splits)')
    parser.add_argument('-s', '--save', type=str, help='Save dir for coco json')

    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    root_obj = args.dataset
    index_objectron = args.index
    if args.save is not None:
        save = args.save
    else:
        save = join(root_obj, 'coco_converted', 'objectron_cleared_train.json')

    save_dir = os.path.dirname(save)
    save_images_dir = join(root_obj, 'images_train')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_images_dir):
        os.makedirs(save_images_dir)

    coco_converted = convert(root_obj, save_images_dir, index_objectron)

    with open(save, 'w') as f:
        json.dump(coco_converted, f)


if __name__ == '__main__':
    main()