from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import albumentations as A


DEBUG = False

if DEBUG:
    from src.lib.models.decode import project_points
    from Objectron.objectron.dataset.box import Box


class Dataset3D(data.Dataset):
    def __init__(self, opt):
        super(Dataset3D, self).__init__()
        self.opt = opt
        self.augs = A.Compose([
            A.LongestMaxSize(max(self.opt.input_h, self.opt.input_w), always_apply=True),
            A.PadIfNeeded(self.opt.input_h, self.opt.input_w, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
            A.Blur(blur_limit=(4, 8), p=0.1),
            # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.4, 0.2), rotate_limit=0,
            #                    border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), p=0.8),
            A.OneOf([
                A.RandomBrightnessContrast(always_apply=True),
                A.RandomGamma(gamma_limit=(60, 140), always_apply=True),
                # A.CLAHE(always_apply=True)
            ], p=0.3),
            A.OneOf([
                A.RGBShift(),
                A.HueSaturationValue(),
                A.ToGray()
            ], p=0.1)
        ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
        )

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i


    def grab_frame(self, video_path, frame):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
        cap.release()
        return img

    def __getitem__(self, index):
        img_id = self.images[index]
        video_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = video_info['file_name']
        image_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)
        input_h, input_w = self.opt.input_h, self.opt.input_w

        centers = np.array([ann['keypoints_2d'] for ann in anns])[:, 0::9, :2]
        centers = centers.reshape(num_objs, 2)

        img = cv2.imread(image_path)

        # resize, pad, and color augs
        centers[:, 0], centers[:, 1] = centers[:, 0]*img.shape[1], centers[:, 1]*img.shape[0]
        augmented = self.augs(image=img, keypoints=centers)
        inp, centers = augmented['image'], np.array(augmented['keypoints'])
        centers[:, 0], centers[:, 1] = centers[:, 0] / inp.shape[1], centers[:, 1] / inp.shape[0]

        aug = False
        if self.split == 'train' and 0 > self.opt.aug_ddd: # np.random.random() < self.opt.aug_ddd:
            aug = True
            sf = self.opt.scale
        #     # cf = self.opt.shift
            scale_rand = 0.5  # np.random.randn()

            centers[:, 0] -= centers[:, 0] * scale_rand * sf
            centers[:, 1] -= centers[:, 1] * scale_rand * sf

        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio

        # empty input
        heat_map = np.zeros([self.num_classes, output_h, output_w], dtype=np.float32)
        scales = np.zeros([self.max_objs, 3], dtype=np.float32)
        translations = np.zeros([self.max_objs, 3], dtype=np.float32)
        rotvecs = np.zeros([self.max_objs, 3], dtype=np.float32)
        reg_mask = np.zeros([self.max_objs], dtype=np.uint8)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)


        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = np.array(ann['bbox'])
            scale = np.array(ann['scale'])
            rot_angles = np.array(ann['rot'])
            translation = np.array(ann['translation'])

            if aug:
                translation[2] *= np.clip(scale_rand * sf + 1, 1 - sf, 1 + sf)
                # translation[0] += translation[0] * y_shift * cf
                # translation[1] -= (x_shift * cf) * 0.3

            ct = centers[k][:2]

            ct[0], ct[1] = ct[0] * output_h, ct[1] * output_w
            ct[0], ct[1] = np.clip(ct[0], 0, output_w - 1), np.clip(ct[1], 0, output_w - 1)

            cls_id = int(self.cat_ids[ann['category_id']])

            bbox[[0, 2]] *= output_w
            bbox[[1, 3]] *= output_h

            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius/2))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(heat_map[cls_id], ct_int, radius)
                scales[k] = scale
                translations[k] = translation
                rotvecs[k] = rot_angles

                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                if DEBUG:
                    lines = (
                        [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
                        [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
                        [1, 2], [3, 4], [5, 6], [7, 8]  # lines along z-axis
                    )

                    plt.scatter(ct_int[0], ct_int[1])
                    r = R.from_euler('zyx', rot_angles).as_matrix()

                    box_3d = Box.from_transformation(r, translation, scale).vertices
                    points_2d = project_points(box_3d, np.array(video_info['projection_matrix']))
                    points_2d[:, 0] = points_2d[:, 0]*96 + (128-96)/2
                    points_2d[:, 1] *= 128
                    points_2d = points_2d.astype(int)
                    for ids in lines:
                        plt.plot(
                            (points_2d[ids[0]][0], points_2d[ids[1]][0]),
                            (points_2d[ids[0]][1], points_2d[ids[1]][1]),
                            color='r',
                        )

                    # points_2d = np.array(ann['keypoints_2d'])
                    # points_2d[:, 0] *= 128
                    # points_2d[:, 1] *= 128
                    #
                    # points_2d = points_2d.astype(int)
                    # for ids in lines:
                    #     plt.plot(
                    #         (points_2d[ids[0]][0], points_2d[ids[1]][0]),
                    #         (points_2d[ids[0]][1], points_2d[ids[1]][1]),
                    #         color='b',
                    #     )


        ret = {
            'input': inp,
            'hm': heat_map,
            'reg_mask': reg_mask,
            'ind': ind,
            'dim': scales,
            'rot': rotvecs,
            'loc': translations
        }

        if self.opt.reg_offset:
            ret.update({'reg': reg})

        if DEBUG:
            if inp.shape[0] == 3:
                plot_img = inp.transpose(1, 2, 0)
                plot_img = (plot_img * self.std) + self.mean
            else:
                plot_img = inp.copy()

            plot_img = cv2.resize(plot_img, (output_w, output_h))
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
            plt.imshow(plot_img)
            plt.show()
            plt.imshow(heat_map[0])
            plt.show()

        return ret

