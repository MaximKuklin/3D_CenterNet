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
from scipy.spatial.transform import Rotation as R
from models.decode import project_points
from Objectron.objectron.dataset.box import Box
import math
import matplotlib.pyplot as plt


DEBUG = True

class Dataset3D(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        video_info = self.coco.loadImgs(ids=[img_id])[0]
        file_name = video_info['file_name']
        frame = video_info['frame']
        video_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        # get image
        cap = cv2.VideoCapture(video_path)
        cap.set(1, frame)
        _, img = cap.read()

        # get image shape and center
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)

        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.opt.input_h, self.opt.input_w


        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio


        # trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        # empty input
        heat_map = np.zeros([self.num_classes, output_h, output_w], dtype=np.float32)
        # regression = np.zeros([self.max_objs, 3, 8], dtype=np.float32)
        # cls_ids = np.zeros([self.max_objs], dtype=np.int32)
        # proj_points = np.zeros([self.max_objs, 2], dtype=np.int32)
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
            rot_quat = np.array(ann['rot_quat'])
            rot_angles = R.from_quat(rot_quat).as_euler('zyx') * 180/math.pi
            translation = np.array(ann['translation'])
            keypoints_2d = np.array(ann['keypoints_2d'])

            ct = keypoints_2d[0][:2]
            ct[0], ct[1] = ct[0] * output_h, ct[1] * output_w
            cls_id = int(self.cat_ids[ann['category_id']])

            bbox[[0, 2]] *= output_w
            bbox[[1, 3]] *= output_h

            # bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            # bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)

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
                    plt.scatter(ct_int[0], ct_int[1])

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

            plot_img = cv2.resize(plot_img, (input_w, input_h))

            projection_matrix = np.array(
                [[1.62688887e+00, 0.00000000e+00, 2.46072412e-02, 0.00000000e+00],
                 [0.00000000e+00, 2.16918516e+00, 7.03930855e-04, 0.00000000e+00],
                 [0.00000000e+00, 0.00000000e+00, - 9.99999762e-01, - 9.99999815e-04],
                 [0.00000000e+00, 0.00000000e+00, - 1.00000000e+00, 0.00000000e+00]
                 ])

            for i in range(num_objs):
                self.plot(plot_img*255, ret['loc'][i], ret['dim'][i], ret['rot'][i], projection_matrix, input_h, input_w )

            # plt.imshow(plot_img)
            # plt.show()
            # plt.imshow(heat_map[0])
            # plt.show()


        return ret

    def plot(self, image, loc, dim, rot, proj, input_h, input_w):
        image = image.astype(np.uint8)
        if rot.shape == (4,):
            rot = R.from_quat(rot).as_rotvec()
            # rot[0], rot[2] = 0, 0
            # rot = R.from_rotvec(rot).as_matrix()
        box = Box.from_transformation(rot, loc, dim).vertices
        points_2d = project_points(box, proj).T[:2]
        points_2d[0] *= input_w
        points_2d[1] *= input_h

        points_2d = points_2d.astype(int).T

        lines = (
            [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
            [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
            [1, 2], [3, 4], [5, 6], [7, 8]  # lines along z-axis
        )

        for dot in points_2d:
            cv2.circle(image, (dot[0], dot[1]), 3, (125, 125, 125), -1)
        for ids in lines:
            cv2.line(
                image,
                (points_2d[ids[0]][0], points_2d[ids[0]][1]),
                (points_2d[ids[1]][0], points_2d[ids[1]][1]),
                (125, 125, 125),
                thickness=1
            )
        # for i in range(9):
        #     cv2.line(image, int(points_2d[0][i]), int(points_2d[1][i]), (125, 125, 125), thickness=2)

        cv2.imshow('a', image)
        cv2.waitKey(0)