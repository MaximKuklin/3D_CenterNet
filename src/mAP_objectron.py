from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mean_average_precision as map
import tqdm

import _init_paths

import os
import cv2
import torch
import numpy as np
import math
import matplotlib.pyplot as plt

from opts import opts
from utils.image import get_affine_transform
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import dataset_factory
from Objectron.objectron.dataset import iou
from Objectron.objectron.dataset import box
# from mean_average_precision.detection_map import DetectionMAP


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


# class Detection3DMAP(DetectionMAP):
#     def __init__(self, n_class, pr_samples=11, overlap_threshold=0.5):
#         super().__init__(n_class, pr_samples=pr_samples, overlap_threshold=overlap_threshold)
#
#     def evaluate(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes):
#         """
#         Update the accumulator for the running mAP evaluation.
#         For exemple, this can be called for each images
#         :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, x2, y2] :     Shape [n_pred, 4]
#         :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
#         :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
#         :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, x2, y2] :  Shape [n_gt, 4]
#         :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
#         :return:
#         """
#
#         IoUmask = None
#         if len(pred_bb) > 0:
#             IoUmask = self.compute_IoU_mask(pred_bb, gt_bb, self.overlap_threshold)
#         for accumulators, r in zip(self.total_accumulators, self.pr_scale):
#             self.evaluate_(IoUmask, accumulators, pred_classes, pred_conf, gt_classes, r)
#
#     def compute_IoU_mask(self, prediction, gt, overlap_threshold):
#         IoU = np.zeros((len(prediction), len(gt)))
#         for i in range(len(prediction)):
#             for j in range(len(gt)):
#                 IoU[i, j] = iou.IoU(prediction[i], gt[j]).iou()
#
#         # for each prediction select gt with the largest IoU and ignore the others
#         for i in range(len(prediction)):
#             maxj = IoU[i, :].argmax()
#             IoU[i, :maxj] = 0
#             IoU[i, (maxj + 1):] = 0
#         # make a mask of all "matched" predictions vs gt
#         return IoU >= overlap_threshold
#
#     def get_map(self, interpolated=True, class_names=None):
#         """
#         Plot all pr-curves for each classes
#         :param interpolated: will compute the interpolated curve
#         :return:
#         """
#
#         mean_average_precision = []
#         # TODO: data structure not optimal for this operation...
#         for cls in range(self.n_class):
#             precisions, recalls = self.compute_precision_recall_(cls, interpolated)
#             average_precision = self.compute_ap(precisions, recalls)
#             mean_average_precision.append(average_precision)
#
#         mean_average_precision = sum(mean_average_precision) / len(mean_average_precision)
#         return mean_average_precision


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.images = dataset.images
        self.coco = dataset.coco
        self.load_image_func = dataset.coco.loadImgs
        self.img_dir = dataset.img_dir
        self.pre_process_func = pre_process_func
        self.mean, self.std = dataset.mean, dataset.std
        self.opt = opt

    def grab_frame(self, video_path, frame):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, img = cap.read()
        cap.release()
        return img

    def __getitem__(self, index):
        img_id = self.images[index]
        video_info = self.load_image_func(ids=[img_id])[0]
        file_name = video_info['file_name']

        image_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        gt_3d_box = []
        for k in anns:
            bbox = np.array(k['keypoints_3d']).reshape(-1, 3)
            gt_3d_box.append(bbox)
        gt_3d_box = np.stack(gt_3d_box)

        img = cv2.imread(image_path)
        images, meta = {}, {}

        for scale in [1.0]:
           images[scale], meta[scale] = self.pre_process_func(img, scale)
        return img_id, {'images': images, 'image': img, 'meta': meta}, gt_3d_box

    def __len__(self):
        return len(self.images)


def calc_metric(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    torch.cuda.set_device(int(opt.gpus_str))

    split = 'test'

    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    Dataset = dataset_factory[opt.dataset]
    dataset = Dataset(opt, split)

    data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, dataset, detector.pre_process),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )

    frames = []

    for idx, (img_id, pre_processed_images, boxes_gt) in enumerate(tqdm.tqdm(data_loader)):

        ret = detector.run(pre_processed_images)
        boxes_3d = [ret['results'][i][:, 27:-2] for i in ret['results']][0]
        probs = [ret['results'][i][:, -2] for i in ret['results']][0]
        pred_classes = [ret['results'][i][:, -1] for i in ret['results']][0]
        box_pred = [box.Box(vertices=box_pred.reshape(-1, 3)) for box_pred in boxes_3d]
        boxes_gt = [box.Box(vertices=box_gt) for box_gt in boxes_gt[0].numpy()]
        if len(boxes_gt) == 0 or len(box_pred) == 0:
            print()
        frames.append([box_pred, pred_classes, probs, boxes_gt, np.zeros((len(boxes_gt)))])

    n_class = 1

    mAP = Detection3DMAP(n_class, overlap_threshold=0.5)
    for frame in frames:
        mAP.evaluate(*frame)

    mAP_score = mAP.get_map()
    print(f"mAP_score: {mAP_score}")

if __name__ == '__main__':
    opt = opts().init()
    calc_metric(opt)
