from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import tqdm

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
import Objectron.objectron.dataset.box as box
from src.tools.mean_average_precision_objectron import MetricBuilderObjectron
from datasets.dataset_factory import get_dataset
from src.mAP_objectron import PrefetchDataset

from .base_trainer import BaseTrainer
from detectors.detector_factory import detector_factory
from datasets.dataset_factory import dataset_factory

class Det3DLoss(torch.nn.Module):
  def __init__(self, opt):
    super(Det3DLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, off_loss, dim_loss, rot_loss, loc_loss = 0, 0, 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])
      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.dim_weight > 0:
          dim_loss += self.crit_reg(
            output['dim'], batch['reg_mask'],
            batch['ind'], batch['dim']) / opt.num_stacks
      if opt.loc_weight > 0:
          loc_loss += self.crit_reg(
            output['loc'], batch['reg_mask'],
            batch['ind'], batch['loc']) / opt.num_stacks
      if opt.rot_weight > 0:
          rot_loss += self.crit_reg(
            output['rot'], batch['reg_mask'],
            batch['ind'], batch['rot']) / opt.num_stacks


      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
        
    loss = opt.hm_weight * hm_loss \
           + opt.dim_weight * dim_loss + opt.loc_weight * loc_loss \
           + opt.rot_weight * rot_loss

    if opt.reg_offset:
      loss += opt.off_weight * off_loss

    loss_stats = {
      'loss': loss,
      'hm_loss': hm_loss,
      'dim_loss': dim_loss,
      'rot_loss': rot_loss,
      'loc_loss': loc_loss,
    }
    if opt.reg_offset:
      loss_stats.update({'off_loss': off_loss})

    return loss, loss_stats

class Det3DTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(Det3DTrainer, self).__init__(opt, model, optimizer=optimizer)
    self.detector = detector_factory[opt.task](opt)
    self.dataset = dataset_factory[opt.dataset](opt, 'test')
    self.data_loader = torch.utils.data.DataLoader(
        PrefetchDataset(opt, self.dataset, self.detector.pre_process),
        batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )

  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'dim_loss', 'rot_loss', 'loc_loss']
    loss = Det3DLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

  def val(self, epoch, _):
    frames = []

    for idx, (img_id, pre_processed_images, boxes_gt) in enumerate(tqdm.tqdm(self.data_loader)):
        ret = self.detector.run(pre_processed_images)
        boxes_3d = [ret['results'][i][:, 27:-2] for i in ret['results']][0]
        probs = [ret['results'][i][:, -2] for i in ret['results']][0]
        pred_classes = [ret['results'][i][:, -1] for i in ret['results']][0]
        box_pred = [box.Box(vertices=box_pred.reshape(-1, 3)) for box_pred in boxes_3d]
        boxes_gt = [box.Box(vertices=box_gt) for box_gt in boxes_gt[0].numpy()]
        if len(boxes_gt) == 0 or len(box_pred) == 0:
            print()
        frames.append([box_pred, pred_classes, probs, boxes_gt, np.zeros((len(boxes_gt)))])

    preds, gts = [], []

    for frame in frames:
      # [3d_box_gt, class_id, confidence]
      preds.append(np.array((frame[0], frame[1], frame[2])))

      # [3d_box_gt, class_id, difficult, crowd]
      gts.append(np.array((frame[3], frame[4], frame[4], frame[4])))

    metric_fn = MetricBuilderObjectron.build_evaluation_metric("map_3d", async_mode=False, num_classes=1)

    for pred, gt in zip(preds, gts):
        metric_fn.add(pred.T, gt.T)

    mAP = metric_fn.value(
      iou_thresholds=np.arange(0.5, 1.0, 0.05),
      recall_thresholds=np.arange(0., 1.01, 0.01),
      mpolicy='soft'
    )

    mAP_05 = mAP[0.5][0]['ap']
    mAP_05_095 = mAP['mAP']

    self.tb_logger.add_scalar(f"val/mAP 50", mAP_05, epoch)
    self.tb_logger.add_scalar(f"val/mAP 50:95", mAP_05_095, epoch)

    return mAP_05, mAP_05_095
