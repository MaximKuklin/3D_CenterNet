import pandas as pd
import numpy as np
from mean_average_precision import MetricBuilder, MeanAveragePrecision2d
from mean_average_precision.adapter import AdapterDefault
import Objectron.objectron.dataset.iou as iou


class MetricBuilderObjectron(MetricBuilder):
    def __init__(self):
        super(MetricBuilderObjectron, self).__init__()

    @staticmethod
    def get_metrics_list():
        """ Get evaluation metrics list."""
        return list(metrics_dict.keys())

    @staticmethod
    def build_evaluation_metric(metric_type, async_mode=False, adapter_type=AdapterDefault, *args, **kwargs):
        """ Build evaluation metric.

        Arguments:
            metric_type (str): type of evaluation metric.
            async_mode (bool): use multiprocessing metric.
            adapter_type (AdapterBase): type of adapter class.

        Returns:
            metric_fn (MetricBase): instance of the evaluation metric.
        """
        assert metric_type in metrics_dict, "Unknown metric_type"
        if not async_mode:
            metric_fn = metrics_dict[metric_type](*args, **kwargs)
        else:
            metric_fn = MetricMultiprocessing(metrics_dict[metric_type], *args, **kwargs)
        return adapter_type(metric_fn)


class MeanAveragePrecision3d(MeanAveragePrecision2d):
    def __init__(self, num_classes):
        super(MeanAveragePrecision3d, self).__init__(num_classes=num_classes)

    def add(self, preds, gt):
        """ Add sample to evaluation.

        Arguments:
            preds (np.array): predicted boxes.
            gt (np.array): ground truth boxes.

        Input format:
            preds: [xmin, ymin, xmax, ymax, class_id, confidence]
            gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        """
        assert preds.ndim == 2 and preds.shape[1] == 3
        assert gt.ndim == 2 and gt.shape[1] == 4
        class_counter = np.zeros((1, self.num_classes), dtype=np.int32)
        for c in range(self.num_classes):
            gt_c = gt[gt[:, -2] == c]
            class_counter[0, c] = gt_c.shape[0]
            preds_c = preds[preds[:, -2] == c]
            if preds_c.shape[0] > 0:
                match_table = compute_match_table(preds_c, gt_c, self.imgs_counter)
                self.match_table[c] = self.match_table[c].append(match_table)
        self.imgs_counter = self.imgs_counter + 1
        self.class_counter = np.concatenate((self.class_counter, class_counter), axis=0)


metrics_dict = {
    'map_2d': MeanAveragePrecision2d,
    'map_3d': MeanAveragePrecision3d,
}


def compute_match_table(preds, gt, img_id):
    """ Compute match table.

    Arguments:
        preds (np.array): predicted boxes.
        gt (np.array): ground truth boxes.
        img_id (int): image id

    Returns:
        match_table (pd.DataFrame)


    Input format:
        preds: [xmin, ymin, xmax, ymax, class_id, confidence]
        gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]

    Output format:
        match_table: [img_id, confidence, iou, difficult, crowd]
    """
    def _tile(arr, nreps, axis=0):
        return np.repeat(arr, nreps, axis=axis).reshape(nreps, -1).tolist()

    def _empty_array_2d(size):
        return [[] for i in range(size)]

    match_table = {}
    match_table["img_id"] = [img_id for i in range(preds.shape[0])]
    match_table["confidence"] = preds[:, -1].tolist()
    if gt.shape[0] > 0:
        match_table["iou"] = compute_iou(preds[:, 0], gt[:, 0]).tolist()
        match_table["difficult"] = _tile(gt[:, -2], preds.shape[0], axis=0)
        match_table["crowd"] = _tile(gt[:, -1], preds.shape[0], axis=0)
    else:
        match_table["iou"] = _empty_array_2d(preds.shape[0])
        match_table["difficult"] = _empty_array_2d(preds.shape[0])
        match_table["crowd"] = _empty_array_2d(preds.shape[0])
    return pd.DataFrame(match_table, columns=list(match_table.keys()))


def compute_iou(prediction, gt):
    IoU = np.zeros((len(prediction), len(gt)))
    for i in range(len(prediction)):
        for j in range(len(gt)):
            try:
                IoU[i, j] = iou.IoU(prediction[i], gt[j]).iou()
            except:
                IoU[i, j] = 0.0

    return IoU
