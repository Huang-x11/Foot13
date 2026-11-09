"""Model-independent Foot13 metric for MMPose validation and testing."""
from mmengine.evaluator import BaseMetric

from mmpose.registry import METRICS
from .obb_errors import OBBErrorReference


@METRICS.register_module()
class Foot13OBBMetric(BaseMetric):
    """Report mean pixel distance and OBB-normalized error alongside COCO AP.
    """

    default_prefix = 'foot13_obb'

    def __init__(self, ann_file, geometry_file, collect_device='cpu', prefix=None,
                 require_full_coverage=True):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.reference = OBBErrorReference(ann_file, geometry_file)
        self.require_full_coverage = require_full_coverage

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            self.results.append(self.reference.measure(
                image_id=sample['img_id'],
                annotation_id=sample.get('id'),
                coordinates=sample['pred_instances']['keypoints']))

    def compute_metrics(self, results):
        summary = self.reference.summarize(results, self.require_full_coverage)
        metrics = dict(summary['overall'])
        for point in summary['per_keypoint']:
            if point['count']:
                metrics[f"{point['keypoint']}/MPE_px"] = point['MPE_px']
                metrics[f"{point['keypoint']}/NME"] = point['NME']
        return metrics
