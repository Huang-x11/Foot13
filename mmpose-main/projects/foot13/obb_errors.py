"""Shared Foot13 errors in native crop pixels, independent of model codec.

"""
import json
from pathlib import Path

import numpy as np


KEYPOINT_NAMES = ('FM', 'FL', 'MM', 'ML', 'HM', 'HL', 'HaMP', 'HeMP',
                  'FMMP', 'HR', 'STF', 'NT', 'DIM')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _points(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    points = np.asarray(value, dtype=np.float64)
    if points.shape == (1, 13, 2):
        points = points[0]
    if points.shape != (13, 2) or not np.isfinite(points).all():
        raise ValueError('Expected one finite 13 x 2 native-crop prediction')
    return points


class OBBErrorReference:
    """Read COCO landmarks and match OBB diagonals by image ID."""

    def __init__(self, ann_file, geometry_file):
        self.ann_file = str(Path(ann_file).resolve())
        self.geometry_file = str(Path(geometry_file).resolve())
        coco = _read_json(ann_file)
        geometry = _read_json(geometry_file)
        categories = coco['categories']
        if len(categories) != 1 or tuple(categories[0]['keypoints']) != KEYPOINT_NAMES:
            raise ValueError('Expected the canonical Foot13 keypoint order')

        normalizers = {}
        for item in geometry['annotations']:
            image_id = int(item['image_id'])
            if image_id in normalizers:
                raise ValueError(f'Duplicate OBB normalizer for image_id={image_id}')
            normalizers[image_id] = float(item['error_normalizer_px'])

        images = {int(item['id']): item for item in coco['images']}
        if len(images) != len(coco['images']):
            raise ValueError('Duplicate COCO image IDs')
        self.by_image = {}
        annotation_ids = set()
        for ann in coco['annotations']:
            image_id, ann_id = int(ann['image_id']), int(ann['id'])
            if image_id in self.by_image or ann_id in annotation_ids:
                raise ValueError('Expected one uniquely identified foot per image')
            if ann['category_id'] != categories[0]['id'] or ann.get('iscrowd', 0):
                raise ValueError(f'Unsupported category or crowd annotation: {ann_id}')
            normalizer = normalizers.get(image_id)
            if normalizer is None or not np.isfinite(normalizer) or normalizer <= 0:
                raise ValueError(f'Missing or invalid OBB diagonal for image_id={image_id}')
            triples = np.asarray(ann['keypoints'], dtype=np.float64).reshape(13, 3)
            mask = triples[:, 2] > 0
            if not mask.any():
                raise ValueError(f'No labelled keypoints: {image_id}')
            self.by_image[image_id] = dict(
                image_id=image_id, annotation_id=ann_id,
                file_name=images[image_id]['file_name'],
                gt=triples[:, :2], mask=mask, normalizer=normalizer)
            annotation_ids.add(ann_id)
        if not self.by_image or set(self.by_image) != set(images):
            raise ValueError('Expected nonempty COCO with one annotation per image')

    def measure(self, image_id, coordinates, annotation_id=None):
        image_id = int(image_id)
        if image_id not in self.by_image:
            raise ValueError(f'Prediction image {image_id} is outside this split')
        ref = self.by_image[image_id]
        if annotation_id is not None and int(annotation_id) != ref['annotation_id']:
            raise ValueError(f'Prediction annotation/image mismatch: {image_id}')
        points = _points(coordinates)
        distances = np.linalg.norm(points - ref['gt'], axis=1)
        return dict(image_id=image_id, annotation_id=ref['annotation_id'],
                    prediction=points.copy(), errors_px=distances,
                    errors_normalized=distances / ref['normalizer'],
                    mask=ref['mask'].copy())

    def summarize(self, results, require_full_coverage=True):
        if not results:
            raise ValueError('No predictions to evaluate')
        image_ids = [item['image_id'] for item in results]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError('Duplicate prediction image IDs')
        unknown = set(image_ids) - self.by_image.keys()
        missing = self.by_image.keys() - set(image_ids)
        if unknown or (require_full_coverage and missing):
            raise ValueError(f'Prediction coverage mismatch: unknown={len(unknown)}, missing={len(missing)}')
        errors = np.stack([item['errors_px'] for item in results])
        normalized = np.stack([item['errors_normalized'] for item in results])
        mask = np.stack([item['mask'] for item in results])
        overall = dict(images=len(results), labelled_points=int(mask.sum()),
                       MPE_px=float(errors[mask].mean()),
                       NME=float(normalized[mask].mean()),
                       NME_percent=float(normalized[mask].mean() * 100))
        per_keypoint = []
        for i, name in enumerate(KEYPOINT_NAMES):
            valid = mask[:, i]
            count = int(valid.sum())
            per_keypoint.append(dict(keypoint=name, count=count,
                                     MPE_px=float(errors[valid, i].mean()) if count else None,
                                     NME=float(normalized[valid, i].mean()) if count else None))
        return dict(overall=overall, per_keypoint=per_keypoint,
                    complete_split=not missing,
                    aggregation='mean over all GT-labelled keypoints; per-point normalization before averaging',
                    coordinate_space='native_crop_pixels',
                    normalizer='continuous expanded OBB diagonal before integer rounding')
