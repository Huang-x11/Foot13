"""LiteHRNet-18: upstream recipe adapted to Foot13, input width192/height480.

"""

_base_ = ['../body_2d_keypoint/topdown_heatmap/coco/td-hm_litehrnet-18_8xb64-210e_coco-256x192.py']

data_root = '../data/Foot13/'

dataset_metainfo = {'from_file': 'configs/_base_/datasets/foot13.py'}

work_dir = './work_dirs/foot13_480x192_litehrnet18'

randomness = {'seed': 42}

load_from = None

resume = False

codec = {'type': 'MSRAHeatmap', 'input_size': (192, 480), 'heatmap_size': (48, 120), 'sigma': 2}

model = {'backbone': {'init_cfg': {'type': 'Pretrained',
                           'prefix': 'backbone.',
                           'checkpoint': '../checkpoints/litehrnet18_coco_256x192-6bace359_20211230.pth'}},
 'head': {'out_channels': 13,
          'decoder': {'type': 'MSRAHeatmap',
                      'input_size': (192, 480),
                      'heatmap_size': (48, 120),
                      'sigma': 2}}}

train_cfg = {'val_interval': 1}

train_pipeline = [{'type': 'LoadImage'},
 {'type': 'GetBBoxCenterScale', 'padding': 1.0},
 {'type': 'RandomFlip', 'direction': 'horizontal'},
 {'type': 'RandomBBoxTransform',
  'rotate_factor': 5,
  'scale_factor': (1.0, 1.1),
  'shift_factor': 0.02,
  'shift_prob': 0.3,
  'scale_prob': 1.0,
  'rotate_prob': 0.6},
 {'type': 'TopdownAffine', 'input_size': (192, 480)},
 {'type': 'Foot13PhotometricAug',
  'brightness_limit': 0.1,
  'contrast_limit': 0.1,
  'brightness_contrast_prob': 0.3,
  'blur_kernel': 3,
  'blur_prob': 0.1},
 {'type': 'GenerateTarget',
  'encoder': {'type': 'MSRAHeatmap',
              'input_size': (192, 480),
              'heatmap_size': (48, 120),
              'sigma': 2}},
 {'type': 'PackPoseInputs'}]

val_pipeline = [{'type': 'LoadImage'},
 {'type': 'GetBBoxCenterScale', 'padding': 1.0},
 {'type': 'TopdownAffine', 'input_size': (192, 480)},
 {'type': 'PackPoseInputs'}]

train_dataloader = {'num_workers': 0,
 'persistent_workers': False,
 'dataset': {'_delete_': True,
             'type': 'CocoDataset',
             'data_mode': 'topdown',
             'data_root': '../data/Foot13/',
             'ann_file': 'annotations/foot13_train.json',
             'bbox_file': None,
             'data_prefix': {'img': 'images/'},
             'metainfo': {'from_file': 'configs/_base_/datasets/foot13.py'},
             'test_mode': False,
             'pipeline': [{'type': 'LoadImage'},
                          {'type': 'GetBBoxCenterScale', 'padding': 1.0},
                          {'type': 'RandomFlip', 'direction': 'horizontal'},
                          {'type': 'RandomBBoxTransform',
                           'rotate_factor': 5,
                           'scale_factor': (1.0, 1.1),
                           'shift_factor': 0.02,
                           'shift_prob': 0.3,
                           'scale_prob': 1.0,
                           'rotate_prob': 0.6},
                          {'type': 'TopdownAffine', 'input_size': (192, 480)},
                          {'type': 'Foot13PhotometricAug',
                           'brightness_limit': 0.1,
                           'contrast_limit': 0.1,
                           'brightness_contrast_prob': 0.3,
                           'blur_kernel': 3,
                           'blur_prob': 0.1},
                          {'type': 'GenerateTarget',
                           'encoder': {'type': 'MSRAHeatmap',
                                       'input_size': (192, 480),
                                       'heatmap_size': (48, 120),
                                       'sigma': 2}},
                          {'type': 'PackPoseInputs'}]},
 'batch_size': 32}

val_dataloader = {'num_workers': 0,
 'persistent_workers': False,
 'dataset': {'_delete_': True,
             'type': 'CocoDataset',
             'data_mode': 'topdown',
             'data_root': '../data/Foot13/',
             'ann_file': 'annotations/foot13_val.json',
             'bbox_file': None,
             'data_prefix': {'img': 'images/'},
             'metainfo': {'from_file': 'configs/_base_/datasets/foot13.py'},
             'test_mode': True,
             'pipeline': [{'type': 'LoadImage'},
                          {'type': 'GetBBoxCenterScale', 'padding': 1.0},
                          {'type': 'TopdownAffine', 'input_size': (192, 480)},
                          {'type': 'PackPoseInputs'}]},
 'sampler': {'type': 'DefaultSampler', 'shuffle': False, 'round_up': False}}

val_evaluator = [{'type': 'CocoMetric', 'ann_file': '../data/Foot13/annotations/foot13_val.json'},
 {'type': 'Foot13OBBMetric',
  'ann_file': '../data/Foot13/annotations/foot13_val.json',
  'geometry_file': '../data/Foot13/metadata/geometry_all.json'}]

test_dataloader = {'num_workers': 0,
 'persistent_workers': False,
 'dataset': {'_delete_': True,
             'type': 'CocoDataset',
             'data_mode': 'topdown',
             'data_root': '../data/Foot13/',
             'ann_file': 'annotations/foot13_test.json',
             'bbox_file': None,
             'data_prefix': {'img': 'images/'},
             'metainfo': {'from_file': 'configs/_base_/datasets/foot13.py'},
             'test_mode': True,
             'pipeline': [{'type': 'LoadImage'},
                          {'type': 'GetBBoxCenterScale', 'padding': 1.0},
                          {'type': 'TopdownAffine', 'input_size': (192, 480)},
                          {'type': 'PackPoseInputs'}]},
 'sampler': {'type': 'DefaultSampler', 'shuffle': False, 'round_up': False}}

test_evaluator = [{'type': 'CocoMetric', 'ann_file': '../data/Foot13/annotations/foot13_test.json'},
 {'type': 'Foot13OBBMetric',
  'ann_file': '../data/Foot13/annotations/foot13_test.json',
  'geometry_file': '../data/Foot13/metadata/geometry_all.json'}]

# Full training state each epoch; retain the latest two plus separate best AP weights.
default_hooks = dict(checkpoint=dict(
    interval=1,
    by_epoch=True,
    max_keep_ckpts=2,
    save_optimizer=True,
    save_param_scheduler=True,
    save_best='coco/AP',
    rule='greater',
    save_last=True))

custom_imports = {'imports': ['projects.foot13.transforms', 'projects.foot13.metrics'],
 'allow_failed_imports': False}

optim_wrapper = {'type': 'AmpOptimWrapper',
 'loss_scale': 'dynamic',
 'dtype': 'float16',
 'accumulative_counts': 1,
 'optimizer': {'lr': 0.0005}}
auto_scale_lr = {'enable': False, 'base_batch_size': 32}

custom_hooks = [
    dict(type='SyncBuffersHook'),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/AP',
        rule='greater',
        patience=15,
        min_delta=0.0001,
        strict=True,
    ),
]
