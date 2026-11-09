"""RTMPose-M: upstream recipe adapted to Foot13, input width192/height480.

"""

_base_ = ['../body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py']

data_root = '../data/Foot13/'

dataset_metainfo = {'from_file': 'configs/_base_/datasets/foot13.py'}

work_dir = './work_dirs/foot13_480x192_rtmpose_m'

randomness = {'seed': 42}

load_from = None

resume = False

codec = {'type': 'SimCCLabel',
 'input_size': (192, 480),
 'sigma': (4.9, 5.66),
 'simcc_split_ratio': 2.0,
 'normalize': False,
 'use_dark': False}

model = {'backbone': {'init_cfg': {'type': 'Pretrained',
                           'prefix': 'backbone.',
                           'checkpoint': '../checkpoints/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'}},
 'head': {'out_channels': 13,
          'decoder': {'type': 'SimCCLabel',
                      'input_size': (192, 480),
                      'sigma': (4.9, 5.66),
                      'simcc_split_ratio': 2.0,
                      'normalize': False,
                      'use_dark': False},
          'input_size': (192, 480),
          'in_featuremap_size': (6, 15)}}

train_cfg = dict(max_epochs=210, val_interval=1)

train_pipeline = [{'type': 'LoadImage', 'backend_args': {'backend': 'local'}},
 {'type': 'GetBBoxCenterScale', 'padding': 1.0},
 {'type': 'RandomFlip', 'direction': 'horizontal'},
 {'type': 'RandomBBoxTransform',
  'scale_factor': (1.0, 1.1),
  'rotate_factor': 5,
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
  'encoder': {'type': 'SimCCLabel',
              'input_size': (192, 480),
              'sigma': (4.9, 5.66),
              'simcc_split_ratio': 2.0,
              'normalize': False,
              'use_dark': False}},
 {'type': 'PackPoseInputs'}]

val_pipeline = [{'type': 'LoadImage', 'backend_args': {'backend': 'local'}},
 {'type': 'GetBBoxCenterScale', 'padding': 1.0},
 {'type': 'TopdownAffine', 'input_size': (192, 480)},
 {'type': 'PackPoseInputs'}]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/AP',
        rule='greater',
        patience=15,
        min_delta=0.0001,
        strict=True,
    ),
]

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
             'pipeline': [{'type': 'LoadImage', 'backend_args': {'backend': 'local'}},
                          {'type': 'GetBBoxCenterScale', 'padding': 1.0},
                          {'type': 'RandomFlip', 'direction': 'horizontal'},
                          {'type': 'RandomBBoxTransform',
                           'scale_factor': (1.0, 1.1),
                           'rotate_factor': 5,
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
                           'encoder': {'type': 'SimCCLabel',
                                       'input_size': (192, 480),
                                       'sigma': (4.9, 5.66),
                                       'simcc_split_ratio': 2.0,
                                       'normalize': False,
                                       'use_dark': False}},
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
             'pipeline': [{'type': 'LoadImage', 'backend_args': {'backend': 'local'}},
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
             'pipeline': [{'type': 'LoadImage', 'backend_args': {'backend': 'local'}},
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

# Repository-local image-only augmentation; preserve upstream custom imports.
custom_imports = {'imports': ['projects.foot13.transforms', 'projects.foot13.metrics'],
 'allow_failed_imports': False}

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    dtype='float16',
    accumulative_counts=1,
    optimizer=dict(_delete_=True, type='Adam', lr=0.0005, weight_decay=0.0),
)
auto_scale_lr = dict(enable=False, base_batch_size=32)

param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),
    dict(
        type='MultiStepLR', begin=0, end=210, milestones=[170, 200],
        gamma=0.1, by_epoch=True),
]
