# Foot13 Benchmark Code

Foot13 provides benchmark code for localising 13 landmarks in single-foot podoscopic images and evaluating five footprint measurements derived from those landmarks. The six model configurations use the prepared Foot13 crops and the supplied training, validation and test splits.

## Repository layout

~~~text
Foot13/
├── README.md
├── mmpose-main/                   # Foot13 files to copy into an MMPose checkout
│   ├── configs/
│   │   ├── _base_/datasets/foot13.py
│   │   └── foot13/
│   ├── projects/foot13/
│   └── tools/
│       ├── train.py
│       ├── test.py
│       └── measurements_tools/
├── data/                          
└── checkpoints/                
~~~

| Path under `mmpose-main/` | Purpose |
| --- | --- |
| `configs/_base_/datasets/foot13.py` | Canonical landmark order, dataset metadata and landmark-specific object keypoint similarity (OKS) tolerances. |
| `configs/foot13/` | Training and test configurations for six models with a network input of 192 pixels wide by 480 pixels high. |
| `projects/foot13/transforms.py` | Brightness and contrast augmentation and box blur for training. |
| `projects/foot13/metrics.py`, `projects/foot13/obb_errors.py` | Match predictions to the reference annotations and compute overall and per-landmark mean point error (MPE) and normalised mean error (NME). NME uses the expanded oriented bounding box diagonal. |
| `tools/train.py`, `tools/test.py` | Training and testing entry points. The test script exports summary metrics and predictions for offline analysis. |
| `tools/measurements_tools/` | Measurement functions and notebooks for calculating footprint values and summarising their errors. |

## Setup

1. In a separate working directory, obtain [MMPose v1.3.2](https://github.com/open-mmlab/mmpose/releases/tag/v1.3.2). For a fresh checkout, run:

   ~~~sh
   git clone --branch v1.3.2 --depth 1 https://github.com/open-mmlab/mmpose.git mmpose-main
   ~~~

2. Copy the **contents** of this repository's `mmpose-main/` into the root of that checkout. Merge the `configs/`, `projects/` and `tools/` directories and use the supplied versions of `tools/train.py` and `tools/test.py`.
3. Set up an environment using the [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html). 
4. Place the prepared Foot13 dataset and the required pretrained weights in the working directory alongside the checkout. Create the parent folders locally as needed.

   ~~~text
   foot13-workspace/
   ├── mmpose-main/
   ├── data/
   │   └── Foot13/
   │       ├── images/
   │       ├── annotations/
   │       │   ├── foot13_train.json
   │       │   ├── foot13_val.json
   │       │   └── foot13_test.json
   │       └── metadata/
   │           └── geometry_all.json
   └── checkpoints/
   ~~~

   The configuration paths `../data/Foot13/` and `../checkpoints/` resolve from the working MMPose checkout. Run the training and test commands from that directory.

## Model configurations

Download the initialisation weights for the models you plan to train into `checkpoints/` using the filenames shown below. Testing uses the corresponding trained Foot13 checkpoint passed to `tools/test.py`.

| Model | Config in `configs/foot13/` | Initialisation weights | Output stem |
| --- | --- | --- | --- |
| SimpleBaseline (ResNet-50) | `td-hm_res50_foot13-480x192.py` | [resnet50-0676ba61.pth](https://download.pytorch.org/models/resnet50-0676ba61.pth) | `simplebaseline_res50` |
| HRNet-W32 | `td-hm_hrnet-w32_foot13-480x192.py` | [hrnet_w32-36af842e.pth](https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth) | `hrnet_w32` |
| LiteHRNet-18 | `td-hm_litehrnet-18_foot13-480x192.py` | [litehrnet18_coco_256x192-6bace359_20211230.pth](https://download.openmmlab.com/mmpose/top_down/litehrnet/litehrnet18_coco_256x192-6bace359_20211230.pth) | `litehrnet18` |
| RTMPose-M | `rtmpose-m_foot13-480x192.py` | [cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth) | `rtmpose_m` |
| ViTPose-B | `td-hm_vitpose-b_foot13-480x192.py` | [mae_pretrain_vit_base_20230913.pth](https://download.openmmlab.com/mmpose/v1/pretrained_models/mae_pretrain_vit_base_20230913.pth) | `vitpose_b` |
| Four-stage MSPN | `td-hm_4xmspn50_foot13-480x192.py` | [resnet50-0676ba61.pth](https://download.pytorch.org/models/resnet50-0676ba61.pth) | `mspn_4stage` |

## Train and test

The commands below use SimpleBaseline. The configurations monitor validation average precision (AP) for early stopping and save the best checkpoint by validation AP.

~~~sh
python tools/train.py configs/foot13/td-hm_res50_foot13-480x192.py
~~~

Replace `N` in the checkpoint filename with the epoch of the saved best checkpoint.

~~~sh
python tools/test.py configs/foot13/td-hm_res50_foot13-480x192.py work_dirs/foot13_480x192_simplebaseline_res50/best_coco_AP_epoch_N.pth --work-dir test_results --out test_results/simplebaseline_res50_metrics.json --dump test_results/simplebaseline_res50_predictions.pkl
~~~

The metrics JSON contains OKS-based AP and average recall (AR), as well as overall and per-landmark MPE and NME. To test the other models, use their config and trained checkpoint and name their outputs `<stem>_metrics.json` and `<stem>_predictions.pkl` using the stems in the table.

## Evaluate footprint measurements

With Jupyter and pandas installed in the benchmark environment, open the notebooks in `tools/measurements_tools/` and run them in order. The notebooks resolve paths from that directory and expect the six prediction files under `test_results/`.

1. `0_compute_measurements.ipynb` applies `measurements.py` to the reference landmarks and each model's predictions. It saves one measurement CSV per model and a reference CSV under `test_results/0_measurements/`.
2. `1_compute_errors.ipynb` calculates prediction-minus-reference errors and summarises mean error (ME), mean absolute error (MAE) and root mean square error (RMSE). The summary is saved to `test_results/0_measurements/errors/measurement_error_summary.csv`.

Clarke's angle (CA) and the footprint-derived hallux valgus angle (HVA) are measured in degrees. Normalised medial prominence (NMP), the Staheli index (SI) and the Chippaux–Smirak index (CSI) are expressed as percentages, so their errors are reported in percentage points.
