"""Foot13 keypoint definitions and evaluation tolerances.
"""

dataset_info = dict(
    dataset_name='foot13',
    paper_info=dict(
        author='',
        title='Plantar Foot Keypoint Dataset',
        container='',
        year='2026',
        homepage='',
    ),
    keypoint_info={
        0: dict(name='FM', id=0, color=[255, 0, 0], type='', swap=''),
        1: dict(name='FL', id=1, color=[255, 85, 0], type='', swap=''),
        2: dict(name='MM', id=2, color=[255, 170, 0], type='', swap=''),
        3: dict(name='ML', id=3, color=[255, 255, 0], type='', swap=''),
        4: dict(name='HM', id=4, color=[170, 255, 0], type='', swap=''),
        5: dict(name='HL', id=5, color=[85, 255, 0], type='', swap=''),
        6: dict(name='HaMP', id=6, color=[0, 255, 0], type='', swap=''),
        7: dict(name='HeMP', id=7, color=[0, 255, 85], type='', swap=''),
        8: dict(name='FMMP', id=8, color=[0, 255, 170], type='', swap=''),
        9: dict(name='HR', id=9, color=[0, 255, 255], type='', swap=''),
        10: dict(name='STF', id=10, color=[0, 170, 255], type='', swap=''),
        11: dict(name='NT', id=11, color=[0, 85, 255], type='', swap=''),
        12: dict(name='DIM', id=12, color=[0, 0, 255], type='', swap=''),
    },
    skeleton_info={},
    joint_weights=[1.0] * 13,
    sigmas=[
        0.025,  # FM
        0.025,  # FL
        0.075,  # MM
        0.075,  # ML
        0.025,  # HM
        0.025,  # HL
        0.025,  # HaMP
        0.050,  # HeMP
        0.025,  # FMMP
        0.025,  # HR
        0.025,  # STF
        0.050,  # NT
        0.050,  # DIM
    ],
)