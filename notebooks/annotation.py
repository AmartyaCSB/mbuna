"""
Basic Dataset Annotation Example
"""

from mbuna.constants import DATA_DIR
import fiftyone as fo
import fiftyone.brain as fob
import os


dataset_dir = os.path.join(DATA_DIR, 'kl')
img_dir = os.path.join(dataset_dir, 'images')
name = 'kl'

dataset = fo.Dataset.from_dir(
    dataset_dir=img_dir,
    dataset_type=fo.types.ImageDirectory,
    name=name,
)

session = fo.launch_app(dataset)

anno_key = 'demo_anno'
anno_results = dataset.annotate(
    anno_key=anno_key,
    label_type='detections',
    label_field='detect_ground_truth',
    classes=['fish'],
    launch_editor=True
)
