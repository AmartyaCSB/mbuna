"""
Basic Dataset Annotation Example
"""

from mbuna.constants import DATA_DIR
import fiftyone as fo
import fiftyone.brain as fob
import os

if 'demo' in fo.list_datasets():
    fo.delete_dataset('demo')

dataset_dir = os.path.join(DATA_DIR, 'demo')
img_dir = os.path.join(dataset_dir, 'images')
name = 'demo'

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
    label_field='sex',
    classes=['male', 'female'],
    launch_editor=True
)
