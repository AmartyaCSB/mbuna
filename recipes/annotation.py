"""
Basic Dataset Annotation Example
"""

from mbuna.constants import DATA_DIR
import fiftyone as fo
import fiftyone.brain as fob
import os

dataset_dir = os.path.join(DATA_DIR, 'sand')
img_dir = os.path.join(dataset_dir, 'images')
name = 'sand'

dataset = fo.Dataset.from_dir(
    dataset_dir=img_dir,
    dataset_type=fo.types.ImageDirectory,
    name=name,
)

dataset.persistent = True
results = fob.compute_similarity(dataset, brain_key="img_sim")
results.find_unique()
unique_view = dataset.select(results.unique_ids)
session = fo.launch_app(unique_view)



