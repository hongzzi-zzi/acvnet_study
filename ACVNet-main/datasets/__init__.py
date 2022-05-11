from .kitti_dataset_1215 import KITTIDataset
from .sceneflow_dataset import SceneFlowDatset
from .ourdata_dataset import OURDataset


__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "kitti": KITTIDataset,
    "ourdata": OURDataset
}
