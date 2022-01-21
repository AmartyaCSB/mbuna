import mbuna.constants as coc
import fiftyone.core.cli as focli
from yolov5.detect import run as detect
from yolov5.export import run as export
from yolov5.train import main as train
from yolov5.val import run as val


class MbunaCommand(focli.Command):
    """The mbuna command-line interface."""

    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        focli._register_command(subparsers, "fiftyone", focli.FiftyOneCommand)
        focli._register_command(subparsers, "yolov5", YoloV5Command)


    @staticmethod
    def execute(parser, args):
        parser.print_help()

class YoloV5Command(focli.Command):
    """tools for analysing data using yolov5"""
    @staticmethod
    def setup(parser):
        subparsers = parser.add_subparsers(title="available commands")
        focli._register_command(subparsers, "train", TrainCommand)
        focli._register_command(subparsers, "val", focli.FiftyOneCommand)
        focli._register_command(subparsers, "detect", focli.FiftyOneCommand)
        focli._register_command(subparsers, "export", focli.FiftyOneCommand)

    @staticmethod
    def execute(parser, args):
        parser.print_help()

class TrainCommand(focli.Command):
    """train a yolov5 network on custom data"""
    @staticmethod
    def setup(parser):
        parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
        parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
        parser.add_argument('--data', type=str, default='', help='data.yaml path')
        parser.add_argument('--hyp', type=str, default='', help='hyperparameters.hyp path')
        parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
        parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
        parser.add_argument('--rect', action='store_true', help='rectangular training')
        parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
        parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
        parser.add_argument('--noval', action='store_true', help='only validate final epoch')
        parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
        parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
        parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
        parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
        parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
        parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
        parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
        parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
        parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
        parser.add_argument('--project', default='runs/train', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--quad', action='store_true', help='quad dataloader')
        parser.add_argument('--linear-lr', action='store_true', help='linear LR')
        parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
        parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
        parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
        parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
        parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
        parser.add_argument('--mmdet_tags', action='store_true', help='Log train/val keys in MMDetection format')

        # Weights & Biases arguments
        parser.add_argument('--entity', default=None, help='W&B: Entity')
        parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
        parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

        # Neptune AI arguments
        parser.add_argument('--neptune_token', type=str, default=None, help='neptune.ai api token')
        parser.add_argument('--neptune_project', type=str, default=None, help='https://docs.neptune.ai/api-reference/neptune')

        # AWS arguments
        parser.add_argument('--s3_upload_dir', type=str, default=None, help='aws s3 folder directory to upload best weight and dataset')
        parser.add_argument('--upload_dataset', action='store_true', help='upload dataset to aws s3')

    @staticmethod
    def execute(parser, args):
        print(args)
        print('\n\n')
        print(parser.parse_args())
        pass


def main():
    """Executes the `mbuna` tool with the given command-line args."""
    parser = focli._register_main_command(MbunaCommand, version=coc.VERSION_LONG)
    args = parser.parse_args()
    args.execute(args)

