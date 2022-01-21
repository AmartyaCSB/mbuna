import mbuna.constants as coc
import fiftyone.core.cli as focli
from yolov5.detect import run as detect
from yolov5.export import run as export
from yolov5.train import run as train
from yolov5.val import run as val
import os


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
        train(**vars(args))

class ValCommand(focli.Command):
    @staticmethod
    def setup(parser):
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--batch-size', type=int, default=32, help='batch size')
        parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
        parser.add_argument('--task', default='val', help='train, val, test, speed or study')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--verbose', action='store_true', help='report mAP by class')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
        parser.add_argument('--project', default='runs/val', help='save to project/name')
        parser.add_argument('--name', default='exp', help='save to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    @staticmethod
    def execute(parser, args):
        val(**vars(args))

class DetectCommand(focli.Command):
    @staticmethod
    def setup(parser):
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default=os.path.join(coc.BASE_DIR, 'data/images'), help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    @staticmethod
    def execute(parser, args):
        detect(**vars(args))

class ExportCommand(focli.Command):
    @staticmethod
    def setup(parser):
        parser.add_argument('--data', type=str, default=os.path.join(coc.BASE_DIR, 'data/coco128.yaml'), help='dataset.yaml path')
        parser.add_argument('--weights', type=str, default='yolov5s.pt', help='weights path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640], help='image (h, w)')
        parser.add_argument('--batch-size', type=int, default=1, help='batch size')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
        parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
        parser.add_argument('--train', action='store_true', help='model.train() mode')
        parser.add_argument('--optimize', action='store_true', help='TorchScript: optimize for mobile')
        parser.add_argument('--int8', action='store_true', help='CoreML/TF INT8 quantization')
        parser.add_argument('--dynamic', action='store_true', help='ONNX/TF: dynamic axes')
        parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
        parser.add_argument('--opset', type=int, default=13, help='ONNX: opset version')
        parser.add_argument('--topk-per-class', type=int, default=100, help='TF.js NMS: topk per class to keep')
        parser.add_argument('--topk-all', type=int, default=100, help='TF.js NMS: topk for all classes to keep')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='TF.js NMS: IoU threshold')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='TF.js NMS: confidence threshold')
        parser.add_argument('--include', nargs='+',
                            default=['torchscript', 'onnx'],
                            help='available formats are (torchscript, onnx, coreml, saved_model, pb, tflite, tfjs)')
    @staticmethod
    def execute(parser, args):
        export(**vars(args))

def main(args=None):
    """Executes the `mbuna` tool with the given command-line args."""
    parser = focli._register_main_command(MbunaCommand, version=coc.VERSION_LONG)
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    print(args)
    args.execute(args)

