import mbuna.constants as coc
import fiftyone.core.cli as focli
from mbuna.yolov5.detect import main as detect
from mbuna.yolov5.export import main as export
from mbuna.yolov5.train import main as train
from mbuna.yolov5.val import main as val
from mbuna.yolov5.detect import setup_parser as detect_setup
from mbuna.yolov5.export import setup_parser as export_setup
from mbuna.yolov5.train import setup_parser as train_setup
from mbuna.yolov5.val import setup_parser as val_setup


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
        focli._register_command(subparsers, "val", ValCommand)
        focli._register_command(subparsers, "detect", DetectCommand)
        focli._register_command(subparsers, "export", ExportCommand)

    @staticmethod
    def execute(parser, args):
        parser.print_help()


class TrainCommand(focli.Command):
    """Train a YOLOv5 model on a custom dataset."""
    @staticmethod
    def setup(parser):
        train_setup(parser)

    @staticmethod
    def execute(parser, args):
        train(args)


class ValCommand(focli.Command):
    """Validate a trained YOLOv5 model accuracy on a custom dataset"""
    @staticmethod
    def setup(parser):
        val_setup(parser)

    @staticmethod
    def execute(parser, args):
        val(args)

class DetectCommand(focli.Command):
    """Run inference on images, videos, directories, streams, etc."""
    @staticmethod
    def setup(parser):
        detect_setup(parser)

    @staticmethod
    def execute(parser, args):
        detect(args)

class ExportCommand(focli.Command):
    """Export a YOLOv5 PyTorch model to other formats"""
    @staticmethod
    def setup(parser):
        export_setup(parser)

    @staticmethod
    def execute(parser, args):
        export(args)

def main(args=None):
    """Executes the `mbuna` tool with the given command-line args."""
    parser = focli._register_main_command(MbunaCommand, version=coc.VERSION_LONG)
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    args.execute(args)

