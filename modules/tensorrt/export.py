import argparse
import sys
import time
import warnings

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

import models
from models.experimental import attempt_load, End2End
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from utils.add_nms import RegisterNMS


# onnx -> trt 
import os
import logging
import argparse

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from image_batch import ImageBatcher



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov7.pt', help='weights path')
    parser.add_argument('--save-name' ,type=str, default = '')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
    parser.add_argument('--dynamic-batch',default=True, action='store_true', help='dynamic batch onnx for tensorrt and onnx-runtime')
    parser.add_argument('--grid', action='store_true', help='export Detect() layer grid')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='conf threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--include-nms', action='store_true', help='export end2end onnx')
    parser.add_argument('--fp16', action='store_true', help='CoreML FP16 half-precision export')
    parser.add_argument('--int8', action='store_true', help='CoreML INT8 quantization')

    ## tensorrt convert
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load",default='name.onnx')
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine",default='name.trt')
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument("-w", "--workspace", default=20, type=int, help="The max memory workspace size to allow in Gb, "
                                                                       "default: 1")
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default="./calibration.cache",
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=5000, type=int,
                        help="The maximum number of images to use for calibration, default: 5000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 8")
    # parser.add_argument("--end2end", default=False, action="store_true",
    #                     help="export the engine include nms plugin, default: False")
    # parser.add_argument("--conf_thres", default=0.4, type=float,
    #                     help="The conf threshold for the nms, default: 0.4")
    # parser.add_argument("--iou_thres", default=0.5, type=float,
    #                     help="The iou threshold for the nms, default: 0.5")
    parser.add_argument("--max_det", default=100, type=int,
                        help="The total num for results, default: 100")
    parser.add_argument("-d", "--dynamic_batch_size", default=[1,16,32],
                    help="Enable dynamic batch size by providing a comma-separated MIN,OPT,MAX batch size, "
                            "if this option is set, --batch_size is ignored, example: -d 1,16,32, "
                            "default: None, build static engine")
    
    opt = parser.parse_args()
    args = opt
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.dynamic = opt.dynamic and not opt.end2end
    opt.dynamic = False if opt.dynamic_batch else opt.dynamic
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    device = select_device(opt.device)
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
    y = model(img)  # dry run
    if opt.include_nms:
        model.model[-1].include_nms = True
        y = None

    # ONNX export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = opt.weights.replace('.pt', '.onnx')  # filename
        model.eval()
        output_names = ['classes', 'boxes'] if y is None else ['output']
        dynamic_axes = None
        if opt.dynamic:
            dynamic_axes = {'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
             'output': {0: 'batch', 2: 'y', 3: 'x'}}
        if opt.dynamic_batch:
            opt.batch_size = 'batch'
            dynamic_axes = {
                'images': {
                    0: 'batch',
                }, }
            if opt.end2end and opt.max_wh is None:
                output_axes = {
                    'num_dets': {0: 'batch'},
                    'det_boxes': {0: 'batch'},
                    'det_scores': {0: 'batch'},
                    'det_classes': {0: 'batch'},
                }
            else:
                output_axes = {
                    'output': {0: 'batch'},
                }
            dynamic_axes.update(output_axes)
        if opt.grid:
            if opt.end2end:
                print('\nStarting export end2end onnx model for %s...' % 'TensorRT' if opt.max_wh is None else 'onnxruntime')
                model = End2End(model,opt.topk_all,opt.iou_thres,opt.conf_thres,opt.max_wh,device,len(labels))
                if opt.end2end and opt.max_wh is None:
                    output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                    shapes = [opt.batch_size, 1, opt.batch_size, opt.topk_all, 4,
                              opt.batch_size, opt.topk_all, opt.batch_size, opt.topk_all]
                else:
                    output_names = ['output']
            else:
                model.model[-1].concat = True

        torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                          output_names=output_names,
                          dynamic_axes=dynamic_axes)

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model

        if opt.end2end and opt.max_wh is None:
            for i in onnx_model.graph.output:
                for j in i.type.tensor_type.shape.dim:
                    j.dim_param = str(shapes.pop(0))

        if opt.simplify:
            try:
                import onnxsim

                print('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                print(f'Simplifier failure: {e}')

        if opt.save_name != '':
            f = opt.save_name
        else:
            f = opt.weights[:-2] + 'onnx'
        onnx.save(onnx_model,f)

        print('ONNX export success, saved as %s' % f)

        # if opt.include_nms:
        #     print('Registering NMS plugin for ONNX...')
        #     mo = RegisterNMS(f)
        #     mo.register_nms()
        #     mo.save(f)

    except Exception as e:
        print('ONNX export failure: %s' % e)

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))



    # onnx -> trt


    logging.basicConfig(level=logging.INFO)
    logging.getLogger("EngineBuilder").setLevel(logging.INFO)
    log = logging.getLogger("EngineBuilder")

    class EngineCalibrator(trt.IInt8EntropyCalibrator2):
        """
        Implements the INT8 Entropy Calibrator 2.
        """

        def __init__(self, cache_file):
            """
            :param cache_file: The location of the cache file.
            """
            super().__init__()
            self.cache_file = cache_file
            self.image_batcher = None
            self.batch_allocation = None
            self.batch_generator = None

        def set_image_batcher(self, image_batcher: ImageBatcher):
            """
            Define the image batcher to use, if any. If using only the cache file, an image batcher doesn't need
            to be defined.
            :param image_batcher: The ImageBatcher object
            """
            self.image_batcher = image_batcher
            size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
            self.batch_allocation = cuda.mem_alloc(size)
            self.batch_generator = self.image_batcher.get_batch()

        def get_batch_size(self):
            """
            Overrides from trt.IInt8EntropyCalibrator2.
            Get the batch size to use for calibration.
            :return: Batch size.
            """
            if self.image_batcher:
                return self.image_batcher.batch_size
            return 1

        def get_batch(self, names):
            """
            Overrides from trt.IInt8EntropyCalibrator2.
            Get the next batch to use for calibration, as a list of device memory pointers.
            :param names: The names of the inputs, if useful to define the order of inputs.
            :return: A list of int-casted memory pointers.
            """
            if not self.image_batcher:
                return None
            try:
                batch, _, _ = next(self.batch_generator)
                log.info("Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images))
                cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
                return [int(self.batch_allocation)]
            except StopIteration:
                log.info("Finished calibration batches")
                return None

        def read_calibration_cache(self):
            """
            Overrides from trt.IInt8EntropyCalibrator2.
            Read the calibration cache file stored on disk, if it exists.
            :return: The contents of the cache file, if any.
            """
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    log.info("Using calibration cache file: {}".format(self.cache_file))
                    return f.read()

        def write_calibration_cache(self, cache):
            """
            Overrides from trt.IInt8EntropyCalibrator2.
            Store the calibration cache to a file on disk.
            :param cache: The contents of the calibration cache to store.
            """
            with open(self.cache_file, "wb") as f:
                log.info("Writing calibration cache data to: {}".format(self.cache_file))
                f.write(cache)

    class EngineBuilder:
        """
        Parses an ONNX graph and builds a TensorRT engine from it.
        """
        def __init__(self, verbose=False, workspace=8):
            """
            :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
            :param workspace: Max memory workspace to allow, in Gb.
            """
            self.trt_logger = trt.Logger(trt.Logger.INFO)
            if verbose:
                self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

            trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

            self.builder = trt.Builder(self.trt_logger)
            self.config = self.builder.create_builder_config()
            self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (2 ** 30))
            # self.config.max_workspace_size = workspace * (2 ** 30)  # Deprecation

            self.batch_size = None
            self.network = None
            self.parser = None

        def create_network(self, onnx_path, end2end, conf_thres, iou_thres, max_det ,dynamic_batch_size):

            """
            Parse the ONNX graph and create the corresponding TensorRT network definition.
            :param onnx_path: The path to the ONNX graph to load.
            """

            network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

            self.network = self.builder.create_network(network_flags)
            self.parser = trt.OnnxParser(self.network, self.trt_logger)

            onnx_path = os.path.realpath(onnx_path)
            with open(onnx_path, "rb") as f:
                if not self.parser.parse(f.read()):
                    print("Failed to load ONNX file: {}".format(onnx_path))
                    for error in range(self.parser.num_errors):
                        print(self.parser.get_error(error))
                    sys.exit(1)

            inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
            outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

            print("Network Description")
            for input in inputs:
                print('self.batch_size : ',input.shape[0])
                print('input.shape : ',input.shape)
                self.batch_size = input.shape[0]
                print("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
            for output in outputs:
                print("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))

            ## onnx 가 dynamic batch 인 경우 tensorrt build 할 떄 profile 을 정의 해줘야한다.
            ## profile
            profile = self.builder.create_optimization_profile()
            dynamic_inputs = False
            for input in inputs:
                log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
                if input.shape[0] == -1:
                    dynamic_inputs = True
                    print(dynamic_batch_size)
                    
                    if dynamic_batch_size:
                        if type(dynamic_batch_size) is str:
                            dynamic_batch_size = [int(v) for v in dynamic_batch_size.split(",")]
                        print(dynamic_batch_size)
                        assert len(dynamic_batch_size) == 3
                        min_shape = [dynamic_batch_size[0]] + list(input.shape[1:])
                        opt_shape = [dynamic_batch_size[1]] + list(input.shape[1:])
                        max_shape = [dynamic_batch_size[2]] + list(input.shape[1:])
                        profile.set_shape(input.name, min_shape, opt_shape, max_shape)
                        log.info("Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(
                            input.name, min_shape, opt_shape, max_shape))
                    else:
                        shape = [1] + list(input.shape[1:])
                        profile.set_shape(input.name, shape, shape, shape)
                        log.info("Input '{}' Optimization Profile with shape {}".format(input.name, shape))
            if dynamic_inputs:
                self.config.add_optimization_profile(profile)


            # assert self.batch_size > 0
            # self.builder.max_batch_size = self.batch_size  # This no effect for networks created with explicit batch dimension mode. Also DEPRECATED.

            # if end2end:
            #     previous_output = self.network.get_output(0)
            #     self.network.unmark_output(previous_output)
            #     # output [1, 8400, 85]
            #     # slice boxes, obj_score, class_scores
            #     strides = trt.Dims([1,1,1])
            #     starts = trt.Dims([0,0,0])
            #     print('previous_output.shape :',previous_output.shape)
            #     bs, num_boxes, temp = previous_output.shape
            #     shapes = trt.Dims([bs, num_boxes, 4])
            #     # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
            #     boxes = self.network.add_slice(previous_output, starts, shapes, strides)
            #     num_classes = temp -5 
            #     starts[2] = 4
            #     shapes[2] = 1
            #     # [0, 0, 4] [1, 8400, 1] [1, 1, 1]
            #     obj_score = self.network.add_slice(previous_output, starts, shapes, strides)
            #     starts[2] = 5
            #     shapes[2] = num_classes
            #     # [0, 0, 5] [1, 8400, 80] [1, 1, 1]
            #     scores = self.network.add_slice(previous_output, starts, shapes, strides)
            #     # scores = obj_score * class_scores => [bs, num_boxes, nc]
            #     updated_scores = self.network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)

            #     '''
            #     "plugin_version": "1",
            #     "background_class": -1,  # no background class
            #     "max_output_boxes": detections_per_img,
            #     "score_threshold": score_thresh,
            #     "iou_threshold": nms_thresh,
            #     "score_activation": False,
            #     "box_coding": 1,
            #     '''
            #     registry = trt.get_plugin_registry()
            #     assert(registry)
            #     creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
            #     assert(creator)
            #     fc = []
            #     fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
            #     fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
            #     fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            #     fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
            #     fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
                
            #     fc = trt.PluginFieldCollection(fc) 
            #     nms_layer = creator.create_plugin("nms_layer", fc)

            #     layer = self.network.add_plugin_v2([boxes.get_output(0), updated_scores.get_output(0)], nms_layer)
            #     layer.get_output(0).name = "num"
            #     layer.get_output(1).name = "boxes"
            #     layer.get_output(2).name = "scores"
            #     layer.get_output(3).name = "classes"
            #     for i in range(4):
            #         self.network.mark_output(layer.get_output(i))


        def create_engine(self, engine_path, precision, calib_input=None, calib_cache=None, calib_num_images=5000,
                        calib_batch_size=8):
            """
            Build the TensorRT engine and serialize it to disk.
            :param engine_path: The path where to serialize the engine to.
            :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
            :param calib_input: The path to a directory holding the calibration images.
            :param calib_cache: The path where to write the calibration cache to, or if it already exists, load it from.
            :param calib_num_images: The maximum number of images to use for calibration.
            :param calib_batch_size: The batch size to use for the calibration process.
            """
            engine_path = os.path.realpath(engine_path)
            engine_dir = os.path.dirname(engine_path)
            os.makedirs(engine_dir, exist_ok=True)
            print("Building {} Engine in {}".format(precision, engine_path))
            inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

            # TODO: Strict type is only needed If the per-layer precision overrides are used
            # If a better method is found to deal with that issue, this flag can be removed.
            self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            print('precision : ',precision)
            if precision == "fp16":
                if not self.builder.platform_has_fast_fp16:
                    print("FP16 is not supported natively on this platform/device")
                else:
                    self.config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                if not self.builder.platform_has_fast_int8:
                    print("INT8 is not supported natively on this platform/device")
                else:
                    if self.builder.platform_has_fast_fp16:
                        # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                        self.config.set_flag(trt.BuilderFlag.FP16)
                    self.config.set_flag(trt.BuilderFlag.INT8)
                    self.config.int8_calibrator = EngineCalibrator(calib_cache)
                    if not os.path.exists(calib_cache):
                        calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                        calib_dtype = trt.nptype(inputs[0].dtype)
                        self.config.int8_calibrator.set_image_batcher(
                            ImageBatcher(calib_input, calib_shape, calib_dtype, max_num_images=calib_num_images,
                                        exact_batches=True))

            # with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
                print("Serializing engine to file: {:}".format(engine_path))
                f.write(engine)  # .serialize()


    def main(args):
        builder = EngineBuilder(args.verbose, args.workspace)
        # builder.create_network(f, args.end2end, args.conf_thres, args.iou_thres, args.max_det, args.dynamic_batch_size)
        args.onnx = f
        builder.create_network(args.onnx, args.end2end, args.conf_thres, args.iou_thres, args.max_det, args.dynamic_batch_size)
        builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
                            args.calib_batch_size)

    print(args)
    if not all([args.onnx, args.engine]):
        parser.print_help()
        log.error("These arguments are required: --onnx and --engine")
        sys.exit(1)
    if args.precision == "int8" and not (args.calib_input or os.path.exists(args.calib_cache)):
        parser.print_help()
        log.error("When building in int8 precision, --calib_input or an existing --calib_cache file is required")
        sys.exit(1)
    
    main(args)




