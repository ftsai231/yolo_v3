from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()

# Set the class name
__C.YOLO.CLASSES = "./cocoapi/classes.txt"
__C.YOLO.ANCHORS = "1.25, 1.625, 2.0, 3.75, 4.125, 2.875, 1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375, 3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875"
__C.YOLO.MOVING_AVE_DECAY = 0.9995
__C.YOLO.STRIDES = [16, 32, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.3
__C.YOLO.UPSAMPLE_METHOD = "resize"
__C.YOLO.ORIGINAL_WEIGHT = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT = "./checkpoint/yolov3_coco_demo.ckpt"

# original
__C.YOLO.confidence_threshold = 0.3  # confidence score threshold
__C.YOLO.iou_threshold = 0.3
__C.YOLO.max_output_size = 5

__C.YOLO.BATCH_NORM_DECAY = 0.9
__C.YOLO.BATCH_NORM_EPSILON = 1e-05
__C.YOLO.LEAKY_RELU = 0.1
__C.YOLO.ANCHORS = [(10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)]
__C.YOLO.MODEL_SIZE = (416, 416)

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "./cocoapi/annotations/instances_train2014.json"
__C.TRAIN.BATCH_SIZE = 32
# __C.TRAIN.INPUT_SIZE = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE = 416
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LEARN_RATE_INIT = 1e-4
__C.TRAIN.LEARN_RATE_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.FISRT_STAGE_EPOCHS = 20
__C.TRAIN.SECOND_STAGE_EPOCHS = 30
# __C.TRAIN.INITIAL_WEIGHT = "./checkpoint/yolov3_test_loss=1115.8914.ckpt-1.data-00000-of-00001"
__C.TRAIN.EPOCH = 2
__C.TRAIN.N_CLASSES = 2

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "./cocoapi/annotations/image_info_test2014.json"
__C.TEST.BATCH_SIZE = 32
__C.TEST.INPUT_SIZE = 544
__C.TEST.DATA_AUG = False
__C.TEST.WRITE_IMAGE = True
__C.TEST.WRITE_IMAGE_PATH = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE = "./checkpoint/yolov3_test_loss=9.2099.ckpt-5"
__C.TEST.SHOW_LABEL = True
__C.TEST.SCORE_THRESHOLD = 0.3
__C.TEST.IOU_THRESHOLD = 0.45

