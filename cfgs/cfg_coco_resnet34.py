'''cfg file for coco2017 dataset'''


# anchors
ANCHOR_SCALES = [8]
ANCHOR_RATIOS = [0.5, 1, 2]
ANCHOR_SIZE_BASES = [4, 8, 16, 32, 64]
# RPN, RoI settings
TRAIN_RPN_PRE_NMS_TOP_N = 2000
TRAIN_RPN_POST_NMS_TOP_N = 2000
TRAIN_RPN_NMS_THRESH = 0.7
TRAIN_RPN_NEGATIVE_OVERLAP = 0.3
TRAIN_RPN_POSITIVE_OVERLAP = 0.7
TRAIN_RPN_FG_FRACTION = 0.5
TRAIN_RPN_BATCHSIZE = 256
TRAIN_BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
TRAIN_BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
TRAIN_POOLING_METHOD = ['align', 'pool'][0]
TRAIN_POOLING_SIZE = 7
TRAIN_POOLING_SAMPLE_NUM = 2
TRAIN_ROI_MAP_LEVEL_SCALE = 56
TRAIN_ROI_BATCHSIZE = 512
TRAIN_ROI_FG_FRACTION = 0.25
TRAIN_ROI_FG_THRESH = 0.5
TRAIN_ROI_BG_THRESH_HI = 0.5
TRAIN_ROI_BG_THRESH_LO = 0.0
TEST_RPN_PRE_NMS_TOP_N = 1000
TEST_RPN_POST_NMS_TOP_N = 1000
TEST_RPN_NMS_THRESH = 0.7
TEST_RPN_NEGATIVE_OVERLAP = 0.3
TEST_RPN_POSITIVE_OVERLAP = 0.7
TEST_RPN_FG_FRACTION = 0.5
TEST_RPN_BATCHSIZE = 256
TEST_BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
TEST_BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
TEST_POOLING_METHOD = ['align', 'pool'][0]
TEST_POOLING_SIZE = 7
TEST_POOLING_SAMPLE_NUM = 2
TEST_ROI_MAP_LEVEL_SCALE = 56
TEST_ROI_BATCHSIZE = 512
TEST_ROI_FG_FRACTION = 0.25
TEST_ROI_FG_THRESH = 0.5
TEST_ROI_BG_THRESH_HI = 0.5
TEST_ROI_BG_THRESH_LO = 0.0
# backbone
BACKBONE_TYPE = 'resnet34'
PRETRAINED_MODEL_PATH = ''
USE_CAFFE_PRETRAINED_MODEL = False
FIXED_FRONT_BLOCKS = True
ADDED_MODULES_WEIGHT_INIT_METHOD = {'fpn': 'xavier', 'rpn': 'normal', 'rcnn': 'normal'}
IS_MULTI_GPUS = True
IS_CLASS_AGNOSTIC = False
# dataset
DATASET_ROOT_DIR = ''
MAX_NUM_GT_BOXES = 50
NUM_CLASSES = 81
NUM_WORKERS = 8
PIN_MEMORY = True
BATCHSIZE = 16
CLSNAMESPATH = 'names/coco.names'
USE_COLOR_JITTER = False
IMAGE_NORMALIZE_INFO = {'caffe': {'mean_rgb': (0.4814576470588235, 0.4546921568627451, 0.40384352941176466), 'std_rgb': (1., 1., 1.)}, 'pytorch': {'mean_rgb': (0.485, 0.456, 0.406), 'std_rgb': (0.229, 0.224, 0.225)}}
# loss function
RPN_CLS_LOSS_SET = {'type': ['binary_cross_entropy'][0], 'binary_cross_entropy': {'size_average': True, 'weight': 1.}}
RCNN_CLS_LOSS_SET = {'type': ['cross_entropy'][0], 'cross_entropy': {'size_average': True, 'weight': 1.}}
RPN_REG_LOSS_SET = {'type': ['betaSmoothL1Loss'][0], 'betaSmoothL1Loss': {'beta': 1./9., 'size_average': True, 'weight': 1.}}
RCNN_REG_LOSS_SET = {'type': ['betaSmoothL1Loss'][0], 'betaSmoothL1Loss': {'beta': 1., 'size_average': True, 'weight': 1.}}
# optimizer
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LEARNING_RATES = [[2e-2, 2e-3, 2e-4], [1e-2, 1e-3, 1e-4]][int(USE_CAFFE_PRETRAINED_MODEL)]
LR_ADJUST_EPOCHS = [9, 12]
MAX_EPOCHS = 10
IS_USE_WARMUP = True
NUM_WARMUP_STEPS = 500
GRAD_CLIP_MAX_NORM = 35
GRAD_CLIP_NORM_TYPE = 2
# image size (max_len, min_len)
IMAGESIZE_DICT = {'LONG_SIDE': 1333, 'SHORT_SIDE': 800}
# record
TRAIN_BACKUPDIR = 'fpn_res34_trainbackup_coco'
TRAIN_LOGFILE = 'fpn_res34_trainbackup_coco/train.log'
TEST_BACKUPDIR = 'fpn_res34_testbackup_coco'
TEST_LOGFILE = 'fpn_res34_testbackup_coco/test.log'
TEST_BBOXES_SAVE_PATH = 'fpn_res34_testbackup_coco/fpn_res34_detection_results_coco.json'
SAVE_INTERVAL = 1