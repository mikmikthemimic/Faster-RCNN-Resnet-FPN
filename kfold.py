'''
    Implementation of K-Fold Cross Validation
'''

import json
import os
import torch
import warnings
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from modules.utils import *
from cfgs.getcfg import getCfgByDatasetAndBackbone
from modules.fasterRCNN import FasterRCNNFPNResNets
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold

def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def parseArgs():
	parser = argparse.ArgumentParser(description='Faster R-CNN with FPN')
	parser.add_argument('--datasetname', dest='datasetname', help='dataset for kfold.', default='', type=str, required=True)
	parser.add_argument('--annfilepath', dest='annfilepath', help='used to specify annfilepath.', default='', type=str)
	parser.add_argument('--datasettype', dest='datasettype', help='used to specify datasettype.', default='val2017', type=str)
	parser.add_argument('--backbonename', dest='backbonename', help='backbone network for kfold.', default='', type=str, required=True)
	parser.add_argument('--checkpointspath', dest='checkpointspath', help='checkpoints you want to use.', default='', type=str, required=True)
	parser.add_argument('--nmsthresh', dest='nmsthresh', help='thresh used in nms.', default=0.5, type=float)
	args = parser.parse_args()
	return args

def kfold():
    args = parseArgs()
    cfg, cfg_file_path = getCfgByDatasetAndBackbone(datasetname=args.datasetname, backbonename=args.backbonename)
    checkDir(cfg.KFOLD_BACKUPDIR)
    logger_handle = Logger(cfg.KFOLD_LOGFILE)
    use_cude = torch.cuda.is_available()
    is_multi_gpus = cfg.IS_MULTI_GPUS
    if is_multi_gpus: assert use_cuda
    # prepare dataset
    if args.datasetname == 'coco':
    	dataset = COCODataset(rootdir=cfg.DATASET_ROOT_DIR, image_size_dict=cfg.IMAGESIZE_DICT, max_num_gt_boxes=cfg.MAX_NUM_GT_BOXES, use_color_jitter=cfg.USE_COLOR_JITTER, img_norm_info=cfg.IMAGE_NORMALIZE_INFO, use_caffe_pretrained_model=cfg.USE_CAFFE_PRETRAINED_MODEL, mode='TRAIN', datasettype='train2017')
    	dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCHSIZE, sampler=NearestRatioRandomSampler(dataset.img_ratios, cfg.BATCHSIZE), num_workers=cfg.NUM_WORKERS, collate_fn=COCODataset.paddingCollateFn, pin_memory=cfg.PIN_MEMORY)
    else:
    	raise ValueError('Unsupport datasetname <%s> now...' % args.datasetname)
    # prepare model
    if args.backbonename.find('resnet') != -1:
    	model = FasterRCNNFPNResNets(mode='TRAIN', cfg=cfg, logger_handle=logger_handle)
    else:
    	raise ValueError('Unsupport backbonename <%s> now...' % args.backbonename)
    
    if use_cuda:
        model = model.cuda()
    model.reset_weights()
    # configuration options
    k_folds = 5         #from kfold guide
    num_epochs = 1      #from kfold guide
    results = []
    kfold = KFold(n_splits=k_folds, shuffle=True)
    # prepare optimizer
    learning_rate_idx = 0
    if cfg.IS_USE_WARMUP:
    	learning_rate = cfg.LEARNING_RATES[learning_rate_idx] / 3
    else:
    	learning_rate = cfg.LEARNING_RATES[learning_rate_idx]
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    # check checkpoints path
    if args.checkpointspath:
    	checkpoints = loadCheckpoints(args.checkpointspath, logger_handle)
    	model.load_state_dict(checkpoints['model'])
    	optimizer.load_state_dict(checkpoints['optimizer'])
    	start_epoch = checkpoints['epoch'] + 1
    	for epoch in range(0, start_epoch):
    		if epoch in cfg.LR_ADJUST_EPOCHS:
    			learning_rate_idx += 1
    #data parallel
    if is_multi_gpus:
    	model = nn.DataParallel(model)
    #print config
    logger_handle.info('Dataset used: %s, Number of images: %s' % (args.datasetname, len(dataset)))
    logger_handle.info('Backbone used: %s' % args.backbonename)
    logger_handle.info('Checkpoints used: %s' % args.checkpointspath)
    logger_handle.info('Config file used: %s' % cfg_file_path)
    #train
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        trainloader = torch.utils.data.DataLoader()
        testloader = torch.utils.data.DataLoader()




    #prepare dataset for testing

    #prepare model - mode from  TRAIN to TEST

    #load checkpoints

    #test mAP


if __name__ == '__main__':
    kfold()