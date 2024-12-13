python tools/relation_train_net.py --config-file "configs/wsup-50-VGG.yaml" \
  DATASETS.TRAIN \(\"50DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.IMS_PER_BATCH 12 \
  TEST.IMS_PER_BATCH 1 \
  DTYPE "float16" SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ../glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/VGG_faster_rcnn/model_0044000.pth \
  OUTPUT_DIR ../checkpoint/motif-precls-vgg_0.95-5.0  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
  SOLVER.PRE_VAL False \
  SOLVER.BASE_LR 0.001 \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  TEST.INFERENCE "SOFTMAX"  \
  IETRANS.RWT False \
  WSUPERVISE.LOSS_TYPE ce \
  WSUPERVISE.DATASET InTransDataset \
  WSUPERVISE.SPECIFIED_DATA_FILE motifs-vgg.pk_HLM_1.0