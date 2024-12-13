 python tools/relation_train_net.py --config-file "configs/sup-50.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
  SOLVER.PRE_VAL False \
  SOLVER.IMS_PER_BATCH 12 \
  TEST.IMS_PER_BATCH 1 \
  DTYPE "float16" \
  SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ../glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ../checkpoint/motif-sgdet_0.95-5.0  \
  SOLVER.BASE_LR 0.001 \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True\
  TEST.METRIC "R"
