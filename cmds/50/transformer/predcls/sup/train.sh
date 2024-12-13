python tools/relation_train_net.py --config-file "configs/sup-50.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
  SOLVER.PRE_VAL False \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR  \
  SOLVER.IMS_PER_BATCH 16 \
  TEST.IMS_PER_BATCH 2 \
  DTYPE "float16" \
  SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ../glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ../checkpoint/transformer-precls-ori \
  SOLVER.BASE_LR 0.0005  \
  TEST.METRIC "R"


