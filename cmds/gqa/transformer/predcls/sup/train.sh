python tools/relation_train_net.py --config-file "configs/sup-GQA200.yaml" \
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
  MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/faster-rcnn-gqa/model_final_from_gqa.pth \
  OUTPUT_DIR ../checkpoint/trans-precls-ori \
  SOLVER.BASE_LR 0.005  \
  TEST.METRIC "R"


