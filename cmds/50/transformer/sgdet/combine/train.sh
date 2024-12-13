python tools/relation_train_net.py --config-file "configs/wsup-50.yaml" \
  DATASETS.TRAIN \(\"50DS_VG_VGKB_train\",\) \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
  SOLVER.IMS_PER_BATCH 16 \
  TEST.IMS_PER_BATCH 1 \
  DTYPE "float16" \
  SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ../glove \
  MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/pretrained_faster_rcnn/model_final.pth \
  OUTPUT_DIR ../checkpoint/transformer-sgdet-0.95-5.0  \
  MODEL.ROI_RELATION_HEAD.NUM_CLASSES 51 \
  SOLVER.PRE_VAL False \
  SOLVER.BASE_LR 0.005 \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  TEST.INFERENCE "SOFTMAX" \
  WSUPERVISE.LOSS_TYPE ce \
  IETRANS.RWT False \
  WSUPERVISE.DATASET InTransDataset  \
  WSUPERVISE.SPECIFIED_DATA_FILE  ./hlm/transformer.pk_HLM_1.0
