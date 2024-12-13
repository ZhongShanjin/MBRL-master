python tools/relation_test_net.py --config-file "configs/sup-50.yaml" \
          MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
          MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
          MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
          TEST.IMS_PER_BATCH 1 \
          DTYPE "float16" GLOVE_DIR ../glove \
          MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/motif-sgcls-1.0/last_checkpoint \
          OUTPUT_DIR ../checkpoint/motif-sgcls-1.0 \
          MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
          Confusion True \
          Confusion_Path ../checkpoint/motif-sgcls-1.0/
