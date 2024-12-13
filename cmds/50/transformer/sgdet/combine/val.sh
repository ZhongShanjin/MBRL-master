python tools/relation_test_net.py --config-file "configs/sup-50.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
        TEST.IMS_PER_BATCH 1 \
        DTYPE "float16" \
        GLOVE_DIR ../glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/transformer-precls-0.95-5.0/last_checkpoint \
        OUTPUT_DIR ../checkpoint/transformer-sgdet-0.95-5.0   \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True
