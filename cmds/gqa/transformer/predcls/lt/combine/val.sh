python tools/relation_test_net.py --config-file "configs/sup-GQA200.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor \
        TEST.IMS_PER_BATCH 1 \
        DTYPE "float16" \
        GLOVE_DIR ../glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/trans-precls-0.95-5.0/last_checkpoint \
        OUTPUT_DIR ../checkpoint/trans-precls-0.95-5.0   \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True
