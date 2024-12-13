python tools/relation_test_net.py --config-file "configs/sup-GQA200.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
        TEST.IMS_PER_BATCH 1 \
        DTYPE "float16" \
        GLOVE_DIR ../glove \
        MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/vctree-sgcls_gqa-0.95-5.0/last_checkpoint \
        OUTPUT_DIR ../checkpoint/vctree-sgcls_gqa-0.95-5.0  \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True