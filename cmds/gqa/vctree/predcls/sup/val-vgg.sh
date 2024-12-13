python tools/relation_test_net.py --config-file "configs/sup-50-VGG.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
        TEST.IMS_PER_BATCH 1 \
        DTYPE "float16" \
        GLOVE_DIR ../glove \
        MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
        MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/vctree-predcls-vgg/last_checkpoint \
        OUTPUT_DIR ../checkpoint/vctree-predcls-vgg