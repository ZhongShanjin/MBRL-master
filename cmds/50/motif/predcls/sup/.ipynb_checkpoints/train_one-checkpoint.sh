#OUTPATH=../../../../../../IETrans-SGG.pytorch/exps/50/motif/predcls/sup/batch_one
python tools/relation_train_net.py \
--config-file "configs/sup-50.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True  \
MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
SOLVER.PRE_VAL False \
SOLVER.IMS_PER_BATCH 1 \
TEST.IMS_PER_BATCH 1 \
DTYPE "float16" \
SOLVER.MAX_ITER 1 \
SOLVER.VAL_PERIOD 1 \
SOLVER.CHECKPOINT_PERIOD 1 \
GLOVE_DIR ../glove \
MODEL.PRETRAINED_DETECTOR_CKPT ../checkpoint/pretrained_faster_rcnn/model_final.pth \
OUTPUT_DIR exps/50/motif/predcls/sup/batch_one  TEST.METRIC "R"
