# train a supervised model
bash cmds/50/motif/predcls/sup/train.sh
bash cmds/50/vctree/predcls/sup/train.sh
bash cmds/50/transformer/predcls/sup/train.sh
#bash cmds/50/motif/sgcls/lt/combine/train_motif.sh

# conduct internal transfer
bash cmds/50/motif/predcls/lt/internal/relabel.sh
bash cmds/50/vctree/predcls/lt/internal/relabel.sh
bash cmds/50/transformer/predcls/lt/internal/relabel.sh
bash cmds/50/motif/predcls/lt/internal/relabel_cut.sh
bash cmds/50/vctree/predcls/lt/internal/relabel_cut.sh

#external
#bash cmds/50/motif/predcls/lt/external/relabel.sh

# train a new model
bash cmds/50/motif/predcls/lt/combine/train.sh
bash cmds/50/motif/sgcls/lt/combine/train.sh
bash cmds/50/motif/sgdet/lt/combine/train.sh
bash cmds/50/vctree/predcls/lt/combine/train.sh
bash cmds/50/motif/sgcls/lt/combine/train.sh
bash cmds/50/motif/sgdet/lt/combine/train.sh+

#val
bash cmds/50/motif/predcls/lt/combine/val.sh
bash cmds/50/vctree/predcls/lt/combine/val.sh
bash cmds/50/motif/sgcls/lt/combine/val.sh
bash cmds/50/motif/sgdet/lt/combine/val.sh

#confusion
bash cmds/50/motif/predcls/lt/internal/confusion.sh
bash cmds/50/motif/predcls/lt/internal/confusion_ori.sh
bash cmds/50/motif/predcls/lt/internal/confusion_cut.sh
bash cmds/50/motif/predcls/lt/external/relabel.sh
bash cmds/50/motif/sgcls/lt/combine/sgcls_confusion.sh
bash cmds/50/motif/sgcls/lt/combine/confusion.sh
bash cmds/50/motif/sgcls/lt/combine/confusion_ori.sh
bash cmds/50/motif/sgcls/lt/combine/confusion_cut.sh
bash cmds/50/motif/sgcls/lt/combine/confusion_cut_ori.sh

#val
bash cmds/50/motif/predcls/lt/combine/confusion_val.sh
bash cmds/50/motif/sgcls/lt/combine/confusion_val.sh