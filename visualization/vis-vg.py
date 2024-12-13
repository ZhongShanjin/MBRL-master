import pickle, json, os, sys, torch, h5py
import numpy as np
from tqdm import tqdm
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
from IPython.display import display

# 读取数据集
dic_hlm = pickle.load(open("../tmp/motifs_test-hlm.pk", "rb"))
dic = pickle.load(open("../tmp/motifs_test.pk", "rb"))
vocab = json.load(open("../datasets/vg/50/VG-SGG-dicts-with-attri.json"))

# 绘制边界框
def draw_single_box(pic, box, color='red', draw_info=None):
    draw = ImageDraw.Draw(pic)
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])  # 边界框坐标
    draw.rectangle(((x1, y1), (x2, y2)), outline=color)  # 绘制边界框
    if draw_info:  # 绘制边界框名称
        draw.rectangle(((x1, y1), (x1 + 50, y1 + 10)), fill=color)
        info = draw_info
        draw.text((x1, y1), info, font_size=12)

def draw_image(img_path, data):
    # 路径合并
    img_path = os.path.join("../", img_path)
    # 调整图片尺寸
    print(Image.open(img_path).size)
    print(data['size'])
    pic = Image.open(img_path).resize(data['size'])
    boxes = np.array(data['bbox'])
    labels = data['labels']
    labels = [str(i) + "-" + vocab['idx_to_label'][str(int(x))] for i, x in enumerate(labels)]

    for i in data['gt_pair'][:, 0]:
        info = labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)
    for i in data['gt_pair'][:, 1]:
        info = labels[i]
        draw_single_box(pic, boxes[i], draw_info=info)

    all_rel_pairs = data['gt_pair']
    all_rel_labels = data['gt_rel']
    all_rel_scores = data['pred_rel']
    rel_labels = []
    rel_scores = []
    for i in range(len(all_rel_pairs)):
        rel_scores.append(str(all_rel_scores[i]) + '-' + vocab["idx_to_predicate"][str(int(all_rel_scores[i]))])
        label = '(' + labels[all_rel_pairs[i][0]] + ', ' + str(int(all_rel_labels[i])) + '-' + \
                vocab["idx_to_predicate"][str(int(all_rel_labels[i]))] + \
                ', ' + labels[all_rel_pairs[i][1]] + ')'
        rel_labels.append(label)

    for a, b in list(zip(rel_labels, rel_scores)):
        print(a, b)

    display(pic)
    return None

more_ids = np.array([2,3,4,5,10,11,12,13,14,15,16,17,18,19,23,24,26,27,28,32,33,34,35,36,37,38,39,41,42,44,45,46])
idxs = [98,288,558,1772,2019,3569,4945,7945,9261,9281,9377,9975,10738,14128,16895,18879,25085]
for idx in idxs:
    count_true = np.count_nonzero(np.equal(dic_hlm[idx]['gt_rel'], dic_hlm[idx]['pred_rel']))
    count_p = count_true / len(dic_hlm[idx]['gt_rel'])
    if count_p>=1:
        print(idx)
        draw_image(dic_hlm[idx]['image_path'], dic_hlm[idx])
        draw_image(dic[idx]['image_path'], dic[idx])

print('end')