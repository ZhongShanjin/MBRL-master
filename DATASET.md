# Download:

**For VG Dataset**

1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. Download [image_data.json](https://thunlp.oss-cn-qingdao.aliyuncs.com/vg/image_data.json). Put the file at `datasets/vg`.
3. Download the VG-50 data from [GoogleDrive](https://drive.google.com/file/d/1JWa9DAxIlUc5wZsL6QM_29awKIGh7WrK/view?usp=sharing). Extract these files to `datasets/` and forms the file structure `datasets/vg/50`.

### For GQA Dataset:

1. Download the GQA images [Full (20.3 Gb)](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip). Extract these images to the file `datasets/gqa/images`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`.
2. In order to achieve a representative split like VG150, we manually clean up a substantial fraction of annotations that have poor-quality or ambiguous meanings, and then select Top-200 object classes as well as Top-100 predicate classes by their frequency, thus establishing the GQA200 split. You can download the annotation file from [this link](https://1drv.ms/u/s!AjK8-t5JiDT1kwwKFbdBB3ZU3c49?e=06qeZc), and put all three files to `datasets/gqa/`.
