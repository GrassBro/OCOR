# OCOR
This is a reproduction of the CVPR'22 paper, Bi-directional Object-context Prioritization Learning for Saliency Ranking.

## Installation
step 1: install pytorch and mmcv==1.3.9 referring to [this](https://github.com/open-mmlab/mmcv).

step 2: clone this repository and execute:
```bash
python setup.py develop
```

step 3: install apex following [this](https://github.com/NVIDIA/apex). (optional)

## Dataset
Download COCO-style JSON files for the ASSR dataset from: https://pan.baidu.com/s/1XvYwBCn3sc6lAlJbJ94gUQ (pwd: ocor) 

## Training

```bash
bash tools/dist_train.sh configs/ocor/ocor_swin...py num_gpus 
```

## Inference & Evaluation
```bash
python inference.py
```

```bash
python evaluate_SOR.py
```

## Pre-trained Model
Download it from our Baidu cloud: https://pan.baidu.com/s/15tINLiVC8kPQm6xqxyaJlA (pwd: ocor), then use it for fine-tuning, and inference.


## Citation
```BibTeX
@inproceedings{tian2022bi,
  title={Bi-directional object-context prioritization learning for saliency ranking},
  author={Tian, Xin and Xu, Ke and Yang, Xin and Du, Lin and Yin, Baocai and Lau, Rynson WH},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5882--5891},
  year={2022}
}
```
