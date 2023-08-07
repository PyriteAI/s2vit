# Spatial Shift ViT

S²-ViT is a hierarchical vision transformer with shifted window attention.
In contrast to Swin, the shift operation used is based on S²-MLP, which shifts in all four directions simultaneously, and does not use the roll or unroll operation.
Additionally, in leverages the patch embedding and positional encoding methods from Twins-SVT.

## Prerequisites

- Python 3.10+
- PyTorch 2.0+

## Installation

```sh
pip install s2vit
```

## Usage

```python
import torch
from s2vit import S2ViT

vit = S2ViT(
    depths=(2, 2, 6, 2),
    dims=(64, 128, 160, 320),
    global_pool=True
    num_classes=1000,
)

img = torch.randn(1, 3, 256, 256)
vit(img) # (1, 1000)
```

## Citations

```bibtex
@article{Yu2021S2MLPSM,
  title={S2-MLP: Spatial-Shift MLP Architecture for Vision},
  author={Tan Yu and Xu Li and Yunfeng Cai and Mingming Sun and Ping Li},
  journal={2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2021},
  pages={3615-3624},
  url={https://api.semanticscholar.org/CorpusID:235422259}
}
```

```bibtex
@article{Liu2021SwinTH,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Ze Liu and Yutong Lin and Yue Cao and Han Hu and Yixuan Wei and Zheng Zhang and Stephen Lin and Baining Guo},
  journal={2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021},
  pages={9992-10002},
  url={https://api.semanticscholar.org/CorpusID:232352874}
}
```

```bibtex
@inproceedings{Chu2021TwinsRT,
  title={Twins: Revisiting the Design of Spatial Attention in Vision Transformers},
  author={Xiangxiang Chu and Zhi Tian and Yuqing Wang and Bo Zhang and Haibing Ren and Xiaolin Wei and Huaxia Xia and Chunhua Shen},
  booktitle={Neural Information Processing Systems},
  year={2021},
  url={https://api.semanticscholar.org/CorpusID:234364557}
}
```
