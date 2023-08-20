# Spatial Shift ViT

S²-ViT is a hierarchical vision transformer with shifted window attention.
In contrast to Swin, the shift operation used is based on S²-MLP, which shifts in all four directions simultaneously, and does not use the roll or unroll operation.
Additionally, it leverages the patch embedding and positional encoding methods from Twins-SVT, and StarReLU from MetaFormer.

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

## Acknowledgements

lucidrains for his excellent work, including [vit-pytorch](https://github.com/lucidrains/vit-pytorch), [x-transformers](https://github.com/lucidrains/x-transformers), and his discovery of [shared key / value attention](https://github.com/lucidrains/PaLM-pytorch/blob/7164d13d5a831647edb5838544017f387130f987/palm_pytorch/palm_lite.py#L142C8-L142C8).

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
@article{Liu2021SwinTV,
  title={Swin Transformer V2: Scaling Up Capacity and Resolution},
  author={Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
  journal={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
  pages={11999-12009},
  url={https://api.semanticscholar.org/CorpusID:244346076}
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

```bibtex
@article{Yu2022MetaFormerBF,
  title={MetaFormer Baselines for Vision},
  author={Weihao Yu and Chenyang Si and Pan Zhou and Mi Luo and Yichen Zhou and Jiashi Feng and Shuicheng Yan and Xinchao Wang},
  journal={ArXiv},
  year={2022},
  volume={abs/2210.13452},
  url={https://api.semanticscholar.org/CorpusID:253098429}
}
```

```bibtex
@article{Touvron2022ThreeTE,
  title={Three things everyone should know about Vision Transformers},
  author={Hugo Touvron and Matthieu Cord and Alaaeldin El-Nouby and Jakob Verbeek and Herv'e J'egou},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.09795},
  url={https://api.semanticscholar.org/CorpusID:247594673}
}
```

```bibtex
@article{Chowdhery2022PaLMSL,
  title={PaLM: Scaling Language Modeling with Pathways},
  author={Aakanksha Chowdhery and Sharan Narang and Jacob Devlin and Maarten Bosma and Gaurav Mishra and Adam Roberts and Paul Barham and Hyung Won Chung and Charles Sutton and Sebastian Gehrmann and Parker Schuh and Kensen Shi and Sasha Tsvyashchenko and Joshua Maynez and Abhishek Rao and Parker Barnes and Yi Tay and Noam M. Shazeer and Vinodkumar Prabhakaran and Emily Reif and Nan Du and Benton C. Hutchinson and Reiner Pope and James Bradbury and Jacob Austin and Michael Isard and Guy Gur-Ari and Pengcheng Yin and Toju Duke and Anselm Levskaya and Sanjay Ghemawat and Sunipa Dev and Henryk Michalewski and Xavier Garc{\'i}a and Vedant Misra and Kevin Robinson and Liam Fedus and Denny Zhou and Daphne Ippolito and David Luan and Hyeontaek Lim and Barret Zoph and Alexander Spiridonov and Ryan Sepassi and David Dohan and Shivani Agrawal and Mark Omernick and Andrew M. Dai and Thanumalayan Sankaranarayana Pillai and Marie Pellat and Aitor Lewkowycz and Erica Moreira and Rewon Child and Oleksandr Polozov and Katherine Lee and Zongwei Zhou and Xuezhi Wang and Brennan Saeta and Mark D{\'i}az and Orhan Firat and Michele Catasta and Jason Wei and Kathleen S. Meier-Hellstern and Douglas Eck and Jeff Dean and Slav Petrov and Noah Fiedel},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.02311},
  url={https://api.semanticscholar.org/CorpusID:247951931}
}
```

```bibtex
@article{Bondarenko2023QuantizableTR,
  title={Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing},
  author={Yelysei Bondarenko and Markus Nagel and Tijmen Blankevoort},
  journal={ArXiv},
  year={2023},
  volume={abs/2306.12929},
  url={https://api.semanticscholar.org/CorpusID:259224568}
}
```
