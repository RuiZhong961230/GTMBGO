# GTMBGO
Gene-targeting multiplayer battle game optimizer for large-scale global optimization via cooperative coevolution

## Abstract
This paper proposes an efficient variant of the multiplayer battle game optimizer (MBGO) named gene-targeting MBGO (GTMBGO). We simplify the original MBGO and introduce a well-performed gene-targeting search operator to strengthen its optimization performance. Comprehensive numerical experiments on CEC2017 and CEC2022 benchmark functions confirm the efficiency and effectiveness of GTMBGO compared with state-of-the-art optimizers, and the ablation experiments are also implemented to investigate the contribution of proposed strategies independently. Additionally, we extend the proposed GTMBGO to solve large-scale optimization problems (LSOPs). Since the existence of the curse of dimensionality, LSGO challenges the performance of optimizers severely. Inspired by the divide-and-conquer, the cooperative coevolution (CC) framework decomposes the LSOP and optimizes them alternatively, which provides a potential avenue to solve LSOPs. Based on the efficient recursive differential grouping (ERDG) decomposition method, we propose an enhanced version named ERDG with maximum volume (ERDG$k$) and incorporate it with GTMBGO to deal with LSOPs. The experimental results and statistical analyses on CEC2013 LSGO benchmark functions verify the competitiveness of GTMBGO-ERDG$k$ when compared with state-of-the-art methodologies specifically designed for LSOPs.

## Citation
@article{Zhong:24,  
  title={Gene Targeting Multiplayer Battle Game Optimizer for Large-scale Global Optimization via Cooperative Coevolution},  
  author={Rui Zhong and Jun Yu},  
  journal={Cluster Computing},  
  volume={27},  
  pages={12483–12508},  
  year={2024},  
  publisher={Springer},  
  doi = {https://doi.org/10.1007/s10586-024-04600-6 },  
}

## Datasets and Libraries
CEC benchmarks are provided by opfunu and cec2013lsgo libraries and engineering problems are provided by the enoppy library.
