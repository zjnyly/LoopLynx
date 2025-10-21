# This is repo for "LoopLynx: A Scalable Dataflow Architecture for Efficient LLM Inference"
We provide xclbin files for evaluating per-token latency of GPT-2(345M). The target platfrom is Alveo U50lv. 

# Note
We have extended this work with larger model support and more fluently hardware pipeline. Please refer to https://github.com/zjnyly/TeraFly to view our open sourced project.

# Cite
We appreciate for your interest in our work!
```
@inproceedings{LoopLynx,
  author       = {Jianing Zheng and
                  Gang Chen},
  title        = {LoopLynx: {A} Scalable Dataflow Architecture for Efficient {LLM} Inference},
  booktitle    = {Design, Automation {\&} Test in Europe Conference, {DATE} 2025,
                  Lyon, France, March 31 - April 2, 2025},
  pages        = {1--7},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.23919/DATE64628.2025.10993078},
  doi          = {10.23919/DATE64628.2025.10993078},
  timestamp    = {Fri, 30 May 2025 14:55:24 +0200},
  biburl       = {https://dblp.org/rec/conf/date/ZhengC25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}


@ARTICLE{Terafly,
  author={Zheng, Jianing and Chen, Gang and Huang, Libo and Lou, Xin and Zheng, Wei-shi},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems}, 
  title={Terafly : A Multi-Node FPGA Based Accelerator Design for Efficient Cooperative Inference in LLMs}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Computer architecture;Computational modeling;Field programmable gate arrays;Quantization (signal);Hardware;Throughput;Integrated circuit modeling;Earth;Adaptation models;Vectors;FPGA;High-Level Synthesis;Large Language Models;Hardware Acceleration},
  doi={10.1109/TCAD.2025.3616078}}
