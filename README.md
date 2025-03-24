
Implementation of **[HiLoTs: High-Low Temporal Sensitive Representation Learning for Semi-Supervised LiDAR Segmentation in Autonomous Driving](#)**

## Motivation

In driving experience, we observe a phenomenon: objects closer to the vehicle, such as roads and cars, tend to have stable categories and shapes as the vehicle moves, while distant objects, such as pedestrians, guardrails, plants, and buildings, exhibit significant variations in category and shape. 

<p align="center"><img src="./imgs/motivation.png" width="400"/></p>
<p style="text-align:center;"><i>Figure 1. Motivation</i></p>

## Methods

Our segmentation model involves three stages. During voxelization, cylindrical voxelization is applied to transform unordered points into volumetric grids, followed by a spatial feature extraction backbone. Then, HiLoTs processes the labeled and unlabeled cylindrical features through a student-teacher framework. It also integrates the attention map from HiLoTs embedding unit (HEU) to produce voxel-level segmentation maps. Finally, a point-wise refinement network is utilized to obtain point-level segmentation results. 

HEU consists of High Temporal Sensitive Flow (HTSF) and Low Temporal Sensitive Flow (LTSF). The HTSF focuses on regions where distant objects experience significant changes in category and shape, while the LTSF focuses on nearby regions where object categories and shapes remain relatively stable. Furthermore, the features from HTSF and LTSF are fused and interact through a cross-attention mechanism.

<p align="center"><img src="./imgs/methods.png" width="700"/></p>
<p style="text-align:center;"><i>Figure 2. Overall Architecture</i></p>

## Code Implementation

Comming soon
