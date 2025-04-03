# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .semi_base import SemiBase3DSegmentor


@MODELS.register_module()
class HiLoTs(SemiBase3DSegmentor):
    def __init__(self,
                 segmentor_student: ConfigType,
                 segmentor_teacher: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 loss_mse: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(HiLoTs, self).__init__(
            segmentor_student=segmentor_student,
            segmentor_teacher=segmentor_teacher,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        if loss_mse is not None:
            self.loss_mse = MODELS.build(loss_mse)
        else:
            self.loss_mse = None

    def loss(self, multi_batch_inputs: Dict[str, dict],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()

        # sup loss
        logits_sup_s, losses_sup = self.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup'])
        losses.update(**losses_sup)

        
        # generate pseudo instances for unlabeled data
        logits_unsup_t, pseudo_data_samples = self.get_pseudo_instances(
            multi_batch_inputs['unsup'], multi_batch_data_samples['unsup']
        )

        # mt loss
        if self.loss_mse is not None:
            logits_sup_t = self.teacher(multi_batch_inputs['sup'], multi_batch_data_samples['sup'], mode='tensor')
            logits_unsup_s = self.student(multi_batch_inputs['unsup'], multi_batch_data_samples['unsup'], mode='tensor')

            logits_s = torch.cat([logits_sup_s['logits'], logits_unsup_s['logits']])
            logits_t = torch.cat([logits_sup_t['logits'], logits_unsup_t['logits']])

            logits_s = F.softmax(logits_s, dim=1)
            logits_t = F.softmax(logits_t, dim=1)

            losses['loss_mt'] = self.loss_mse(logits_s, logits_t.detach())  

        return losses

    def transform(
            self, points_sup: Tensor, points_unsup: Tensor, labels: Tensor,
            pseudo_labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        pitch_angle_down = self.semi_train_cfg.pitch_angles[0] / 180 * np.pi
        pitch_angle_up = self.semi_train_cfg.pitch_angles[1] / 180 * np.pi

        rho_sup = torch.sqrt(points_sup[:, 0]**2 + points_sup[:, 1]**2)
        pitch_sup = torch.atan2(points_sup[:, 2], rho_sup)
        pitch_sup = torch.clamp(pitch_sup, pitch_angle_down + 1e-5,
                                pitch_angle_up - 1e-5)

        rho_unsup = torch.sqrt(points_unsup[:, 0]**2 + points_unsup[:, 1]**2)
        pitch_unsup = torch.atan2(points_unsup[:, 2], rho_unsup)
        pitch_unsup = torch.clamp(pitch_unsup, pitch_angle_down + 1e-5,
                                  pitch_angle_up - 1e-5)

        num_areas = np.random.choice(self.semi_train_cfg.num_areas, size=1)[0]
        angle_list = np.linspace(pitch_angle_up, pitch_angle_down,
                                 num_areas + 1)
        points_mix1 = []
        points_mix2 = []

        labels_mix1 = []
        labels_mix2 = []

        for i in range(num_areas):
            start_angle = angle_list[i + 1]
            end_angle = angle_list[i]
            idx_sup = (pitch_sup > start_angle) & (pitch_sup <= end_angle)
            idx_unsup = (pitch_unsup > start_angle) & (
                pitch_unsup <= end_angle)
            if i % 2 == 0:
                points_mix1.append(points_sup[idx_sup])
                labels_mix1.append(labels[idx_sup])
                points_mix2.append(points_unsup[idx_unsup])
                labels_mix2.append(pseudo_labels[idx_unsup])
            else:
                points_mix1.append(points_unsup[idx_unsup])
                labels_mix1.append(pseudo_labels[idx_unsup])
                points_mix2.append(points_sup[idx_sup])
                labels_mix2.append(labels[idx_sup])

        points_mix1 = torch.cat(points_mix1)
        points_mix2 = torch.cat(points_mix2)
        labels_mix1 = torch.cat(labels_mix1)
        labels_mix2 = torch.cat(labels_mix2)

        return points_mix1, points_mix2, labels_mix1, labels_mix2

