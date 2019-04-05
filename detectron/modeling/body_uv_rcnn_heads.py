# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core

from detectron.core.config import cfg

import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils
import copy
# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #

def add_body_uv_outputs(model, blob_in, dim, pref=''):
    ####
    roi_res = cfg.BODY_UV_RCNN.MULTI_LEVEL_RES
    #k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
    #k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
    k_max = max(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT)
    k_min = min(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT)
    num_level_feat = len(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT)
    if cfg.BODY_UV_RCNN.INTERMEDIA_SUPER:
        for lvl in range(k_min, k_max + 1):
            if num_level_feat == 4:
                if lvl == 3 or lvl == 5:
                    model.net.Sum(['bl_fcn_res_' + str(lvl), 'bl_fcn_res_' + str(lvl-1)],
                    'bl_fcn_res_sum_'+str(lvl)+'_'+str(lvl-1))
                    person_mask_inter = model.Conv(
                        'bl_fcn_res_sum_'+str(lvl)+'_'+str(lvl-1),
                        'person_mask_inter_'+str(lvl),
                        dim,
                        2,
                        3,
                        stride=1,
                        pad=1,
                        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                        bias_init=('ConstantFill', {'value': 0.})
                    )
                    Ann_Index_inter = model.Conv(
                        'bl_fcn_res_sum_'+str(lvl)+'_'+str(lvl-1),
                        'AnnIndex_inter_'+str(lvl),
                        dim,
                        cfg.BODY_UV_RCNN.NUM_BODY_PARTS + 1,
                        3,
                        stride=1,
                        pad=1,
                        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                        bias_init=('ConstantFill', {'value': 0.})
                    )
                    if not cfg.BODY_UV_RCNN.ONLY_PARTSEG:
                        blob_Index_inter  = model.Conv(
                           'bl_fcn_res_sum_'+str(lvl)+'_'+str(lvl-1),
                            'Index_UV_inter_'+str(lvl),
                            dim,
                            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
                            3,
                            stride=1,
                            pad=1,
                            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                            bias_init=('ConstantFill', {'value': 0.})
                        )
                        blob_U_inter = model.Conv(
                            'bl_fcn_res_sum_'+str(lvl)+'_'+str(lvl-1),
                            'U_estimated_inter_'+str(lvl),
                            dim,
                            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
                            3,
                            stride=1,
                            pad=1,
                            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                            bias_init=('ConstantFill', {'value': 0.})
                        )
                        blob_V_inter  = model.Conv(
                            'bl_fcn_res_sum_'+str(lvl)+'_'+str(lvl-1),
                            'V_estimated_inter_'+str(lvl),
                            dim,
                            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
                            3,
                            stride=1,
                            pad=1,
                            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                            bias_init=('ConstantFill', {'value': 0.})
                        )
            else:
                print('Not support feature levels: ',num_level_feat)
            '''
            person_mask_inter = model.Conv(
            'bl_fcn_res_' + str(lvl),
            'person_mask_inter_'+str(lvl),
            dim,
            2,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
            )
            Ann_Index_inter = model.Conv(
                'bl_fcn_res_' + str(lvl),
                'AnnIndex_inter_'+str(lvl),
                dim,
                cfg.BODY_UV_RCNN.NUM_BODY_PARTS + 1,
                3,
                stride=1,
                pad=1,
                weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                bias_init=('ConstantFill', {'value': 0.})
            )
            if not cfg.BODY_UV_RCNN.ONLY_PARTSEG:
            #if not ONLY_PARTSEG:
                blob_Index_inter  = model.Conv(
                    'bl_fcn_res_' + str(lvl),
                    'Index_UV_inter_'+str(lvl),
                    dim,
                    cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
                    3,
                    stride=1,
                    pad=1,
                    weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                    bias_init=('ConstantFill', {'value': 0.})
                )
                blob_U_inter = model.Conv(
                    'bl_fcn_res_' + str(lvl),
                    'U_estimated_inter_'+str(lvl),
                    dim,
                    cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
                    3,
                    stride=1,
                    pad=1,
                    weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                    bias_init=('ConstantFill', {'value': 0.})
                )
                blob_V_inter  = model.Conv(
                    'bl_fcn_res_' + str(lvl),
                    'V_estimated_inter_'+str(lvl),
                    dim,
                    cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
                    3,
                    stride=1,
                    pad=1,
                    weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                    bias_init=('ConstantFill', {'value': 0.})
                )
                '''
    blob_person_mask  = model.Conv(
        blob_in,
        'person_mask'+pref,
        dim,
        2,
        3,
        stride=1,
        pad=1,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.})
    )

    blob_Ann_Index  = model.Conv(
        blob_in,
        'AnnIndex'+pref,
        dim,
        cfg.BODY_UV_RCNN.NUM_BODY_PARTS + 1,
        3,
        stride=1,
        pad=1,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    blob_U = None
    blob_V = None
    blob_Index = None
    if not cfg.BODY_UV_RCNN.ONLY_PARTSEG:
    #if not ONLY_PARTSEG:
        blob_Index  = model.Conv(
            blob_in,
            'Index_UV'+pref,
            dim,
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        blob_U = model.Conv(
            blob_in,
            'U_estimated'+pref,
            dim,
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        blob_V  = model.Conv(
            blob_in,
            'V_estimated'+pref,
            dim,
            cfg.BODY_UV_RCNN.NUM_PATCHES + 1,
            3,
            stride=1,
            pad=1,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        #blob_Ann_Index = model.BilinearInterpolation('AnnIndex_lowres'+pref, 'AnnIndex'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
        #blob_Index = model.BilinearInterpolation('Index_UV_lowres'+pref, 'Index_UV'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
        #blob_U = model.BilinearInterpolation('U_lowres'+pref, 'U_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
        #blob_V = model.BilinearInterpolation('V_lowres'+pref, 'V_estimated'+pref,  cfg.BODY_UV_RCNN.NUM_PATCHES+1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1, cfg.BODY_UV_RCNN.UP_SCALE)
        ###
        return blob_U,blob_V,blob_Index,blob_Ann_Index
    else:
        return blob_Ann_Index


def add_body_uv_losses(model, pref=''):

    if not cfg.BODY_UV_RCNN.ONLY_PARTSEG:
    #if not ONLY_PARTSEG:
        ## Reshape for GT blobs.
        model.net.Reshape( ['body_uv_X_points'], ['X_points_reshaped'+pref, 'X_points_shape'+pref],  shape=( -1 ,1 ) )
        model.net.Reshape( ['body_uv_Y_points'], ['Y_points_reshaped'+pref, 'Y_points_shape'+pref],  shape=( -1 ,1 ) )
        model.net.Reshape( ['body_uv_I_points'], ['I_points_reshaped'+pref, 'I_points_shape'+pref],  shape=( -1 ,1 ) )
        model.net.Reshape( ['body_uv_Ind_points'], ['Ind_points_reshaped'+pref, 'Ind_points_shape'+pref],  shape=( -1 ,1 ) )
        ## Concat Ind,x,y to get Coordinates blob.
        model.net.Concat( ['Ind_points_reshaped'+pref,'X_points_reshaped'+pref, \
                           'Y_points_reshaped'+pref],['Coordinates'+pref,'Coordinate_Shapes'+pref ], axis = 1 )
        ##
        ### Now reshape UV blobs, such that they are 1x1x(196*NumSamples)xNUM_PATCHES
        ## U blob to
        ##
        model.net.Reshape(['body_uv_U_points'], \
                          ['U_points_reshaped'+pref, 'U_points_old_shape'+pref],\
                          shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
        model.net.Transpose(['U_points_reshaped'+pref] ,['U_points_reshaped_transpose'+pref],axes=(0,2,1) )
        model.net.Reshape(['U_points_reshaped_transpose'+pref], \
                          ['U_points'+pref, 'U_points_old_shape2'+pref], \
                          shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
        ## V blob
        ##
        model.net.Reshape(['body_uv_V_points'], \
                          ['V_points_reshaped'+pref, 'V_points_old_shape'+pref],\
                          shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
        model.net.Transpose(['V_points_reshaped'+pref] ,['V_points_reshaped_transpose'+pref],axes=(0,2,1) )
        model.net.Reshape(['V_points_reshaped_transpose'+pref], \
                          ['V_points'+pref, 'V_points_old_shape2'+pref], \
                          shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))
        ###
        ## UV weights blob
        ##
        model.net.Reshape(['body_uv_point_weights'], \
                          ['Uv_point_weights_reshaped'+pref, 'Uv_point_weights_old_shape'+pref],\
                          shape=(-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1,196))
        model.net.Transpose(['Uv_point_weights_reshaped'+pref] ,['Uv_point_weights_reshaped_transpose'+pref],axes=(0,2,1) )
        model.net.Reshape(['Uv_point_weights_reshaped_transpose'+pref], \
                          ['Uv_point_weights'+pref, 'Uv_point_weights_old_shape2'+pref], \
                          shape=(1,1,-1,cfg.BODY_UV_RCNN.NUM_PATCHES+1))

        #####################
        ###  Pool IUV for points via bilinear interpolation.
        model.PoolPointsInterp(['U_estimated','Coordinates'+pref], ['interp_U'+pref])
        model.PoolPointsInterp(['V_estimated','Coordinates'+pref], ['interp_V'+pref])
        model.PoolPointsInterp(['Index_UV'+pref,'Coordinates'+pref], ['interp_Index_UV'+pref])

        ## Reshape interpolated UV coordinates to apply the loss.

        model.net.Reshape(['interp_U'+pref], \
                          ['interp_U_reshaped'+pref, 'interp_U_shape'+pref],\
                          shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))

        model.net.Reshape(['interp_V'+pref], \
                          ['interp_V_reshaped'+pref, 'interp_V_shape'+pref],\
                          shape=(1, 1, -1 , cfg.BODY_UV_RCNN.NUM_PATCHES+1))
        ###


        ###
        model.net.Cast( ['I_points_reshaped'+pref], ['I_points_reshaped_int'+pref], to=core.DataType.INT32)

        ## Point Patch Index Loss.
        probs_IndexUVPoints, loss_IndexUVPoints = model.net.SoftmaxWithLoss(\
                              ['interp_Index_UV'+pref,'I_points_reshaped_int'+pref],\
                              ['probs_IndexUVPoints'+pref,'loss_IndexUVPoints'+pref], \
                              scale=cfg.BODY_UV_RCNN.PART_WEIGHTS / cfg.NUM_GPUS, spatial=0)
        ## U and V point losses.
        loss_Upoints = model.net.SmoothL1Loss( \
                              ['interp_U_reshaped'+pref, 'U_points'+pref, \
                                   'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                              'loss_Upoints'+pref, \
                                scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS  / cfg.NUM_GPUS)

        loss_Vpoints = model.net.SmoothL1Loss( \
                              ['interp_V_reshaped'+pref, 'V_points'+pref, \
                                   'Uv_point_weights'+pref, 'Uv_point_weights'+pref], \
                              'loss_Vpoints'+pref, scale=cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)

    ### Do the actual labels here !!!!
    model.net.Reshape( ['body_uv_ann_labels'],    \
                      ['body_uv_ann_labels_reshaped'   +pref, 'body_uv_ann_labels_old_shape'+pref], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))

    model.net.Reshape( ['body_uv_ann_weights'],   \
                      ['body_uv_ann_weights_reshaped'   +pref, 'body_uv_ann_weights_old_shape'+pref], \
                      shape=( -1 , cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))
    ### Now add the actual losses
    ## The mask segmentation loss (dense)
    probs_seg_AnnIndex, loss_seg_AnnIndex = model.net.SpatialSoftmaxWithLoss( \
                          ['AnnIndex'+pref, 'body_uv_ann_labels_reshaped'+pref,'body_uv_ann_weights_reshaped'+pref],\
                          ['probs_seg_AnnIndex'+pref,'loss_seg_AnnIndex'+pref], \
                           scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)

    # add person mask loss by gyy & wxh
    model.net.Reshape( ['body_mask_labels'],    \
                      ['body_mask_labels_reshaped', 'body_uv_mask_labels_old_shape'], \
                      shape=(-1, cfg.BODY_UV_RCNN.HEATMAP_SIZE , cfg.BODY_UV_RCNN.HEATMAP_SIZE))

    # mask loss
    probs_mask, loss_mask = model.net.SpatialSoftmaxWithLoss( \
                          ['person_mask', 'body_mask_labels_reshaped','body_uv_ann_weights_reshaped'],\
                          ['probs_mask','loss_mask'], \
                           scale=cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
    # intermedia loss
    #k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
    #k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
    k_max = max(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT)
    k_min = min(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT)
    probs_mask_inter = []
    loss_mask_inter = []
    loss_mask_inter_str = []
    probs_seg_AnnIndex_inter = []
    loss_seg_AnnIndex_inter = []
    loss_seg_AnnIndex_str = []
    probs_IndexUVPoints_inter = []
    loss_IndexUVPoints_inter = []
    loss_IndexUVPoints_str = []
    loss_Upoints_inter = []
    loss_Upoints_inter_str = []
    loss_Vpoints_inter = []
    loss_Vpoints_inter_str = []
    num_level_feat = len(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT)
    if cfg.BODY_UV_RCNN.INTERMEDIA_SUPER:
        for lvl in range(k_min, k_max + 1):
            if (num_level_feat == 4 and (lvl == 3 or lvl == 5)) or num_level_feat == 2 :
                inter_weights = cfg.BODY_UV_RCNN.INTER_WEIGHTS
                probs_mask_i, loss_mask_i = model.net.SpatialSoftmaxWithLoss( \
                    ['person_mask_inter_'+str(lvl), 'body_mask_labels_reshaped', 'body_uv_ann_weights_reshaped'], \
                    ['probs_mask_inter_'+str(lvl), 'loss_mask_inter_'+str(lvl)], \
                    scale=inter_weights * cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
                probs_mask_inter.append(probs_mask_i)
                loss_mask_inter.append(loss_mask_i)
                loss_mask_inter_str.append('loss_mask_inter_'+str(lvl))
                probs_seg_AnnIndex_i, loss_seg_AnnIndex_i = model.net.SpatialSoftmaxWithLoss( \
                    ['AnnIndex_inter_'+str(lvl), 'body_uv_ann_labels_reshaped' + pref, 'body_uv_ann_weights_reshaped' + pref], \
                    ['probs_seg_AnnIndex_inter_'+str(lvl), 'loss_seg_AnnIndex_inter_'+str(lvl)], \
                    scale=inter_weights * cfg.BODY_UV_RCNN.INDEX_WEIGHTS / cfg.NUM_GPUS)
                probs_seg_AnnIndex_inter.append(probs_seg_AnnIndex_i)
                loss_seg_AnnIndex_inter.append(loss_seg_AnnIndex_i)
                loss_seg_AnnIndex_str.append('loss_seg_AnnIndex_inter_'+str(lvl))
                if not cfg.BODY_UV_RCNN.ONLY_PARTSEG:
                    ###  Pool IUV for points via bilinear interpolation.
                    model.PoolPointsInterp(['U_estimated_inter_'+str(lvl), 'Coordinates' + pref], ['interp_U_inter_'+str(lvl)])
                    model.PoolPointsInterp(['V_estimated_inter_'+str(lvl), 'Coordinates' + pref], ['interp_V_inter_'+str(lvl)])
                    model.PoolPointsInterp(['Index_UV_inter_'+str(lvl), 'Coordinates' + pref], ['interp_Index_UV_inter_'+str(lvl)])
                    model.net.Reshape(['interp_U_inter_'+str(lvl)],
                                      ['interp_U_reshaped_inter_'+str(lvl), 'interp_U_shape_inter_'+str(lvl)],
                                      shape=(1, 1, -1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1))

                    model.net.Reshape(['interp_V_inter_'+str(lvl)],
                                      ['interp_V_reshaped_inter_'+str(lvl), 'interp_V_shape_inter_'+str(lvl)],
                                      shape=(1, 1, -1, cfg.BODY_UV_RCNN.NUM_PATCHES + 1))

                    ## Point Patch Index Loss.
                    probs_IndexUVPoints_i, loss_IndexUVPoints_i = model.net.SoftmaxWithLoss(
                        ['interp_Index_UV_inter_'+str(lvl), 'I_points_reshaped_int' + pref],
                        ['probs_IndexUVPoints_inter_'+str(lvl), 'loss_IndexUVPoints_inter_'+str(lvl)],
                        scale=inter_weights * cfg.BODY_UV_RCNN.PART_WEIGHTS / cfg.NUM_GPUS, spatial=0)
                    probs_IndexUVPoints_inter.append(probs_IndexUVPoints_i)
                    loss_IndexUVPoints_inter.append(loss_IndexUVPoints_i)
                    loss_IndexUVPoints_str.append('loss_IndexUVPoints_inter_' + str(lvl))
                    ## U and V point losses.
                    loss_Upoints_i = model.net.SmoothL1Loss(
                        ['interp_U_reshaped_inter_'+str(lvl), 'U_points' + pref,
                         'Uv_point_weights' + pref, 'Uv_point_weights' + pref],
                        'loss_Upoints_inter_'+str(lvl),
                        scale=inter_weights * cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)
                    loss_Upoints_inter.append(loss_Upoints_i)
                    loss_Upoints_inter_str.append('loss_Upoints_inter_'+str(lvl))
                    loss_Vpoints_i = model.net.SmoothL1Loss(
                        ['interp_V_reshaped_inter_'+str(lvl), 'V_points' + pref,
                         'Uv_point_weights' + pref, 'Uv_point_weights' + pref],
                        'loss_Vpoints_inter_'+str(lvl),
                        scale=inter_weights * cfg.BODY_UV_RCNN.POINT_REGRESSION_WEIGHTS / cfg.NUM_GPUS)
                    loss_Vpoints_inter.append(loss_Vpoints_i)
                    loss_Vpoints_inter_str.append('loss_Vpoints_inter_'+str(lvl))

    if not cfg.BODY_UV_RCNN.ONLY_PARTSEG:
    #if not ONLY_PARTSEG:
        ## Add the losses.
        loss_gradients = blob_utils.get_loss_gradients(model, \
                           [ loss_Upoints, loss_Vpoints, loss_seg_AnnIndex, loss_IndexUVPoints, \
                             loss_mask]+loss_mask_inter+loss_seg_AnnIndex_inter+loss_IndexUVPoints_inter+ \
                             loss_Upoints_inter+loss_Vpoints_inter)
        model.losses = list(set(model.losses + \
                           ['loss_Upoints'+pref , 'loss_Vpoints'+pref , \
                            'loss_seg_AnnIndex'+pref ,'loss_IndexUVPoints'+pref, 'loss_mask']+loss_mask_inter_str + \
                            loss_seg_AnnIndex_str + loss_IndexUVPoints_str + \
                            loss_Upoints_inter_str + loss_Vpoints_inter_str ))
    else:
        ## Add the losses.
        loss_gradients = blob_utils.get_loss_gradients(model, \
                           [ loss_seg_AnnIndex, \
                             loss_mask]+loss_mask_inter+loss_seg_AnnIndex_inter)
        model.losses = list(set(model.losses + \
                           ['loss_seg_AnnIndex'+pref ,'loss_mask']+loss_mask_inter_str+loss_seg_AnnIndex_str))

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Body UV heads
# ---------------------------------------------------------------------------- #

def add_ResNet_roi_conv5_head_for_bodyUV(
        model, blob_in, dim_in, spatial_scale
):
    """Add a ResNet "conv5" / "stage5" head for body UV prediction."""
    model.RoIFeatureTransform(
        blob_in, '_[body_uv]_pool5',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
    # Using the prefix '_[body_uv]_' to 'res5' enables initializing the head's
    # parameters using pretrained 'res5' parameters if given (see
    # utils.net.initialize_from_weights_file)
    s, dim_in = ResNet.add_stage(
        model,
        '_[body_uv]_res5',
        '_[body_uv]_pool5',
        3,
        dim_in,
        2048,
        512,
        cfg.BODY_UV_RCNN.DILATION,
        stride_init=int(cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION / 7)
    )
    return s, 2048

def add_roi_body_uv_head_v1convX(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in,
        '_[body_uv]_roi_feat',
        blob_rois='body_uv_rois',
        method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.BODY_UV_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current,
            'body_conv_fcn' + str(i + 1),
            dim_in,
            hidden_dim,
            kernel_size,
            stride=1,
            pad=pad_size,
            weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim

    return current, hidden_dim


def add_roi_body_uv_head_v1convX_parallroires(model, blob_in, dim_in, spatial_scale):
    """v1convX design: X * (conv)."""
    hidden_dim = cfg.BODY_UV_RCNN.CONV_HEAD_DIM
    kernel_size = cfg.BODY_UV_RCNN.CONV_HEAD_KERNEL
    roi_res = cfg.BODY_UV_RCNN.MULTI_LEVEL_RES
    pad_size = kernel_size // 2
    multi_level_bl_list = []
    #k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
    #k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
    k_max = max(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT)
    k_min = min(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT)
    k_max_b = cfg.FPN.ROI_MAX_LEVEL
    #assert len(blob_in) == k_max - k_min + 1
    for lvl in range(k_min, k_max + 1):
        roi_res_i = roi_res[k_max - lvl]
        bl_in = blob_in[k_max_b - lvl]
        spatial_scale_i = spatial_scale[k_max_b - lvl]
        #bl_rois = 'body_uv_rois' + '_fpn' + str(lvl)
        current = model.RoIFeatureTransform(
            bl_in,
            'uv_roi_feat'+'_fpn'+str(lvl),
            blob_rois='body_uv_rois',
            method=cfg.BODY_UV_RCNN.ROI_XFORM_METHOD,
            resolution=roi_res_i,
            sampling_ratio=cfg.BODY_UV_RCNN.ROI_XFORM_SAMPLING_RATIO,
            spatial_scale=spatial_scale_i
        )
        for i in range(cfg.BODY_UV_RCNN.NUM_STACKED_CONVS):
            if lvl - k_min == 0:
                current = model.Conv(
                    current,
                    'uv_conv_fcn' + str(i + 1) + '_rs'+str(lvl),
                    dim_in,
                    hidden_dim,
                    kernel_size,
                    stride=1,
                    pad=pad_size,
                    weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
                    bias_init=('ConstantFill', {'value': 0.})
                )
            else:
                current = model.ConvShared(
                    current,
                    'uv_conv_fcn' + str(i + 1)+'_rs'+str(lvl),
                    dim_in,
                    hidden_dim,
                    kernel_size,
                    stride=1,
                    pad=pad_size,
                    weight='uv_conv_fcn' + str(i + 1) + '_rs'+str(k_min)+'_w',
                    bias='uv_conv_fcn' + str(i + 1) + '_rs'+str(k_min)+'_b'
                )
            current = model.Relu(current, current)
            dim_in = hidden_dim
        bl_ouput = 'bl_fcn_res_' + str(lvl)
        current_bilinear = model.BilinearInterpolation(current,
                                              bl_ouput,
                                              hidden_dim,
                                              hidden_dim,
                                              int(cfg.BODY_UV_RCNN.HEATMAP_SIZE/roi_res_i))
        multi_level_bl_list.append(bl_ouput)
    current_concat, _ = model.net.Concat(
    multi_level_bl_list, ['uv_head_concat','uv_head_concat_dims'],
    axis=1
    )
    current = model.Conv(
        current_concat,
        'uvtail_conv1_fcn',
        hidden_dim * len(cfg.BODY_UV_RCNN.MULTI_LEVEL_FEAT),
        int(hidden_dim),
        1,
        stride=1,
        pad=0,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    current = model.Relu(current, current)
    current = model.Conv(
        current,
        'uvtail_conv3_fcn',
        int(hidden_dim),
        int(hidden_dim),
        3,
        stride=1,
        pad=pad_size,
        weight_init=(cfg.BODY_UV_RCNN.CONV_INIT, {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.})
    )
    current = model.Relu(current, current)
    return current, int(hidden_dim)

