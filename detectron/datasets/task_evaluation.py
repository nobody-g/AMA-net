# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Evaluation interface for supported tasks (box detection, instance
segmentation, keypoint detection, ...).


Results are stored in an OrderedDict with the following nested structure:

<dataset>:
  <task>:
    <metric>: <val>

<dataset> is any valid dataset (e.g., 'coco_2014_minival')
<task> is in ['box', 'mask', 'keypoint', 'box_proposal']
<metric> can be ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR@1000',
                 'ARs@1000', 'ARm@1000', 'ARl@1000', ...]
<val> is a floating point number
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pycocotools.coco import COCO
import detectron.datasets.dataset_catalog as dataset_catalog
from collections import OrderedDict
import logging
import os
import pprint
import glob
import numpy as np
import detectron.utils.parsing as parsing_utils
from detectron.core.config import cfg
from detectron.utils.logging import send_email
import detectron.datasets.cityscapes_json_dataset_evaluator \
    as cs_json_dataset_evaluator
import detectron.datasets.json_dataset_evaluator as json_dataset_evaluator
import detectron.datasets.voc_dataset_evaluator as voc_dataset_evaluator

logger = logging.getLogger(__name__)


def evaluate_all(
    dataset, all_boxes, all_segms, all_keyps, all_personmasks, all_parss, all_bodys,
    output_dir, use_matlab=False
):
    """Evaluate "all" tasks, where "all" includes box detection, instance
    segmentation, and keypoint detection.
    """
    all_results = evaluate_boxes(
        dataset, all_boxes, output_dir, use_matlab=use_matlab
    )
    #logger.info('Evaluating bounding boxes is done!')
    if cfg.MODEL.MASK_ON:
        results = evaluate_masks(dataset, all_boxes, all_segms, output_dir)
        all_results[dataset.name].update(results[dataset.name])
        logger.info('Evaluating segmentations is done!')
    if cfg.MODEL.KEYPOINTS_ON:
        results = evaluate_keypoints(dataset, all_boxes, all_keyps, output_dir)
        all_results[dataset.name].update(results[dataset.name])
        logger.info('Evaluating keypoints is done!')
    if cfg.MODEL.BODY_UV_ON:
        if cfg.BODY_UV_RCNN.ONLY_PARTSEG:
            results = evaluate_parsing(dataset, all_boxes, all_parss, output_dir)
            all_results[dataset.name].update(results[dataset.name])
            if all_personmasks is not None:
                results_personmask = evaluate_person_masks(dataset, all_boxes, all_personmasks, output_dir)
                all_results[dataset.name].update(results_personmask[dataset.name])
            logger.info('Evaluating parsing is done!')
        else:
            results = evaluate_body_uv(dataset, all_boxes, all_bodys, output_dir)
            all_results[dataset.name].update(results[dataset.name])
            results_partseg = evaluate_parsing(dataset, all_boxes, all_parss, output_dir)
            all_results[dataset.name].update(results_partseg[dataset.name])
            results_personmask = evaluate_person_masks(dataset, all_boxes, all_personmasks, output_dir)
            all_results[dataset.name].update(results_personmask[dataset.name])
            logger.info('Evaluating body uv is done!')

    return all_results


def evaluate_boxes(dataset, all_boxes, output_dir, use_matlab=False):
    """Evaluate bounding box detection."""
    logger.info('Evaluating detections')
    not_comp = not cfg.TEST.COMPETITION_MODE
    if _use_json_dataset_evaluator(dataset):
        coco_eval = json_dataset_evaluator.evaluate_boxes(
            dataset, all_boxes, output_dir, use_salt=not_comp, cleanup=not_comp
        )
        box_results = _coco_eval_to_box_results(coco_eval)
    elif _use_cityscapes_evaluator(dataset):
        logger.warn('Cityscapes bbox evaluated using COCO metrics/conversions')
        coco_eval = json_dataset_evaluator.evaluate_boxes(
            dataset, all_boxes, output_dir, use_salt=not_comp, cleanup=not_comp
        )
        box_results = _coco_eval_to_box_results(coco_eval)
    elif _use_voc_evaluator(dataset):
        # For VOC, always use salt and always cleanup because results are
        # written to the shared VOCdevkit results directory
        voc_eval = voc_dataset_evaluator.evaluate_boxes(
            dataset, all_boxes, output_dir, use_matlab=use_matlab
        )
        box_results = _voc_eval_to_box_results(voc_eval)
    else:
        raise NotImplementedError(
            'No evaluator for dataset: {}'.format(dataset.name)
        )
    return OrderedDict([(dataset.name, box_results)])


def evaluate_masks(dataset, all_boxes, all_segms, output_dir):
    """Evaluate instance segmentation."""
    logger.info('Evaluating segmentations')
    not_comp = not cfg.TEST.COMPETITION_MODE
    if _use_json_dataset_evaluator(dataset):
        coco_eval = json_dataset_evaluator.evaluate_masks(
            dataset,
            all_boxes,
            all_segms,
            output_dir,
            use_salt=not_comp,
            cleanup=not_comp
        )
        mask_results = _coco_eval_to_mask_results(coco_eval)
    elif _use_cityscapes_evaluator(dataset):
        cs_eval = cs_json_dataset_evaluator.evaluate_masks(
            dataset,
            all_boxes,
            all_segms,
            output_dir,
            use_salt=not_comp,
            cleanup=not_comp
        )
        mask_results = _cs_eval_to_mask_results(cs_eval)
    else:
        raise NotImplementedError(
            'No evaluator for dataset: {}'.format(dataset.name)
        )
    return OrderedDict([(dataset.name, mask_results)])

def evaluate_person_masks(dataset, all_boxes, all_personmasks, output_dir):
    """Evaluate instance segmentation."""
    logger.info('Evaluating segmentations')
    not_comp = not cfg.TEST.COMPETITION_MODE
    if _use_json_dataset_evaluator(dataset):
        coco_eval = json_dataset_evaluator.evaluate_personmasks(
            dataset,
            all_boxes,
            all_personmasks,
            output_dir,
            use_salt=not_comp,
            cleanup=not_comp
        )
        mask_results = _coco_eval_to_personmask_results(coco_eval)
    else:
        raise NotImplementedError(
            'No evaluator for dataset: {}'.format(dataset.name)
        )
    return OrderedDict([(dataset.name, mask_results)])


def evaluate_keypoints(dataset, all_boxes, all_keyps, output_dir):
    """Evaluate human keypoint detection (i.e., 2D pose estimation)."""
    logger.info('Evaluating detections')
    not_comp = not cfg.TEST.COMPETITION_MODE
    #assert dataset.name.startswith('keypoints_coco_'), \
    #    'Only COCO keypoints are currently supported'
    coco_eval = json_dataset_evaluator.evaluate_keypoints(
        dataset,
        all_boxes,
        all_keyps,
        output_dir,
        use_salt=not_comp,
        cleanup=not_comp
    )
    keypoint_results = _coco_eval_to_keypoint_results(coco_eval)
    return OrderedDict([(dataset.name, keypoint_results)])


def evaluate_body_uv(dataset, all_boxes, all_bodys, output_dir):
    """Evaluate human body uv (i.e. dense pose estimation)."""
    logger.info('Evaluating body uv')
    not_comp = not cfg.TEST.COMPETITION_MODE
    coco_eval = json_dataset_evaluator.evaluate_body_uv(
        dataset,
        all_boxes,
        all_bodys,
        output_dir,
        use_salt=not_comp,
        cleanup=not_comp
    )
    body_uv_results = _coco_eval_to_body_uv_results(coco_eval)
    return OrderedDict([(dataset.name, body_uv_results)])


def filename2imgid(parsing_COCO):
    parsing_COCO.imgname2id = {}
    for img in parsing_COCO.dataset['images']:
        parsing_COCO.imgname2id[img['file_name'].split('.')[0]] = img['id']

def evaluate_parsing(dataset, all_boxes, all_parss, output_dir):
    logger.info('Evaluating parsing')
    pkl_temp = glob.glob(os.path.join(output_dir, '*.pkl'))
    '''
    for pkl in pkl_temp:
        os.remove(pkl)
    '''
    dataset_name = cfg.TEST.DATASETS[0]
    _json_path = dataset_catalog.get_ann_fn(dataset_name)
    parsing_COCO = COCO(_json_path)
    filename2imgid(parsing_COCO)
    parsing_result = _empty_parsing_results()
    if dataset.name.find('test') > -1:
        return OrderedDict([(dataset.name, parsing_result)])
    predict_dir = os.path.join(output_dir, 'parsing_predict')
    assert os.path.exists(predict_dir), \
        'predict dir \'{}\' not found'.format(predict_dir)
    if True:
        _iou, _miou, _miou_s, _miou_m, _miou_l \
            = parsing_utils.parsing_iou(dataset, predict_dir, parsing_COCO)

        parsing_result['parsing']['mIoU'] = _miou
        parsing_result['parsing']['mIoUs'] = _miou_s
        parsing_result['parsing']['mIoUm'] = _miou_m
        parsing_result['parsing']['mIoUl'] = _miou_l

        parsing_name = parsing_utils.get_parsing()
        logger.info('IoU for each category:')
        assert len(parsing_name) == len(_iou), \
            '{} VS {}'.format(str(len(parsing_name)), str(len(_iou)))

        for i, iou in enumerate(_iou):
            print(' {:<30}:  {:.2f}'.format(parsing_name[i], 100 * iou))

        print('----------------------------------------')
        print(' {:<30}:  {:.2f}'.format('mean IoU', 100 * _miou))
        print(' {:<30}:  {:.2f}'.format('mean IoU small', 100 * _miou_s))
        print(' {:<30}:  {:.2f}'.format('mean IoU medium', 100 * _miou_m))
        print(' {:<30}:  {:.2f}'.format('mean IoU large', 100 * _miou_l))

    if True:
        all_ap_p, all_pcp = parsing_utils.eval_seg_ap(dataset, all_boxes[1], all_parss[1], parsing_COCO)
        ap_p_vol = np.mean(all_ap_p)

        logger.info('~~~~ Summary metrics ~~~~')
        print(' Average Precision based on part (APp)               @[mIoU=0.10:0.90 ] = {:.3f}'
            .format(ap_p_vol)
        )
        print(' Average Precision based on part (APp)               @[mIoU=0.10      ] = {:.3f}'
            .format(all_ap_p[0])
        )
        print(' Average Precision based on part (APp)               @[mIoU=0.30      ] = {:.3f}'
            .format(all_ap_p[2])
        )
        print(' Average Precision based on part (APp)               @[mIoU=0.50      ] = {:.3f}'
            .format(all_ap_p[4])
        )
        print(' Average Precision based on part (APp)               @[mIoU=0.70      ] = {:.3f}'
            .format(all_ap_p[6])
        )
        print(' Average Precision based on part (APp)               @[mIoU=0.90      ] = {:.3f}'
            .format(all_ap_p[8])
        )
        print(' Percentage of Correctly parsed semantic Parts (PCP) @[mIoU=0.50      ] = {:.3f}'
            .format(all_pcp[4])
        )
        parsing_result['parsing']['APp50'] = all_ap_p[4]
        parsing_result['parsing']['APpvol'] = ap_p_vol
        parsing_result['parsing']['PCP'] = all_pcp[4]


    return OrderedDict([(dataset.name, parsing_result)])

def evaluate_box_proposals(dataset, roidb):
    """Evaluate bounding box object proposals."""
    res = _empty_box_proposal_results()
    areas = {'all': '', 'small': 's', 'medium': 'm', 'large': 'l'}
    for limit in [100, 1000]:
        for area, suffix in areas.items():
            stats = json_dataset_evaluator.evaluate_box_proposals(
                dataset, roidb, area=area, limit=limit
            )
            key = 'AR{}@{:d}'.format(suffix, limit)
            res['box_proposal'][key] = stats['ar']
    return OrderedDict([(dataset.name, res)])


def log_box_proposal_results(results):
    """Log bounding box proposal results."""
    for dataset in results.keys():
        keys = results[dataset]['box_proposal'].keys()
        pad = max([len(k) for k in keys])
        logger.info(dataset)
        for k, v in results[dataset]['box_proposal'].items():
            logger.info('{}: {:.3f}'.format(k.ljust(pad), v))


def log_copy_paste_friendly_results(results):
    """Log results in a format that makes it easy to copy-and-paste in a
    spreadsheet. Lines are prefixed with 'copypaste: ' to make grepping easy.
    """
    for dataset in results.keys():
        logger.info('copypaste: Dataset: {}'.format(dataset))
        for task, metrics in results[dataset].items():
            logger.info('copypaste: Task: {}'.format(task))
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            logger.info('copypaste: ' + ','.join(metric_names))
            logger.info('copypaste: ' + ','.join(metric_vals))


def check_expected_results(results, atol=0.005, rtol=0.1):
    """Check actual results against expected results stored in
    cfg.EXPECTED_RESULTS. Optionally email if the match exceeds the specified
    tolerance.

    Expected results should take the form of a list of expectations, each
    specified by four elements: [dataset, task, metric, expected value]. For
    example: [['coco_2014_minival', 'box_proposal', 'AR@1000', 0.387], ...].
    """
    # cfg contains a reference set of results that we want to check against
    if len(cfg.EXPECTED_RESULTS) == 0:
        return

    for dataset, task, metric, expected_val in cfg.EXPECTED_RESULTS:
        assert dataset in results, 'Dataset {} not in results'.format(dataset)
        assert task in results[dataset], 'Task {} not in results'.format(task)
        assert metric in results[dataset][task], \
            'Metric {} not in results'.format(metric)
        actual_val = results[dataset][task][metric]
        err = abs(actual_val - expected_val)
        tol = atol + rtol * abs(expected_val)
        msg = (
            '{} > {} > {} sanity check (actual vs. expected): '
            '{:.3f} vs. {:.3f}, err={:.3f}, tol={:.3f}'
        ).format(dataset, task, metric, actual_val, expected_val, err, tol)
        if err > tol:
            msg = 'FAIL: ' + msg
            logger.error(msg)
            if cfg.EXPECTED_RESULTS_EMAIL != '':
                subject = 'Detectron end-to-end test failure'
                job_name = os.environ[
                    'DETECTRON_JOB_NAME'
                ] if 'DETECTRON_JOB_NAME' in os.environ else '<unknown>'
                job_id = os.environ[
                    'WORKFLOW_RUN_ID'
                ] if 'WORKFLOW_RUN_ID' in os.environ else '<unknown>'
                body = [
                    'Name:',
                    job_name,
                    'Run ID:',
                    job_id,
                    'Failure:',
                    msg,
                    'Config:',
                    pprint.pformat(cfg),
                    'Env:',
                    pprint.pformat(dict(os.environ)),
                ]
                send_email(
                    subject, '\n\n'.join(body), cfg.EXPECTED_RESULTS_EMAIL
                )
        else:
            msg = 'PASS: ' + msg
            logger.info(msg)


def _use_json_dataset_evaluator(dataset):
    """Check if the dataset uses the general json dataset evaluator."""
    return dataset.name.find('coco_') > -1 or cfg.TEST.FORCE_JSON_DATASET_EVAL


def _use_cityscapes_evaluator(dataset):
    """Check if the dataset uses the Cityscapes dataset evaluator."""
    return dataset.name.find('cityscapes_') > -1


def _use_voc_evaluator(dataset):
    """Check if the dataset uses the PASCAL VOC dataset evaluator."""
    return dataset.name[:4] == 'voc_'


# Indices in the stats array for COCO boxes and masks
COCO_AP = 0
COCO_AP50 = 1
COCO_AP75 = 2
COCO_APS = 3
COCO_APM = 4
COCO_APL = 5
# Slight difference for keypoints
COCO_KPS_APM = 3
COCO_KPS_APL = 4
# Difference for body uv
COCO_BODY_UV_AP75 = 6
COCO_BODY_UV_APM = 11
COCO_BODY_UV_APL = 12


# ---------------------------------------------------------------------------- #
# Helper functions for producing properly formatted results.
# ---------------------------------------------------------------------------- #

def _coco_eval_to_box_results(coco_eval):
    res = _empty_box_results()
    if coco_eval is not None:
        s = coco_eval.stats
        res['box']['AP'] = s[COCO_AP]
        res['box']['AP50'] = s[COCO_AP50]
        res['box']['AP75'] = s[COCO_AP75]
        res['box']['APs'] = s[COCO_APS]
        res['box']['APm'] = s[COCO_APM]
        res['box']['APl'] = s[COCO_APL]
    return res


def _coco_eval_to_mask_results(coco_eval):
    res = _empty_mask_results()
    if coco_eval is not None:
        s = coco_eval.stats
        res['mask']['AP'] = s[COCO_AP]
        res['mask']['AP50'] = s[COCO_AP50]
        res['mask']['AP75'] = s[COCO_AP75]
        res['mask']['APs'] = s[COCO_APS]
        res['mask']['APm'] = s[COCO_APM]
        res['mask']['APl'] = s[COCO_APL]
    return res


def _coco_eval_to_keypoint_results(coco_eval):
    res = _empty_keypoint_results()
    if coco_eval is not None:
        s = coco_eval.stats
        res['keypoint']['AP'] = s[COCO_AP]
        res['keypoint']['AP50'] = s[COCO_AP50]
        res['keypoint']['AP75'] = s[COCO_AP75]
        res['keypoint']['APm'] = s[COCO_KPS_APM]
        res['keypoint']['APl'] = s[COCO_KPS_APL]
    return res


def _coco_eval_to_body_uv_results(coco_eval):
    res = _empty_body_uv_results()
    if coco_eval is not None:
        s = coco_eval.stats
        res['body_uv']['AP'] = s[COCO_AP]
        res['body_uv']['AP50'] = s[COCO_AP50]
        res['body_uv']['AP75'] = s[COCO_BODY_UV_AP75]
        res['body_uv']['APm'] = s[COCO_BODY_UV_APM]
        res['body_uv']['APl'] = s[COCO_BODY_UV_APL]
    return res

def _coco_eval_to_personmask_results(coco_eval):
    res = _empty_personmask_results()
    if coco_eval is not None:
        s = coco_eval.stats
        res['person_mask']['AP'] = s[COCO_AP]
        res['person_mask']['AP50'] = s[COCO_AP50]
        res['person_mask']['AP75'] = s[COCO_BODY_UV_AP75]
        res['person_mask']['APm'] = s[COCO_BODY_UV_APM]
        res['person_mask']['APl'] = s[COCO_BODY_UV_APL]
    return res


def _voc_eval_to_box_results(voc_eval):
    # Not supported (return empty results)
    return _empty_box_results()


def _cs_eval_to_mask_results(cs_eval):
    # Not supported (return empty results)
    return _empty_mask_results()


def _empty_box_results():
    return OrderedDict({
        'box':
        OrderedDict(
            [
                ('AP', -1),
                ('AP50', -1),
                ('AP75', -1),
                ('APs', -1),
                ('APm', -1),
                ('APl', -1),
            ]
        )
    })


def _empty_mask_results():
    return OrderedDict({
        'mask':
        OrderedDict(
            [
                ('AP', -1),
                ('AP50', -1),
                ('AP75', -1),
                ('APs', -1),
                ('APm', -1),
                ('APl', -1),
            ]
        )
    })


def _empty_keypoint_results():
    return OrderedDict({
        'keypoint':
        OrderedDict(
            [
                ('AP', -1),
                ('AP50', -1),
                ('AP75', -1),
                ('APm', -1),
                ('APl', -1),
            ]
        )
    })


def _empty_body_uv_results():
    return OrderedDict({
        'body_uv':
        OrderedDict(
            [
                ('AP', -1),
                ('AP50', -1),
                ('AP75', -1),
                ('APm', -1),
                ('APl', -1),
            ]
        )
    })

def _empty_personmask_results():
    return OrderedDict({
        'person_mask':
        OrderedDict(
            [
                ('AP', -1),
                ('AP50', -1),
                ('AP75', -1),
                ('APm', -1),
                ('APl', -1),
            ]
        )
    })

def _empty_parsing_results():
    return OrderedDict({
        'parsing':
        OrderedDict(
            [
                ('mIoU', -1),
                ('mIoUs', -1),
                ('mIoUm', -1),
                ('mIoUl', -1),
                ('APp50', -1),
                ('APpvol', -1),
                ('PCP', -1),
            ]
        )
    })
def _empty_box_proposal_results():
    return OrderedDict({
        'box_proposal':
        OrderedDict(
            [
                ('AR@100', -1),
                ('ARs@100', -1),
                ('ARm@100', -1),
                ('ARl@100', -1),
                ('AR@1000', -1),
                ('ARs@1000', -1),
                ('ARm@1000', -1),
                ('ARl@1000', -1),
            ]
        )
    })
