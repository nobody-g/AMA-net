import json
import cv2
import numpy as np
import pycocotools.mask as mask_util
import time
dataset_name = 'CIHP'
dataset_path = '/hhd/guoyuyu/datasets/coco2014/CIHP/'
data_path = dataset_path + '/instance-level_human_parsing/'
output_path = dataset_path + '/annotations/'
mask_resize = (256,256)

def from_seg_to_box(img_human_level, id_human):
    h_min = 99999
    h_max = 0
    w_min = 99999
    w_max = 0
    area_sum = 0
    for i in range(img_human_level.shape[0]):
        for j in range(img_human_level.shape[1]):
            if img_human_level[i][j] == id_human:
                area_sum = area_sum + 1
                if j >= w_max:
                    w_max = j
                if j <= w_min:
                    w_min = j
                if i >= h_max:
                    h_max = i
                if i <= h_min:
                    h_min = i

    return [w_min, h_min, float(w_max-w_min), float(h_max-h_min)], float(area_sum)


def build_ann():
    if dataset_name == 'CIHP':
        subset = ['Training','Validation','Testing']
        image_ids_file = ['train_id.txt','val_id.txt','test_id.txt']
        sub_dir = ['Categories', 'Category_ids', 'Human', 'Human_ids', 'Images', 'Instance_ids', 'Instances']
        class_s = ['Hat', 'Hair','Glove','Sunglasses','UpperClothes',
                     'Dress','Coat','Socks','Pants','Torso-skin',
                     'Scarf','Skirt','Face','Left-arm','Right-arm',
                      'Left-leg','Right-leg','Left-shoe','Right-shoe'
                     ]

        ann_count = 0
        for  ind, subset_i in enumerate(subset):
            dataset_i = {'images':[],
                        'annotations':[],
                        'categories':[{
                        'supercategory':'person',
                        'id': 1,
                        'name':'person',
                        'humanparts':class_s}]}
            if subset_i == 'Testing':
                break
            subset_path = data_path+'/'+ subset_i
            image_name_file = open(subset_path + '/' + image_ids_file[ind],'r')
            image_name_list = image_name_file.readlines()
            image_name_pure = []
            start_time = time.time()
            for img_ind , image_name_i in enumerate(image_name_list):
                if img_ind >= 1:
                    print('now: ', image_name_i,' images ind: ',img_ind, 'all images: ',len(image_name_list))
                    print('Need time: ', ((time.time()-start_time)/(img_ind*1.0))*(len(image_name_list)-img_ind-1))
                image_name_i = image_name_i.split('\n')[0]
                image_i = cv2.imread(subset_path+'/Images/'+image_name_i+'.jpg')
                height_i, width_i, _ = image_i.shape
                # Get Image dictionary
                image_dict = {'file_name':image_name_i+'.jpg',
                'height':height_i,
                'width':width_i,
                'id':int(image_name_i),
                }
                dataset_i['images'].append(image_dict)

                # Each image contains multi-person, we build a annotations dictionary for each person.
                img_ins_human_level = cv2.imread(subset_path+'/Human_ids/'+image_name_i+'.png')[:,:,0]

                img_seg_i = cv2.imread(subset_path+'/Category_ids/'+image_name_i+'.png')[:,:,0]

                num_human = img_ins_human_level.max()
                #print('img_ins_human_level: ',img_ins_human_level.shape)
                if num_human == 0:
                    ann_j = {}
                    print('No person in image: ',image_name_i)
                else:
                    for human_j in range(1, num_human+1):

                        human_j_mask = (img_ins_human_level==human_j) * 1
                        bbox_j, area_sum = from_seg_to_box(img_ins_human_level,human_j)
                        ann_j = {'category_id': 1,
                        'dp_masks': [],
                        'bbox': bbox_j,
                        'iscrowd': 0,
                        'area': area_sum,
                        'image_id':int(image_name_i),
                        'id':int(ann_count),
                        }
                        ann_count = ann_count + 1
                        box_w1 = int(bbox_j[0])
                        box_h1 = int(bbox_j[1])
                        box_w2 = int(bbox_j[0] + bbox_j[2])
                        box_h2 = int(bbox_j[1] + bbox_j[3])

                        img_seg_i_human_j_o = img_seg_i[box_h1:box_h2,box_w1:box_w2]
                        img_seg_i_human_j = img_seg_i_human_j_o
                        dp_masks_list = []
                        for class_ind in range(len(class_s)):
                            mask_ind = (img_seg_i_human_j == class_ind+1) * 1
                            if mask_ind.sum() > 0:
                                mask_ind_r = cv2.resize(mask_ind, mask_resize,interpolation=cv2.INTER_NEAREST)
                                rle = mask_util.encode(
                                    np.array(mask_ind_r[:, :, np.newaxis], order='F', dtype='uint8')
                                )[0]
                            else:
                                rle = []
                            dp_masks_list.append(rle)
                        ann_j['dp_masks'] = dp_masks_list
                        dataset_i['annotations'].append(ann_j)
                #if img_ind == 20:
                #    break
                #   json.dump(dataset_i,open(subset_i+'.json','w'))
            json.dump(dataset_i,open(output_path+subset_i+'.json','w'))
if __name__ == '__main__':
    build_ann()