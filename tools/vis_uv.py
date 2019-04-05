import numpy
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
import os 
import random
from shutil import copyfile
random.randint(0,9)
res_path = '/coco/results/detectron-output_mulres_intersup_mulsaclesup_intw0_5/test/dense_coco_2014_minival/generalized_rcnn/vis/'
output_path = '/home/densepose_wxh/densepose/tools/vis_uv/'
#res_path = '/coco/results/baseline/test/dense_coco_2014_minival/generalized_rcnn/vis/'
#output_path = '/home/densepose_wxh/densepose/tools/vis_uv_baseline/'
only_uv = True
num_sample = 3000
def file_name(file_dir): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pdf':
                L.append(os.path.join(root, file))
    return L

img_list = file_name(res_path)
'''
img_list = [res_path+'1223_COCO_val2014_000000464089.pdf']
'''  
img_list = [res_path+'0_COCO_val2014_000000000785.pdf',    res_path+'1475_COCO_val2014_000000568213.pdf',
            res_path+'1472_COCO_val2014_000000567640.pdf', res_path+'1188_COCO_val2014_000000453166.pdf',
            res_path+'1378_COCO_val2014_000000522889.pdf', res_path+'1053_COCO_val2014_000000401244.pdf',res_path+'916_COCO_val2014_000000352760.pdf',
            res_path+'837_COCO_val2014_000000323799.pdf',  res_path+'223_COCO_val2014_000000081988.pdf',
            res_path+'1453_COCO_val2014_000000559842.pdf', res_path+'1394_COCO_val2014_000000533816.pdf',
            
            
            
            res_path+'46_COCO_val2014_000000013201.pdf',   res_path+'44_COCO_val2014_000000012748.pdf',
            res_path+'1494_COCO_val2014_000000576566.pdf', res_path+'750_COCO_val2014_000000288862.pdf',
            res_path+'1_COCO_val2014_000000000872.pdf',    res_path+'693_COCO_val2014_000000267169.pdf',
            res_path+'1453_COCO_val2014_000000559842.pdf', res_path+'4_COCO_val2014_000000001268.pdf',
            res_path+'1470_COCO_val2014_000000566436.pdf', res_path+'32_COCO_val2014_000000009378.pdf',



            res_path+'1061_COCO_val2014_000000404249.pdf', res_path+'603_COCO_val2014_000000231339.pdf',
            res_path+'1270_COCO_val2014_000000477288.pdf', res_path+'813_COCO_val2014_000000314177.pdf']


'''
1097_COCO_val2014_000000418062.pdf
32_COCO_val2014_000000009378.pdf
829_COCO_val2014_000000321790.pdf
504_COCO_val2014_000000190637.pdf
992_COCO_val2014_000000379332.pdf
'''
img_ind = [random.randint(0,len(img_list)-1) for _ in range(num_sample)]
if num_sample > len(img_list):
    num_sample = len(img_list)
    img_ind = np.arange(num_sample)
dpi = 200
print('len img_ind: ',len(img_ind))

for i in img_ind:
    print('img_list i :',img_list[i])
    img_uname = ((img_list[i]).split('/')[-1]).split('.')[0]
    print('img_uname i :',img_uname)
    img_name = img_uname.split('_')[1]+'_'+img_uname.split('_')[2]+'_'+img_uname.split('_')[3]
    print('img_name: ',img_name)
    im  = cv2.imread('/coco/val2014/'+img_name+'.jpg')
    IUV = cv2.imread(res_path+img_uname+'_IUV.png')
    INDS = cv2.imread(res_path+img_uname+'_INDS.png',  0)
    '''
    fig = plt.figure(figsize=[15,15])
    plt.imshow(   np.hstack((IUV[:,:,0]/24. ,IUV[:,:,1]/256. ,IUV[:,:,2]/256.))  )
    plt.title('I, U and V images.')
    plt.axis('off') ; plt.show()
    '''

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im_f = Image.fromarray(im[:,:,::-1])
    im_2 = ImageEnhance.Color(im_f).enhance(0.3)
    im_2a = numpy.array(im_2)
    #fig = plt.figure(figsize=[12,12])
    ax.imshow(im_2a, aspect='auto')
    ax.contour( IUV[:,:,1]/256.,15, linewidths = 0.7 )
    ax.contour( IUV[:,:,2]/256.,15, linewidths = 0.7 )
    fig.savefig(output_path+img_uname+'_V.png', dpi=dpi)
    copyfile('/coco/val2014/'+img_name+'.jpg', output_path+img_name+'.jpg')
    if not only_uv:
        copyfile(res_path+img_uname+'_Partseg.png', output_path+img_uname+'_Partseg.png')
        copyfile(res_path+img_uname+'_UVPartseg.png', output_path+img_uname+'_UVPartseg.png')
        copyfile(res_path+img_uname+'.pdf', output_path+img_uname+'_Mask.pdf')
    plt.close('all')
    im = None
    IUV = None
    INDS = None
#plt.show()