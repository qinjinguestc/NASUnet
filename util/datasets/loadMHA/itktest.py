import SimpleITK as sitk
import os
import re

path = "\VSD.Brain_3more.XX.O.OT.54517.mha"
img = sitk.ReadImage((os.getcwd()+path))
array = sitk.GetArrayFromImage(img)
vsdId = re.findall("\d+", path)[1]

# name = "VSD.your_description."+sss+".dcm"
name = "VSD.your_description."+vsdId+".mha"
imgs = sitk.GetImageFromArray(array*255)
sitk.WriteImage(sitk.Cast(img, sitk.sitkUInt8), name)

from loadMHA.readMHATrain import BRATS2015

brats = BRATS2015()

for i in range(2):
    img, label = brats.next_train_batch(20)
    print(i, label.shape, img.shape)

