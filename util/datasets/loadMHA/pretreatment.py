import os
import shutil
from glob import glob
import SimpleITK as sitk


def load(path, dst_path):
    L = os.listdir(path)

    for i in L:
        path_1 = path + '/' + i
        if os.path.exists(dst_path + '/' + os.path.basename(path_1)):
            print(dst_path + '/' + os.path.basename(path_1))
            continue

        OT_path = glob(path_1 + '/*OT*/*.mha')

        if not os.path.exists(dst_path + '/' + os.path.basename(path_1)):
            os.makedirs(dst_path + '/' + os.path.basename(path_1))

        if OT_path:
            shutil.copy(OT_path[0], dst_path + '/' + os.path.basename(path_1) + '/' + os.path.basename(OT_path[0]))

        T1_path = glob(path_1 + '/*T1*/*.mha')
        N4(T1_path[0], dst_path + '/' + os.path.basename(path_1) + '/' + os.path.basename(T1_path[0]))

        T1c_path = glob(path_1 + '/*T1c*/*.mha')
        N4(T1c_path[0], dst_path + '/' + os.path.basename(path_1) + '/' + os.path.basename(T1c_path[0]))

        Flair_path = glob(path_1 + '/*Flair*/*.mha')
        N4(Flair_path[0], dst_path + '/' + os.path.basename(path_1) + '/' + os.path.basename(Flair_path[0]))

        T2_path = glob(path_1 + '/*T2*/*.mha')
        N4(T2_path[0], dst_path + '/' + os.path.basename(path_1) + '/' + os.path.basename(T2_path[0]))


        # os.remove(path_1 + '/' + os.path.basename(OT_path[0]))
        print("完成:", i)


def N4(input, outputPath):
    inputImage = sitk.ReadImage(input)

    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    # maskImagePath = input('Enter the name of the mask image to be saved : ')
    # sitk.WriteImage(maskImage, maskImagePath)
    # print("Mask image is saved.")

    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    output = corrector.Execute(inputImage, maskImage)

    outputPath = outputPath
    sitk.WriteImage(output, outputPath)


if __name__=='__main__':
    # path = 'F:\BRATS2015\BRATS2015_Training/HGG'
    # dst_path = r"F:\BRATS2015\brats_n4_bias_training/HGG"
    # load(path, dst_path)
    # path = 'F:\BRATS2015\BRATS2015_Training/LGG'
    # dst_path = r"F:\BRATS2015\brats_n4_bias_training/LGG"
    # load(path, dst_path)
    # path = 'F:\BRATS2015\BRATS2015_Testing/HGG_LGG'
    # dst_path = r"F:\BRATS2015\brats_n4_bias_testing"
    # load(path, dst_path)

    # path = '/media/zlz424/experiment/BRATS2015/BRATS2015_Training/HGG'
    # dst_path = r"/media/zlz424/experiment/BRATS2015/brats_n4_bias_training/HGG"
    # load(path, dst_path)
    # path = '/media/zlz424/experiment/BRATS2015/BRATS2015_Training/LGG'
    # dst_path = r"/media/zlz424/experiment/BRATS2015/brats_n4_bias_training/LGG"
    # load(path, dst_path)
    #
    path = '/media/zlz424/experiment/BRATS2015/BRATS2015_Testing/HGG_LGG'
    dst_path = r"/media/zlz424/experiment/BRATS2015/brats_n4_bias_testing"
    load(path, dst_path)
    print("Finished N4 Bias Field Correction.....")
