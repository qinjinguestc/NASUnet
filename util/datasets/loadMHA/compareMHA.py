import SimpleITK as sitk
from utils import *
import os

region1_avg = 0
region2_avg = 0
region3_avg = 0
i = 0

mha_save_path = r"../mha_save_path/"
mha_ground_truth = mha_save_path + "mha_ground_truth/"
mha_result = mha_save_path + "mha_result/"

result_paths = os.listdir(mha_ground_truth)
labels = []
ups = []
# result_paths = result_paths[1:]
for result_path in result_paths:
    i += 1

    mha = sitk.ReadImage(mha_ground_truth + result_path)
    label = sitk.GetArrayFromImage(mha)

    mha2 = sitk.ReadImage(mha_result + result_path)
    up = sitk.GetArrayFromImage(mha2)

    print(result_path)
    dataR1, labelR1 = Region1(up, label)
    region1 = dice_coef(dataR1, labelR1)
    region1_avg += region1

    dataR2, labelR2 = Region2(up, label)
    region2 = dice_coef(dataR2, labelR2)
    region2_avg += region2

    dataR3, labelR3 = Region3(up, label)
    region3 = dice_coef(dataR3, labelR3)

    region3_avg += region3

    print("[===========result_path===========R1:{:.5f} - R2:{:.5f} - R3:{:.5f}===========]\n".format(region1, region2, region3))

print("[===========R1:{:.5f} - R2:{:.5f} - R3:{:.5f}===========]".format(region1_avg/i, region2_avg/i, region3_avg/i))

# ups = np.array(ups)
# labels = np.array(labels)
#
# datatotle1, labeltotle1 = Region.Region1(ups, labels)
# regiontotle1 = dice_coef(datatotle1, labeltotle1)
# print(regiontotle1)
# datatotle2, labeltotle2 = Region.Region2(ups, labels)
# regiontotle2 = dice_coef(datatotle2, labeltotle2)
# print(regiontotle2)
#
# datatotle3, labeltotle3 = Region.Region3(ups, labels)
# regiontotle3 = dice_coef(datatotle3, labeltotle3)
# print(regiontotle3)

    # print(up.shape)
    # dataR1, labelR1 = Region.Region1(up, label)
    # region1 = dice_coef(dataR1, labelR1)
	#
    # # dataR1, labelR1 = Region.Region1(up, label)
    # # region1 = dice_coef(dataR1, labelR1)
    # dataR2, labelR2 = Region.Region2(up, label)
    # region2 = dice_coef(dataR2, labelR2)
    # #
    # dataR3, labelR3 = Region.Region3(up, label)
    # region3 = dice_coef(dataR3, labelR3)
	#
    # print("region1:\t",region1)
    # print("region2:\t",region2)
    # print("region3:\t",region3)

    # print(len(labels))

# datatotle1, labeltotle1 = Region.Region1(ups, labels)
# regiontotle1 = dice_coef(datatotle1, labeltotle1)
#
# datatotle2, labeltotle2 = Region.Region2(ups, labels)
# regiontotle2 = dice_coef(datatotle2, labeltotle2)
#
# datatotle3, labeltotle3 = Region.Region3(ups, labels)
# regiontotle3 = dice_coef(datatotle3, labeltotle3)
#
# print(regiontotle1+'\t'+regiontotle2+'\t'+regiontotle3)

    # dataR1, labelR1 = Region.Region1(up, label)
    # region1 = dice_coef(dataR1, labelR1)
    # dataR2, labelR2 = Region.Region2(up, label)
    # region2 = dice_coef(dataR2, labelR2)
    # #
    # dataR3, labelR3 = Region.Region3(up, label)
    # region3 = dice_coef(dataR3, labelR3)


    # print(region1)
    # print(region2)
    # print(region3)
# for i in range(155):
#     print("test:" + str(i))
#     dataR1, labelR1 = Region.Region1(up[i], label[i])
#     region1 = dice_coef(dataR1, labelR1)
#     region1_avg += region1
#
#     dataR2, labelR2 = Region.Region2(up[i], label[i])
#     region2 = dice_coef(dataR2, labelR2)
#     region2_avg += region2
#
#     dataR3, labelR3 = Region.Region3(up[i], label[i])
#     region3 = dice_coef(dataR3, labelR3)
#     region3_avg += region3
#
#     print("Region1:", region1 )
#     print("Region2:", region2 )
#     print("Region3:", region3 )
#
# print("Region1:", region1_avg/155)
# print("Region2:", region2_avg/155)
# print("Region3:", region3_avg/155)
#
