import os

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


lp_cha = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

cha_digit = {}

for i in range(len(lp_cha)):
    cha_digit[lp_cha[i]] = i


img_path = "/home/wenqing/lprdata/"
img_save_path = "/home/wenqing/lprcode/EasyPR/digit_image/"


data_path = "/home/wenqing/lprdata/"
label = open("/home/wenqing/lprcode/EasyPR/showresults/right_result.txt")
result = open("digit_all.txt", 'w')


num = 0
for line in label:
    num = num + 1
    print num
    #if num > 5:
    #    break
    line = line.strip()
    img_name = line.split(' ')[0]
    lp_number = img_name.split('_')[1]
    lp_number = lp_number[-10:-4]
    img_number = img_name.split('_')[0]

    img_path = data_path + "/3428/" + img_name
    tmp_comd = "./easypr_test recognize -p " + img_path + " --svm resources/model/svm.xml"

    #print tmp_comd
    os.system(tmp_comd)
    for i in range(6):
        src_img = str(i+1) + ".jpg"
        target_img = "digit_image/" + str(cha_digit[lp_number[i]]) + "/" + img_number + "_" + src_img
        cmd = "mv " + src_img + " " + target_img
        os.system(cmd)
        #print cmd
        result.write(target_img + " " + str(cha_digit[lp_number[i]]) + '\n')
result.close()
label.close()


