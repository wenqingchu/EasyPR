#  -*- coding: utf-8 -*-

import os

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


'''
hanzi_dict = {"1":"zh_cuan",  "2":"zh_e", "3":"zh_gan","4":"zh_gan1", "5":"zh_gui", "6": "zh_gui1","7":"zh_hei",  "8": "zh_hu",
"9":"zh_ji","a":"zh_jin", "b":"zh_jing", "c":"zh_jl","d":"zh_liao","e":"zh_lu", "f":"zh_meng","g":"zh_min", "h":"zh_ning", "i":"zh_qing",
"j":"zh_qiong", "k":"zh_shan", "l":"zh_su","m""zh_sx",  "n":"zh_wan", "o":"zh_xiang","p":"zh_xin", "q":"zh_yu", "r":"zh_yu1","s":"zh_yue", "t":  "zh_yun" ,  "u":"zh_zang","v":"zh_zhe"}
'''

filename = "市心路电警项目NVR005_市心路-道源路南向北1_20160523073000_20160523093000_20160523072959_20160523080059"
data_path = "/home/wenqing/lprdata/traffic_dataset/"
img_list = open(data_path + filename + ".txt", 'r')
result = open("/home/wenqing/lprdata/traffic_dataset/result/" + filename + ".txt", 'w')
num = 0
for line in img_list:
    num = num + 1
    print num
    if num % 2 == 0:
        continue
    #if num > 5:
    #    break
    line = line.strip()
    img_path = data_path + filename + "/" + line
    tmp_comd = "./easypr_test recognize -p " + img_path + " --svm resources/model/svm.xml"
    #print tmp_comd
    #os.system(tmp_comd)
    output = os.popen(tmp_comd)
    img_results = output.read().split("\n")

    result.write(line + " " + " ".join(img_results[:3]) + '\n')
result.close()



