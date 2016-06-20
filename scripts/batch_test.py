import os

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


data_path = "/home/wenqing/lprdata/"
label = open(data_path + "img_list.txt")
result = open("ttresult.txt", 'w')
num = 0
for line in label:
    num = num + 1
    print num
    if num > 5:
        break
    line = line.strip()
    img_path = data_path + "/3428/" + line
    tmp_comd = "./easypr_test recognize -p " + img_path + " --svm resources/model/svm.xml"
    output = os.popen(tmp_comd)
    img_result = output.read().split("\n")[0]
    s = unicode(img_result).encode("utf-8")
    print s
    result.write(line + " " + img_result + '\n')
result.close()



