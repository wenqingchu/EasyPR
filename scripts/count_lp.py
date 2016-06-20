import sys
reload(sys)
sys.setdefaultencoding('utf-8')







hanzi_dict = {"1":"zh_cuan",  "2":"zh_e", "3":"zh_gan","4":"zh_gan1", "5":"zh_gui", "6": "zh_gui1","7":"zh_hei",  "8": "zh_hu",
"9":"zh_ji","a":"zh_jin", "b":"zh_jing", "c":"zh_jl","d":"zh_liao","e":"zh_lu", "f":"zh_meng","g":"zh_min", "h":"zh_ning", "i":"zh_qing",
"j":"zh_qiong", "k":"zh_shan", "l":"zh_su","m""zh_sx",  "n":"zh_wan", "o":"zh_xiang","p":"zh_xin", "q":"zh_yu", "r":"zh_yu1","s":"zh_yue", "t":  "zh_yun" ,  "u":"zh_zang","v":"zh_zhe"}


with open("result.txt", 'r') as f:
	input = f.readlines()

right_lpr = []
wrong_lpr = []
not_lpr = []


for i in range(len(input)):
	if len(input[i]) < 30:
		not_lpr.append(input[i])
		continue
	img_name, label, x1, x2, y1, y2 = input[i].strip().split(' ')
	gt_num = img_name[-10:-4]
	label_num = label[-6:]
	if gt_num == label_num:
		right_lpr.append(input[i])
	else:
		wrong_lpr.append(input[i])

ff = open("right_result.txt", 'w')
for i in range(len(right_lpr)):
	ff.write(right_lpr[i])

ff.close()


fff = open("wrong_result.txt", 'w')
for i in range(len(wrong_lpr)):
	fff.write(wrong_lpr[i])

fff.close()

ffff = open("not_result.txt", 'w')
for i in range(len(not_lpr)):
	ffff.write(not_lpr[i])

ffff.close()
