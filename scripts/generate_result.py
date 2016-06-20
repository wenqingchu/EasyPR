import sys
reload(sys)
sys.setdefaultencoding('utf-8')




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
