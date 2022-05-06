import os
#训练集路径
filePath = 'E:/testswin/train'

#训练集文本路径，放在测试集的上级文件夹，可自动生成txt
fo= open('E:/testswin/train_list.txt','a+')
write=''
#fo.writelines(os.listdir(filePath)[1].split('_')
print(len(os.listdir(filePath)))
for i in range (len(os.listdir(filePath))-1):
    write+= filePath+os.listdir(filePath)[i]+' '+os.listdir(filePath)[i].split('_')[1]+'\n'
fo.write(write)