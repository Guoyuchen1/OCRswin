import os
#测试集路径
filePath = 'E:/testswin/test'
#测试集文本路径，放在测试集的上级文件夹，可自动生成txt
fo= open('E:/testswin/test_list.txt','a+')
write=''
#fo.writelines(os.listdir(filePath)[1].split('_')
print(len(os.listdir(filePath)))
for i in range (len(os.listdir(filePath))-1):
    write+=filePath+os.listdir(filePath)[i]+' '+os.listdir(filePath)[i].split('_')[1]+'\n'
fo.write(write)