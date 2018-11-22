import pandas
import webbrowser
import os


data_table = pandas.read_csv('ml_house_data_set.csv')

html=data_table[0:100].to_html()
'''转化为html 格式进行显示  '''

with open('data.html','w') as f:
    f.write(html)


full_filename = os.path.abspath('data.html')
'''返回path 规范化的绝对路径'''

print(full_filename)

webbrowser.open("file://{}".format(full_filename))
'''file 之后的位置问题不详'''

