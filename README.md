# 哈工大20年情感分析和计算系统实现
## 环境
- Python3.6 PyCharm专业版2020.1
## 数据文件
- 老师给的数据
  - 文件为“UTF-8”编码，数据以xml格式存储，其中id为数据编号，标签内容为文本内容
  - [积极训练数据](source/sample.positive.txt)：5000条
  - [消极训练数据](source/sample.negative.txt)：5000条
  - [测试数据](source/test.txt)：2500条
### 输出数据
- [测试数据结果](source/1172510217.csv)：2500行2列，并以逗号“,”为分隔符；第一列为测试数据id（0到2499），第二列为情感极性预测结果（0为消极，1为积极）
- 评分标准：以宏平均F1（macro-averaged F1-score）作为评分标准。