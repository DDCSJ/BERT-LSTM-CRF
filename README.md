# BERT-LSTM-CRF
命名实体识别 语言python 支持版本python3.8-python3.11  
使用说明：  
从huggingface中搜索bert-base_chinese，并下载config.json ; pytorch_model.bin ; vocab.txt。  
需要在总文件夹中配置一个 bert-chinese-base 文件，文件中包含上步下载内容。  
模型在model文件夹中。  
参数在config.py中可以根据需求调整模型参数和训练数据文件地址。  
predict.py 预测。  
data数据中gushi文件夹有导入数据格式的示范，text和train的格式相同。  
data数据中tag是bio标签类型，可以在参数中修改导入tag的数量。
