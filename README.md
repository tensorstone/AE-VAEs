# AE-VAEs
AutoEncoder and Variational AutoEncoders indifferent datasets

AE in different datasets.py：



VAE_idealTXY.py：
使用VAE进行的三角晶格XYmodel聚类工作，数据使用的是理想数据，也即人工生成的数据，其中包含两个chiral order，可以用VAE进行区分
这个程序主要用来验证不同KL term时候的分类效果，以及检验各个hidden variable对应的权重向量是否正交

实验结果是，KLterm非零时比为零时权重内积更小，但增大KLterm没有明显趋于零的收敛趋势。
带有图表的实验结果记录在有道云笔记中

AE for ideal TXY data.py:
感觉KLterm对正交性没什么贡献，就简化网络为AE
最后一部分代码计算了权重矩阵的内积，接下来需要验证这样的内积和I的关系，从而确定正交性的强弱（与regularizer的关系）
