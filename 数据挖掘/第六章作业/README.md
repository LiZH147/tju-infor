# Apriori & FP-Growth 实现

## 1  运行环境
- Python ≥ 3.6  

## 2  运行示例
cd src
python apriori.py -f ../data/grocery.txt -s 0.3 -c 0.7 > ../result/apriori.txt
python fpgrowth.py -f ../data/grocery.txt -s 0.3 > ../result/fpgrowth.txt          

## 3  文件说明
| 文件 | 作用 |
|---|---|
| apriori.py | Apriori |
| fpgrowth.py | FP-Growth |
| result/ | 示例输出 |

## 4  数据
grocery.txt 为超市购物篮，公开数据集，只需保持“每行一个事务，商品空格分隔”即可。

## 5  结果解释
程序会输出频繁项集（支持度）与关联规则（置信度），例如  
`['diapers'] -&gt; ['beer']  conf=0.80`  
表示购买 diapers 的顾客有 80% 也会买 beer。