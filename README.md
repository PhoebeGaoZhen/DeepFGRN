# DeepFGRN
This is the repository for the manuscript: DeepFGRN: Inference of gene regulatory network with regulation type based on directed graph embedding.

If you have any questions or feedback, please contact Zhen Gao (gaozhenchn@163.com)

# Datasets
## datasets for FGRN (GRN with both direction and regulation type) inference 
  
| Species   | network | sourceE     |  sourceGRN     | numG     |dimE     |numA     |numR     |
|   :----:  |  :----:    |  :----:    |  :----:    |  :----:    |  :----:    |  :----:    |  :----:    |
| DREAM5    | network1 | DREAM5 challenge     |  DREAM5 challenge      | 1643     |  805     |  2236     |  1776     |
| E.coli    | cold stress | GEO(GSE20305)   |  RegulonDB     | 2205     |  24     |  2070     |  2034     |
| E.coli    | heat stress | GEO(GSE20305)   |  RegulonDB     | 2205     |  24     |  2070     |  2034     |
| E.coli    | lactose     | GEO(GSE20305)   |  RegulonDB     | 2205     |  12     |  2070     |  2034     |
| E.coli    | oxidative stress | GEO(GSE20305)   |  RegulonDB     | 2205     |  33     |  2070     |  2034     |
| Human     | COVID-19 | GEND000389   |  TRR, Reg     | 2478     |  42     |  6452     |  1888     |
| Human     | Breast cancer | GEND000024   |  TRR, Reg     | 2478     |  24     |  6452     |  1888     |
| Human     | Liver cancer | GEND000025   |  TRR, Reg     | 2478     |  10     |  6452     |  1888     |
| Human     | Lung cancer | GEND000176   |  TRR, Reg     | 2478     |  130     |  6452     |  1888     |

<div>
Note: sourceE and sourceGRN represent databases that store gene expression data and prior gene regulatory network information, respectively, numG is the number of genes, dimE is the dimension of gene expression data, numA and numR represent the number of regulatory associations for known activation types and known repression types, respectively. TRR and Reg are TRRUST V2 and RegNetwork databases, respectively.

## datasets for regular GRN (GRN with only direction) inference 
| Network   | Organism | numG     |  dimE     | numKA     |
|   :----:  |  :----:    |  :----:    |  :----:    |  :----:    |
| network1    | in silico    | 1643  | 805  | 4012  |
| network2    | S.aureus     | 2810  | 160  | 518   |
| network3    | E.coli       | 4511  |  805 | 2066  |
| network4    | S.cerevisiae | 5950  |  536 | 3940  | 

Note: numG is the number of genes in the corresponding dataset, dimE is the dimension of gene expression profile, numA is the number of known regulatory associations. 


# Requirements
Please install the following software before replicating this framework in your local or server machine.

python=3.6.5

keras==2.2.0

tensorflow==1.9.0

scikit-learn==0.22.1

System: Windows 10

We run the program in pyCharm.

DeepFGRN do not need GPU.


# Usage

## FGRN inference on DREAM5 challenge network1
Create a folder called "dDREAM5net1" and right-click ——open git bash here, Enter the following command
```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd DREAM5_net1_FGRN
python DeepFGRN_DREAM5net1_FCV.py install --user
```
![Image](https://github.com/PhoebeGaoZhen/DeepFGRN/assets/54731874/03b096b1-adc2-42c2-8cd3-ada15cebf213)

Finally, check output folder "results" for results. The csv file shows the mean and standard deviation of AUROC, MCC, F1, Recall, Precision of DeepFGRN on this dataset.

## FGRN inference on E.coli cold stress network
```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd Ecoli_cold_FGRN
python DeepFGRN_cold_FCV.py install --user
```
## FGRN inference on E.coli heat stress network
```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd Ecoli_heat_FGRN
python DeepFGRN_heat_FCV.py install --user
```
## FGRN inference on E.coli lactose stress network
```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd Ecoli_lactose_FGRN
python DeepFGRN_lactose_FCV.py install --user
```
## FGRN inference on E.coli oxidative stress network

```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd Ecoli_oxidative_FGRN
python DeepFGRN_oxidative_FCV.py install --user
```
## FGRN inference on human breast cancer 

```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd human_breast_FGRN
python DeepFGRN_Breast_FCV.py install --user
```
## FGRN inference on human liver cancer 

```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd human_liver_FGRN
python DeepFGRN_Liver_FCV.py install --user
```
## FGRN inference on human lung cancer 

```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd human_lung_FGRN
python DeepFGRN_Lung_FCV.py install --user
```
## FGRN inference on human COVID-19

```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd human_COVID_FGRN
python DeepFGRN_COVID_FCV.py install --user
```
## Regular directed GRN inference on DREAM5 network 1-4
```
git clone -b master https://github.com/PhoebeGaoZhen/DeepFGRN.git
cd DeepFGRN
cd regularGRN
# regular GRN inference on DREAM5 challenge network1
python DeepFGRN_DREAM5net1_FCV.py install --user
# regular GRN inference on DREAM5 challenge network2
python DeepFGRN_DREAM5net2_FCV.py install --user
# regular GRN inference on DREAM5 challenge network3
python DeepFGRN_DREAM5net3_FCV.py install --user
# regular GRN inference on DREAM5 challenge network4
python DeepFGRN_DREAM5net4_FCV.py install --user
```
## FGRN inference on your own datasets

1. To run AGRN using your own data, you should prepare the following data:
(1) bulk gene expression data, the row are genes, the column are samples.
For example, final_Ecoli_cold.csv:

| genename | rep1_T1     | rep1_T2     | rep1_T3     | rep1_T4     | rep1_T5     | rep1_T6     | rep1_T7     | rep1_T8      |
|  :----:  |    :----:   |    :----:   |    :----:   |    :----:   |    :----:   |    :----:   |    :----:   |    :----:    |
| fis      | 7906.679678 | 8071.058674 | 6499.423349 | 7071.662477 | 6761.41511  | 6658.298389 | 4950.269701 | 1549.874613  |
| rob      | 268.7634704 | 304.8048094 | 409.8485438 | 502.459958  | 526.747015  | 500.7725783 | 650.9777228 | 797.9063958  |
| tyrR     | 153.6261072 | 165.0300588 | 162.9723776 | 175.0433137 | 145.9532194 | 149.480681  | 187.7007897 | 237.7467594  |
| argP     | 117.6696196 | 123.8847839 | 105.3549663 | 91.61446598 | 92.93579368 | 79.69010421 | 95.98128736 | 130.0416461  |
| cpxR     | 37.57650657 | 33.99388148 | 30.42700231 | 39.50596433 | 42.47627769 | 50.17389993 | 45.100724   | 27.86825098  |
| modE     | 415.867534  | 427.4287654 | 404.4242026 | 309.0218514 | 330.5271034 | 337.3798891 | 410.9775629 | 714.0313768  |
| fnr      | 350.8975634 | 432.1036921 | 520.7540333 | 483.3465972 | 469.0966565 | 481.0814296 | 503.0141359 | 561.0446351  |
| crp      | 479.9922059 | 542.4784364 | 421.771247  | 415.0021528 | 393.1588899 | 379.9352762 | 425.9823301 | 751.7603158  |
| cadC     | 4.372161138 | 3.933822191 | 8.369199648 | 6.989615816 | 8.913259285 | 8.942484856 | 9.061118141 | 13.46661152  |

Then, set "path_expression" in DeepFGRN_cold_FCV.py to the path of this dataset.

(2) prior GRN with regulation type
For examplw, Ecoli_GRN_3types.csv:

| fis | adhE | activator  |
|:----:|:----:|   :----:   |
| fis | apaG | activator  |
| fis | apaH | activator  |
| fis | carA | activator  |
| fis | carB | activator  |
| fis | cspA | activator  |
| fis | cyoA | activator  |
| fis | cyoB | activator  |
| fis | cyoC | activator  |
| fis | cyoD | activator  |

Then, set "path_network_name_type" in DeepFGRN_cold_FCV.py to the path of this dataset.

(3) gene list, the first column is the name of the gene, and the second column is the number of the gene, starting with 0
For example, gene2205_2.txt:

| name | ids  |
|:----:|:----:|
| fis  | 0    |
| rob  | 1    | 
| tyrR | 2    |
| argP | 3    |
| cpxR | 4    |
| modE | 5    |
| fnr  | 6    |
| crp  | 7    |
| cadC | 8    | 

Then, set "path_node" in DeepFGRN_cold_FCV.py to the path of this dataset.

(4) numbered version of the prior GRN. The gene names in the prior GRN with regulation type are converted to gene numbers.
For example, Ecoli_GRN_3types_ids.tsv:

| 0 | 175  |
|:----:|:----:|
| 0 | 178  |
| 0 | 179  |
| 0 | 183  |
| 0 | 184  |
| 0 | 91   |
| 0 | 185  |
| 0 | 186  |
| 0 | 187  |
| 0 | 188  |

Then, set "path_network_ids" in DeepFGRN_cold_FCV.py to the path of this dataset.

2. Run the DeepFGRN_cold_FCV.py

# Authors
Zhen Gao ,1 Yansen Su,2 Junfeng Xia,3 Rui-Fen Cao,1 Yun Ding,2 Chun-Hou Zheng2,∗ and Pi-Jing Wei3,∗

1 The Key Laboratory of Intelligent Computing and Signal Processing of Ministry of Education, School of Computer Science and Technology, Anhui University, 111 Jiulong Road, Hefei, 230601, Anhui, China

2 The Key Laboratory of Intelligent Computing and Signal Processing of Ministry of Education, School of Artificial Intelligence, Anhui University, 111 Jiulong Road, Hefei, 230601, Anhui, China and

3 Information Materials and Intelligent Sensing Laboratory of Anhui Province, Institute of Physical Science and Information Technology,
Anhui University, 111 Jiulong Road, Hefei, 230601, Anhui, China

∗Corresponding author. zhengch99@126.com, weipj@ahu.edu.cn

Zhen Gao: gaozhenchn@163.com









