# DeepFGRN
DeepFGRN: Inference of gene regulatory network with regulation type based on directed graph embedding

This is the repository for the manuscript: DeepFGRN: Inference of gene regulatory network with regulation type based on directed graph embedding.

If you have any questions or feedback, please contact Zhen Gao (gaozhenchn@163.com)

# Datasets
| Species     | network | sourceE     |  sourceGRN     | numG     |dimE     |numA     |numR     |
|   :----:    |  :----:    |  :----:    |  :----:    |  :----:    |  :----:    |  :----:    |  :----:    |
| DREAM5     | network1 | DREAM5 challenge     |  DREAM5 challenge      | 1643     |  805     |  2236     |  1776     |
| E.coli     | cold stress | GEO(GSE20305)   |  RegulonDB     | 2205     |  24     |  2070     |  2034     |
| E.coli     | heat stress | GEO(GSE20305)   |  RegulonDB     | 2205     |  24     |  2070     |  2034     |
| E.coli     | lactose     | GEO(GSE20305)   |  RegulonDB     | 2205     |  12     |  2070     |  2034     |
| E.coli     | oxidative stress | GEO(GSE20305)   |  RegulonDB     | 2205     |  33     |  2070     |  2034     |
| Human     | COVID-19 | GEND000389   |  TRR, Reg     | 2478     |  42     |  6452     |  1888     |
| Human     | Breast cancer | GEND000024   |  TRR, Reg     | 2478     |  24     |  6452     |  1888     |
| Human     | Liver cancer | GEND000025   |  TRR, Reg     | 2478     |  10     |  6452     |  1888     |
| Human     | Lung cancer | GEND000176   |  TRR, Reg     | 2478     |  130     |  6452     |  1888     |

# Requirements
Please configure the environment according to the following versions:

python=3.6.5

keras==2.2.0

tensorflow==1.9.0

scikit-learn==0.22.1

numpy

pandas

random

math

re

System: Windows 10

We run the program in pyCharm.

DeepFGRN do not need GPU.


# Usage

## FGRN inference on DREAM5 challenge network1
step 1. Create a folder called "dDREAM5net1" and right-click ——open git bash here, Enter the following command

step 2. git clone -b master https://github.com/PhoebeGaoZhen/Demo3.git

step 3. cd Demo3

step 4. python deepGRN_DREAM5net1_FCV.py install --user


## FGRN inference on E.coli cold stress network


## FGRN inference on E.coli heat stress network



## FGRN inference on E.coli lactose stress network




## FGRN inference on E.coli oxidative stress network





## FGRN inference on human breast cancer 



## FGRN inference on human liver cancer 


## FGRN inference on human lung cancer 




## FGRN inference on human COVID




## Regular directed GRN inference on DREAM5 network 1-4




