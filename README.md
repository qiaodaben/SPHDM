# SPHDM 
A Structure based Phish Homology Detection Model (SPHDM)  is proposed to detected the phishing web.

Thank you for your interests in our work!

The dataset we ultilized for training and testing for  is reposited in github.

Dataset Sources
phishing webpage:https://phishtank.org/  
benign webpage:https://www.alexa.com/  

After the data crawling process, each file contains the entire phishing (or normal) data set in the format：{  
'url': {  
        'classstyl': [];  
        'hashcode':'';  
        'idstyle':[];  
        'name': url;  
        'newtagseq':[tag sequence with hierarchy];  
        'tagseq':[Tag sequence has no hierarchical information] }  
}  

Address: https://github.com/qiaodaben/SPHDM-/tree/main/dataset

You can download this notebook as well as the well-organized dataset for training and testing. The toy example for visualization is in SPHDM Respository. If you find this work interesting and helpful to your work, please find the citation of the papers as below. Thank you very much. Any question you can email to fengjian@xust.edu.cn

 @inproceedings{feng2021SPHDM, title={Detecting Phishing Webpages via Homology Analysis of Webpage Structure}, author={Jian Feng, Yuqiang Qiao, Ou Ye, and Ying Zhang }}
