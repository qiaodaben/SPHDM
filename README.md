# SPHDM 
    A Structure based Phish Homology Detection Model (SPHDM)  is proposed to detected the phishing web.

    Thank you for your interests in our work!

    The dataset we ultilized for training and testin is reposited in github.

## Dataset Sources  

The webpages used in the experiments come from Internet. Among them, the benign webpage collection is from [Alexa](https://www.alexa.com/) . Alexa is a website maintained by Amazon that publishes the world rankings of websites. We collect webpages in the top list provided by Alexa which are considered as benign webpages. After filtering out invalid, error, and duplicate pages, 10,922 benign webpages are collected.

The phishing webpage collection comes from [PhishTank.com](https://phishtank.org/). PhishTank is an internationally well-known website which collects suspected phish submitted by anyone, verifies it according to whether it has a fraudulent attempt or not, and then publish a timely and authoritative list of phishing webpages for research. Due to the short survival time of phishing webpages, we collected totally 10,944 phishing webpages listed on PhishTank every day from September 2019 to November 2019, and processed the webpages that did not meet the grammar rules.


[phishing webpage: PhishTank](https://phishtank.org/)  
[benign webpage: Alexa](https://www.alexa.com/)  

## Pretreatment
After webpage data is crawled and preprocessed, each file contains a complete set of phishing (or normal) webpages, __`page_PCA_N.7z`__ represents the benign webpage data set, __`page_PCA_P.7z`__ represents the phishing webpage data set, the format is as follows:{
After webpages are crawled and preprocessed, two files are created. The __`page_PCA_N.7z`_ contains all benign webpages, and the __`page_PCA_P.7z`__ contains all phishing webpages. 
The format of one webpage in files is as follows:
__'url'__: {  
- __'classstyl'__:   [];  
- __'hashcode'__:    '';  
- __'idstyle'__:     [];  
- __'name'__:        url;  
- __'newtagseq'__:   [tag sequence with hierarchy];  
- __'tagseq'__:      [Tag sequence has no hierarchical information] 
}  

## Address [Address](https://github.com/qiaodaben/SPHDM-/tree/main/dataset)

## Contact
You can download this notebook as well as the well-organized dataset for training and testing. The toy example for visualization is in SPHDM Respository. If you find this work interesting and helpful to your work, please find the citation of the papers as below. Thank you very much. Any question you can email to fengjian@xust.edu.cn

 @inproceedings{feng2021SPHDM, title={Detecting Phishing Webpages via Homology Analysis of Webpage Structure}, author={Jian Feng, Yuqiang Qiao, Ou Ye, and Ying Zhang }}
