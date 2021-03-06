<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
# SPHDM 
    A method of Structure based Phish Homology Detection Model (SPHDM) is proposed to detect the phishing webpages.

    Thank you for your interests in our work!

    The dataset used by SPHDM for training and testing is deposited here in SPHDM/dataset.

## Dataset Sources  

The webpages used in the experiments come from Internet. Among them, the benign webpage collection is from [Alexa](https://www.alexa.com/) . Alexa is a website maintained by Amazon that publishes the world rankings of websites. We collect webpages in the top list provided by Alexa which are considered as benign webpages. After filtering out invalid, error, and duplicate pages, 10,922 benign webpages are collected.

The phishing webpage collection comes from [PhishTank.org](https://phishtank.org/). PhishTank is an internationally well-known website which collects suspected phish submitted by anyone, verifies it according to whether it has a fraudulent attempt or not, and then publish a timely and authoritative list of phishing webpages for research. Due to the short survival time of phishing webpages, we collected totally 10,944 phishing webpages listed on PhishTank every day from September 2019 to November 2019, and processed the webpages that did not meet the grammar rules.


[phishing webpage: PhishTank](https://phishtank.org/)  
[benign webpage: Alexa](https://www.alexa.com/)  

## Pretreatment

After webpages are crawled and preprocessed, two files are created. The __`page_PCA_N.7z`__ contains all benign webpages, and the __`page_PCA_P.7z`__ contains all phishing webpages.  
The processed data is stored in [address](https://github.com/qiaodaben/SPHDM-/tree/main/dataset).   
The format of one webpage in files is as follows:

<table class="MsoTableGrid" border="1" cellspacing="0" cellpadding="0" style="border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt">
 <tbody><tr style="mso-yfti-irow:0;mso-yfti-firstrow:yes">
  <td width="56" valign="top" style="width:21.05pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">#<o:p></o:p></span></p>
  </td>
  <td width="170" valign="top" style="width:63.75pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Attribute<o:p></o:p></span></p>
  </td>
  <td width="510" valign="top" style="width:191.4pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Description<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Type<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Nullable</span></span><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:1">
  <td width="56" valign="top" style="width:21.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">1<o:p></o:p></span></p>
  </td>
  <td width="170" valign="top" style="width:63.75pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">classstyle</span></span><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><o:p></o:p></span></p>
  </td>
  <td width="510" valign="top" style="width:191.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">The
  .class selector is the style of all elements of the specified class<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Array<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">No<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:2">
  <td width="56" valign="top" style="width:21.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">2<o:p></o:p></span></p>
  </td>
  <td width="170" valign="top" style="width:63.75pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">hashcode</span></span><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><o:p></o:p></span></p>
  </td>
  <td width="510" valign="top" style="width:191.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">DOM
  sequence hash encoding based on SHA-1<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">String<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">No<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:3">
  <td width="56" valign="top" style="width:21.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">3<o:p></o:p></span></p>
  </td>
  <td width="170" valign="top" style="width:63.75pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">idstyle</span></span><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><o:p></o:p></span></p>
  </td>
  <td width="510" valign="top" style="width:191.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">The
  id selector can specify a specific style for HTML elements marked with a
  specific id.<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Dictionary<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Yes<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:4">
  <td width="56" valign="top" style="width:21.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">4<o:p></o:p></span></p>
  </td>
  <td width="170" valign="top" style="width:63.75pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">name<o:p></o:p></span></p>
  </td>
  <td width="510" valign="top" style="width:191.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">This
  is the identifier of a record<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">String<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">No<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:5">
  <td width="56" valign="top" style="width:21.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">5<o:p></o:p></span></p>
  </td>
  <td width="170" valign="top" style="width:63.75pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">newtagseq</span></span><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><o:p></o:p></span></p>
  </td>
  <td width="510" valign="top" style="width:191.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">The
  depth-first traversal strategy collects the tag sequence, and attaches the
  number of layers where the tag is located<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Array<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">No<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style="mso-yfti-irow:6;mso-yfti-lastrow:yes">
  <td width="56" valign="top" style="width:21.05pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">6<o:p></o:p></span></p>
  </td>
  <td width="170" valign="top" style="width:63.75pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span class="SpellE"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">tagseq</span></span><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;"><o:p></o:p></span></p>
  </td>
  <td width="510" valign="top" style="width:191.4pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Depth-first
  traversal strategy to collect tag sequence<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">Array<o:p></o:p></span></p>
  </td>
  <td width="148" valign="top" style="width:55.6pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt">
  <p class="MsoNormal" align="center" style="text-align:center"><span lang="EN-US" style="font-family:&quot;Times New Roman&quot;,&quot;serif&quot;">No<o:p></o:p></span></p>
  </td>
 </tr>
</tbody></table>

The extraction details of classstyle and newtagseq can be found in section 3.3.1 and 3.3.2.   

## Run The Following Commands
### 
    python FingerPrintCluster.py nfold <10>  <normal sites info> <phishing sites info>  <result> <0.2>

## Usage Policy and Legal Disclaimer
This dataset is being distributed only for Research purposes, under [Creative Commons Attribution-Noncommercial-ShareAlike license (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). By clicking on the download buttons, you are agreeing to use this data only for non-commercial, research, or academic applications. You may cite the above paper if you use this dataset.  

## Contact
You can download this notebook as well as the well-organized dataset for training and testing. The toy example for visualization is in SPHDM Respository. If you find this work interesting and helpful to your work, please find the citation of the papers as below. Thank you very much. Any question you can email to actour@163.com.

 @inproceedings{feng2021SPHDM, title={Detecting Phishing Webpages via Homology Analysis of Webpage Structure}, author={Jian Feng, Yuqiang Qiao, Ou Ye, and Ying Zhang }}
