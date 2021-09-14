#!/usr/bin/python
#-*-coding:utf-8 -*-
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial.distance as dist
import re
import math
import random
import numpy as np
import sys
import time 
from pprint import pprint
from UrlReader import UrlReader
from UrlFilter import UrlFilter
from RandomGen import RandomGen
from collections import defaultdict
from collections import Counter
from math import *
import random as rd
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from lz78 import fingerprint
from test.test_StringIO import TestStringIO

#from numpy import *
################################################################################
class FingerPrintCluster:
    
    def __init__(self, ipFile=None):
        self.uf = UrlFilter(ipFile)
        self.rg = RandomGen()
        self.ur = UrlReader()
        self.cache = {}
        pass
    
    def zeroMean(self,dataMat): 
        meanVal = np.mean(dataMat,axis = 0)#计算该轴上的统计值（0为列，1为行）
        newData = dataMat - meanVal
        return newData,meanVal
    
    def percentage2n(self,eigVals,percentage):  
        sortArray=np.sort(eigVals)   #升序  
        sortArray=sortArray[-1::-1]  #逆转，即降序  
        arraySum=sum(sortArray)  
        tmpSum=0 
        num=0 
        for i in sortArray:  
            tmpSum += i  
            num += 1
            if tmpSum >= arraySum * percentage:  
                return num 
            
    def pca(self,dataMat,percent=0.99):
        '''求协方差矩阵
                        若rowvar=0，说明传入的数据一行代表一个样本，若非0
                        说明传入的数据一列代表一个样本。因为newData每一行代表一个样本，
                        所以将rowvar设置为0 '''
        newData,meanVal=self.zeroMean(dataMat)
        covMat=np.cov(newData,rowvar=0)
        eigVals,eigVects = np.linalg.eig(np.mat(covMat))
        
        n=self.percentage2n(eigVals,percent)          #要达到percent的方差百分比，需要前n个特征向量
        print (str(n) + u"vectors")
        
        eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序  
        n_eigValIndice=eigValIndice[-1:-(n+1):-1]   #最大的n个特征值的下标  
        n_eigVect=eigVects[:,n_eigValIndice]        #最大的n个特征值对应的特征向量  
        lowDDataMat=newData * n_eigVect               #低维特征空间的数据  
        #reconMat=(lowDDataMat * n_eigVect.T) + meanVal  #重构数据  
        #np.savetxt('new.csv', lowDDataMat, delimiter = ',') 
         
        return lowDDataMat
    
    a=['a','acronym','address','applet','embed', 'area','article','aside','b','basefont','big', 'blockquote','br','button','canvas','caption', 'center','cite','code','col','colgroup','datalist','dd','dfn','dialog','ul', 'div', 'dl', 'dt', 'em', 'figcaption', 'figure', 'font', 'footer', 'form', 'frame', 'frameset', 'h1', 'h6', 'header', 'hr','i', 'iframe', 'img', 'input', 'ins', 'kbd', 'keygen', 'label','legend', 'fieldset', 'li', 'link', 'main', 'map', 'mark', 'menu','menuitem','meter', 'nav', 'noframes', 'noscript', 'object', 'ol','optgroup','option', 'output', 'p', 'param', 'pre', 'progress', 'q','rp','rt', 'ruby','samp', 'section', 'select', 'small', 'source','video', 'audio','span', 'strike', 's', 'strong','sub', 'summary', 'sup', 'table', 'tbody', 'td','textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr', 'track','tt','wbr']
    
    #fecth vector list from page info file
    def fetchVectorList(self, fnin, ffin = None):
        pageInfo = None
        with open(fnin, 'r') as fin:
            pageInfo = eval(fin.read())
        
        pageFilter = None
        if ffin:
            with open(ffin, 'r') as fin:
                pageFilter = eval(fin.read())    
        
        res = []
        classRes=[]
        lz78Res=[]  
        
        tagSeqList =[]
        
        for file, content in pageInfo.items():
            
            #ignore the files not in filter
            if pageFilter:
                if not file in pageFilter:
                    continue
            #######################
            
            vector = content['newtagseq']  #字典中的内容
            TestClass = content['classStyle'] #一个列表
            Testlz78 = content['tagseq'] #为了计算压缩
            
            veclist = vector.split(',')
            ls3 = ' '.join(veclist)
            res.append(ls3)  #tfidf的所有标签序列 空格分来
            
            tagSeqList.append(veclist) #标签列表
            
            classRes.append(TestClass)  #得到所有网页的class集合
            lz78Res.append(Testlz78)
            
        dictA=['form10', 'form11', 'form12', 'form13', 'form14', 'form15', 'form16', 'form17', 'form18', 'form19', 'form20', 'form21', 'form22', 'form23', 'form24', 'form25', 'form26', 'form27', 'form28', 'form29', 'form3', 'form30', 'form31', 'form32', 'form34', 'form35', 'form36', 'form37', 'form4', 'form40', 'form44', 'form48', 'form49', 'form5', 'form50', 'form57', 'form6', 'form60', 'form62', 'form68', 'form7', 'form8', 'form9', 'div10', 'div100', 'div101', 'div102', 'div103', 'div104', 'div105', 'div106', 'div107', 'div11', 'div12', 'div13', 'div14', 'div15', 'div16', 'div17', 'div18', 'div19', 'div20', 'div21', 'div22', 'div23', 'div24', 'div25', 'div26', 'div27', 'div28', 'div29', 'div3', 'div30', 'div31', 'div32', 'div33', 'div34', 'div35', 'div36', 'div37', 'div38', 'div39', 'div4', 'div40', 'div41', 'div42', 'div43', 'div44', 'div45', 'div46', 'div47', 'div48', 'div49', 'div5', 'div50', 'div51', 'div52', 'div53', 'div54', 'div55', 'div56', 'div57', 'div58', 'div59', 'div6', 'div60', 'div61', 'div62', 'div63', 'div64', 'div65', 'div66', 'div67', 'div68', 'div69', 'div7', 'div70', 'div71', 'div72', 'div73', 'div74', 'div75', 'div76', 'div77', 'div78', 'div79', 'div8', 'div80', 'div81', 'div82', 'div83', 'div84', 'div85', 'div86', 'div87', 'div88', 'div89', 'div9', 'div90', 'div91', 'div92', 'div93', 'div94', 'div95', 'div96', 'div97', 'div98', 'div99', 'style10', 'style11', 'style12', 'style13', 'style14', 'style15', 'style16', 'style17', 'style18', 'style19', 'style20', 'style21', 'style22', 'style23', 'style24', 'style25', 'style26', 'style27', 'style28', 'style29', 'style3', 'style30', 'style31', 'style34', 'style35', 'style36', 'style38', 'style39', 'style4', 'style46', 'style5', 'style6', 'style7', 'style8', 'style9', 'span10', 'span100', 'span101', 'span102', 'span103', 'span104', 'span105', 'span106', 'span107', 'span108', 'span109', 'span11', 'span111', 'span112', 'span113', 'span12', 'span13', 'span14', 'span15', 'span16', 'span17', 'span18', 'span19', 'span20', 'span21', 'span22', 'span23', 'span24', 'span25', 'span26', 'span27', 'span28', 'span29', 'span3', 'span30', 'span31', 'span32', 'span33', 'span34', 'span35', 'span36', 'span37', 'span38', 'span39', 'span4', 'span40', 'span41', 'span42', 'span43', 'span44', 'span45', 'span46', 'span47', 'span48', 'span49', 'span5', 'span50', 'span51', 'span52', 'span53', 'span54', 'span55', 'span56', 'span57', 'span58', 'span59', 'span6', 'span60', 'span61', 'span62', 'span63', 'span64', 'span65', 'span66', 'span67', 'span68', 'span69', 'span7', 'span70', 'span71', 'span72', 'span73', 'span74', 'span75', 'span76', 'span77', 'span78', 'span79', 'span8', 'span80', 'span81', 'span82', 'span83', 'span84', 'span85', 'span86', 'span87', 'span88', 'span89', 'span9', 'span90', 'span91', 'span92', 'span93', 'span94', 'span95', 'span96', 'span97', 'span98', 'span99',  'td10', 'td11', 'td12', 'td13', 'td14', 'td15', 'td16', 'td17', 'td18', 'td19', 'td20', 'td21', 'td22', 'td23', 'td24', 'td25', 'td26', 'td27', 'td28', 'td29', 'td30', 'td31', 'td32', 'td33', 'td34', 'td35', 'td36', 'td37', 'td38', 'td39', 'td4', 'td40', 'td41', 'td42', 'td43', 'td44', 'td45', 'td46', 'td47', 'td48', 'td49', 'td50', 'td51', 'td52', 'td53', 'td54', 'td55', 'td56', 'td57', 'td58', 'td59', 'td6', 'td60', 'td61', 'td63', 'td64', 'td65', 'td66', 'td67', 'td68', 'td69', 'td7', 'td70', 'td71', 'td72', 'td73', 'td74', 'td75', 'td76', 'td77', 'td78', 'td79', 'td8', 'td80', 'td82', 'td83', 'td84', 'td86', 'td88', 'td9', 'td92','iframe10', 'iframe11', 'iframe12', 'iframe13', 'iframe14', 'iframe15', 'iframe16', 'iframe17', 'iframe18', 'iframe19', 'iframe20', 'iframe21', 'iframe22', 'iframe23', 'iframe24', 'iframe25', 'iframe26', 'iframe27', 'iframe28', 'iframe29', 'iframe3', 'iframe30', 'iframe31', 'iframe32', 'iframe33', 'iframe34', 'iframe35', 'iframe36', 'iframe37', 'iframe39', 'iframe4', 'iframe40', 'iframe43', 'iframe44', 'iframe5', 'iframe50', 'iframe6', 'iframe7', 'iframe8', 'iframe9', 'fieldset10', 'fieldset11', 'fieldset12', 'fieldset13', 'fieldset14', 'fieldset15', 'fieldset16', 'fieldset17', 'fieldset18', 'fieldset19', 'fieldset20', 'fieldset21', 'fieldset22', 'fieldset23', 'fieldset24', 'fieldset25', 'fieldset26', 'fieldset27', 'fieldset28', 'fieldset29', 'fieldset3', 'fieldset30', 'fieldset33', 'fieldset4', 'fieldset5', 'fieldset6', 'fieldset7', 'fieldset8', 'fieldset9', 'select10', 'select11', 'select12', 'select13', 'select14', 'select15', 'select16', 'select17', 'select18', 'select19', 'select20', 'select21', 'select22', 'select23', 'select24', 'select25', 'select26', 'select27', 'select28', 'select29', 'select3', 'select30', 'select31', 'select32', 'select33', 'select34', 'select35', 'select36', 'select37', 'select39', 'select4', 'select41', 'select42', 'select44', 'select45', 'select5', 'select6', 'select7', 'select8', 'select9', 'option10', 'option11', 'option12', 'option13', 'option14', 'option15', 'option16', 'option17', 'option18', 'option19', 'option20', 'option21', 'option22', 'option23', 'option24', 'option25', 'option26', 'option27', 'option28', 'option29', 'option30', 'option31', 'option32', 'option33', 'option34', 'option35', 'option36', 'option37', 'option38', 'option4', 'option40', 'option42', 'option43', 'option45', 'option46', 'option5', 'option6', 'option7', 'option8', 'option9', 'li10', 'li11', 'li12', 'li13', 'li14', 'li15', 'li16', 'li17', 'li18', 'li19', 'li20', 'li21', 'li210', 'li22', 'li23', 'li24', 'li25', 'li26', 'li27', 'li28', 'li29', 'li3', 'li30', 'li31', 'li32', 'li33', 'li34', 'li35', 'li36', 'li37', 'li39', 'li4', 'li40', 'li41', 'li42', 'li43', 'li44', 'li45', 'li46', 'li48', 'li49', 'li5', 'li50', 'li55', 'li57', 'li6', 'li60', 'li61', 'li62', 'li7', 'li8', 'li9', 'input10', 'input11', 'input12', 'input13', 'input14', 'input15', 'input16', 'input17', 'input18', 'input19', 'input20', 'input21', 'input22', 'input23', 'input24', 'input25', 'input26', 'input27', 'input28', 'input29', 'input3', 'input30', 'input31', 'input32', 'input33', 'input34', 'input35', 'input36', 'input37', 'input38', 'input39', 'input4', 'input40', 'input41', 'input42', 'input43', 'input44', 'input45', 'input46', 'input48', 'input49', 'input5', 'input51', 'input53', 'input54', 'input55', 'input57', 'input58', 'input6', 'input60', 'input61', 'input63', 'input69', 'input7', 'input71', 'input75', 'input8', 'input85', 'input87', 'input9', 'input93', 'button10', 'button11', 'button12', 'button13', 'button14', 'button15', 'button16', 'button17', 'button18', 'button19', 'button20', 'button21', 'button22', 'button23', 'button24', 'button25', 'button26', 'button27', 'button28', 'button29', 'button3', 'button30', 'button31', 'button32', 'button33', 'button37', 'button38', 'button4', 'button41', 'button42', 'button43', 'button44', 'button45', 'button5', 'button53', 'button6', 'button7', 'button8', 'button9','area10', 'area11', 'area12', 'area13', 'area14', 'area15', 'area16', 'area17', 'area18', 'area19', 'area20', 'area21', 'area22', 'area23', 'area24', 'area25', 'area26', 'area29', 'area3', 'area30', 'area31', 'area32', 'area4', 'area44', 'area5', 'area6', 'area7', 'area8', 'area9', 'table10', 'table11', 'table12', 'table13', 'table14', 'table15', 'table16', 'table17', 'table18', 'table19', 'table20', 'table21', 'table22', 'table23', 'table24', 'table25', 'table26', 'table27', 'table28', 'table29', 'table3', 'table30', 'table31', 'table32', 'table33', 'table34', 'table35', 'table36', 'table37', 'table38', 'table39', 'table4', 'table40', 'table41', 'table42', 'table43', 'table44', 'table45', 'table46', 'table47', 'table48', 'table49', 'table5', 'table50', 'table51', 'table52', 'table53', 'table54', 'table55', 'table56', 'table57', 'table58', 'table6', 'table60', 'table61', 'table62', 'table63', 'table64', 'table65', 'table66', 'table67', 'table68', 'table69', 'table7', 'table70', 'table71', 'table72', 'table73', 'table74', 'table75', 'table76', 'table77', 'table79', 'table8', 'table80', 'table81', 'table83', 'table85', 'table89', 'table9', 'tr10', 'tr11', 'tr12', 'tr13', 'tr14', 'tr15', 'tr16', 'tr17', 'tr18', 'tr19', 'tr20', 'tr21', 'tr22', 'tr23', 'tr24', 'tr25', 'tr26', 'tr27', 'tr28', 'tr29', 'tr30', 'tr31', 'tr32', 'tr33', 'tr34', 'tr35', 'tr36', 'tr37', 'tr38', 'tr39', 'tr40', 'tr41', 'tr42', 'tr43', 'tr44', 'tr45', 'tr46', 'tr47', 'tr48', 'tr49', 'tr5', 'tr50', 'tr51', 'tr52', 'tr53', 'tr54', 'tr55', 'tr56', 'tr57', 'tr58', 'tr59', 'tr6', 'tr60', 'tr62', 'tr63', 'tr64', 'tr65', 'tr66', 'tr67', 'tr68', 'tr69', 'tr7', 'tr70', 'tr71', 'tr72', 'tr73', 'tr74', 'tr75', 'tr76', 'tr77', 'tr78', 'tr79', 'tr8', 'tr81', 'tr82', 'tr83', 'tr85', 'tr87', 'tr9', 'tr91', 'br10', 'br109', 'br11', 'br12', 'br13', 'br14', 'br15', 'br16', 'br17', 'br18', 'br19', 'br20', 'br21', 'br22', 'br23', 'br24', 'br25', 'br26', 'br27', 'br28', 'br29', 'br3', 'br30', 'br31', 'br32', 'br33', 'br34', 'br35', 'br36', 'br37', 'br38', 'br39', 'br4', 'br40', 'br41', 'br42', 'br43', 'br44', 'br45', 'br46', 'br47', 'br48', 'br49', 'br5', 'br50', 'br51', 'br52', 'br54', 'br55', 'br56', 'br57', 'br59', 'br6', 'br60', 'br62', 'br63', 'br7', 'br78', 'br8', 'br84', 'br9',  'hr10', 'hr11', 'hr12', 'hr13', 'hr14', 'hr15', 'hr16', 'hr17', 'hr18', 'hr19', 'hr20', 'hr21', 'hr22', 'hr23', 'hr24', 'hr25', 'hr26', 'hr27', 'hr28', 'hr29', 'hr3', 'hr30', 'hr31', 'hr34', 'hr35', 'hr37', 'hr38', 'hr4', 'hr40', 'hr41', 'hr43', 'hr45', 'hr5', 'hr54', 'hr55', 'hr56', 'hr6', 'hr7', 'hr8', 'hr9',  'h1020', 'h110', 'h111', 'h112', 'h113', 'h114', 'h115', 'h116', 'h117', 'h118', 'h119', 'h120', 'h121', 'h122', 'h123', 'h124', 'h125', 'h126', 'h127', 'h128', 'h129', 'h13', 'h130', 'h131', 'h132', 'h133', 'h134', 'h135', 'h137', 'h138', 'h139', 'h14', 'h140', 'h141', 'h1413', 'h15', 'h16', 'h1613', 'h17', 'h18', 'h19', 'th10', 'th11', 'th12', 'th13', 'th14', 'th15', 'th16', 'th17', 'th18', 'th19', 'th20', 'th21', 'th22', 'th23', 'th24', 'th25', 'th26', 'th27', 'th28', 'th29', 'th30', 'th31', 'th32', 'th34', 'th36', 'th38', 'th6', 'th7', 'th8', 'th9', 'canvas10', 'canvas11', 'canvas12', 'canvas13', 'canvas14', 'canvas15', 'canvas16', 'canvas17', 'canvas18', 'canvas19', 'canvas20', 'canvas21', 'canvas22', 'canvas23', 'canvas24', 'canvas25', 'canvas27', 'canvas3', 'canvas31', 'canvas35', 'canvas36', 'canvas4', 'canvas5', 'canvas6', 'canvas7', 'canvas8', 'canvas9','h610', 'h611', 'h612', 'h613', 'h614', 'h615', 'h616', 'h617', 'h618', 'h619', 'h620', 'h621', 'h622', 'h623', 'h624', 'h625', 'h626', 'h627', 'h628', 'h629', 'h630', 'h64', 'h65', 'h66', 'h67', 'h68', 'h69', 'tbody10', 'tbody11', 'tbody12', 'tbody13', 'tbody14', 'tbody15', 'tbody16', 'tbody17', 'tbody18', 'tbody19', 'tbody20', 'tbody21', 'tbody22', 'tbody23', 'tbody24', 'tbody25', 'tbody26', 'tbody27', 'tbody28', 'tbody29', 'tbody30', 'tbody31', 'tbody32', 'tbody33', 'tbody34', 'tbody35', 'tbody36', 'tbody37', 'tbody38', 'tbody39', 'tbody4', 'tbody40', 'tbody41', 'tbody42', 'tbody43', 'tbody44', 'tbody45', 'tbody46', 'tbody47', 'tbody48', 'tbody49', 'tbody5', 'tbody50', 'tbody51', 'tbody52', 'tbody53', 'tbody54', 'tbody55', 'tbody56', 'tbody57', 'tbody58', 'tbody59', 'tbody6', 'tbody61', 'tbody62', 'tbody63', 'tbody64', 'tbody65', 'tbody66', 'tbody67', 'tbody68', 'tbody69', 'tbody7', 'tbody70', 'tbody71', 'tbody72', 'tbody73', 'tbody74', 'tbody75', 'tbody76', 'tbody77', 'tbody78', 'tbody8', 'tbody80', 'tbody81', 'tbody82', 'tbody84', 'tbody86', 'tbody9', 'tbody90', 'ul10', 'ul11', 'ul12', 'ul13', 'ul14', 'ul15', 'ul16', 'ul17', 'ul18', 'ul19', 'ul20', 'ul21', 'ul22', 'ul23', 'ul24', 'ul25', 'ul26', 'ul27', 'ul28', 'ul29', 'ul3', 'ul30', 'ul31', 'ul32', 'ul33', 'ul34', 'ul35', 'ul36', 'ul38', 'ul39', 'ul4', 'ul40', 'ul41', 'ul42', 'ul43', 'ul44', 'ul45', 'ul47', 'ul48', 'ul49', 'ul5', 'ul53', 'ul54', 'ul56', 'ul59', 'ul6', 'ul60', 'ul61', 'ul7', 'ul8', 'ul9',  'nav10', 'nav11', 'nav12', 'nav13', 'nav14', 'nav15', 'nav16', 'nav17', 'nav18', 'nav19', 'nav20', 'nav21', 'nav22', 'nav23', 'nav24', 'nav25', 'nav26', 'nav27', 'nav3', 'nav4', 'nav5', 'nav6', 'nav7', 'nav8', 'nav9',  'header10', 'header11', 'header12', 'header13', 'header14', 'header15', 'header16', 'header17', 'header18', 'header19', 'header20', 'header21', 'header22', 'header23', 'header24', 'header26', 'header27', 'header29', 'header3', 'header30', 'header4', 'header5', 'header6', 'header7', 'header8', 'header9','foooter4', 'footer10', 'footer11', 'footer12', 'footer13', 'footer14', 'footer15', 'footer16', 'footer17', 'footer18', 'footer19', 'footer20', 'footer22', 'footer24', 'footer27', 'footer3', 'footer30', 'footer4', 'footer43', 'footer5', 'footer6', 'footer7', 'footer8', 'footer9',  'label10', 'label11', 'label12', 'label13', 'label14', 'label15', 'label16', 'label17', 'label18', 'label19', 'label20', 'label21', 'label22', 'label23', 'label24', 'label25', 'label26', 'label27', 'label28', 'label29', 'label3', 'label30', 'label31', 'label32', 'label33', 'label34', 'label35', 'label36', 'label37', 'label38', 'label39', 'label4', 'label40', 'label41', 'label42', 'label43', 'label46', 'label5', 'label6', 'label7', 'label8', 'label9', 'lable8', 'font10', 'font104', 'font109', 'font11', 'font12', 'font13', 'font14', 'font15', 'font16', 'font17', 'font18', 'font19', 'font20', 'font21', 'font22', 'font23', 'font24', 'font25', 'font26', 'font27', 'font28', 'font29', 'font3', 'font30', 'font31', 'font32', 'font33', 'font34', 'font35', 'font36', 'font37', 'font38', 'font39', 'font4', 'font40', 'font41', 'font42', 'font43', 'font44', 'font45', 'font46', 'font47', 'font48', 'font49', 'font5', 'font50', 'font51', 'font53', 'font54', 'font55', 'font56', 'font57', 'font58', 'font59', 'font6', 'font60', 'font61', 'font62', 'font63', 'font64', 'font65', 'font69', 'font7', 'font74', 'font79', 'font8', 'font84', 'font89', 'font9', 'font94', 'font99', 'article10', 'article11', 'article12', 'article13', 'article14', 'article15', 'article16', 'article17', 'article18', 'article19', 'article20', 'article21', 'article22', 'article23', 'article24', 'article25', 'article26', 'article27', 'article28', 'article29', 'article3', 'article31', 'article4', 'article5', 'article6', 'article7', 'article8', 'article9', 'title10', 'title11', 'title12', 'title13', 'title14', 'title15', 'title16', 'title17', 'title18', 'title19', 'title20', 'title21', 'title22', 'title23', 'title24', 'title26', 'title28', 'title29', 'title3', 'title30', 'title31', 'title37', 'title38', 'title4', 'title40', 'title42', 'title5', 'title6', 'title7', 'title8', 'title9', 'meta10', 'meta11', 'meta12', 'meta13', 'meta14', 'meta15', 'meta16', 'meta17', 'meta18', 'meta19', 'meta20', 'meta21', 'meta22', 'meta23', 'meta24', 'meta25', 'meta26', 'meta27', 'meta28', 'meta29', 'meta3', 'meta31', 'meta37', 'meta38', 'meta4', 'meta5', 'meta6', 'meta7', 'meta8', 'meta9', 'center10', 'center11', 'center12', 'center13', 'center14', 'center15', 'center16', 'center17', 'center18', 'center19', 'center20', 'center21', 'center22', 'center23', 'center24', 'center25', 'center26', 'center27', 'center28', 'center29', 'center3', 'center30', 'center31', 'center32', 'center33', 'center34', 'center35', 'center36', 'center37', 'center38', 'center39', 'center4', 'center40', 'center41', 'center42', 'center43', 'center44', 'center45', 'center46', 'center47', 'center48', 'center49', 'center5', 'center50', 'center51', 'center52', 'center53', 'center54', 'center55', 'center56', 'center57', 'center58', 'center59', 'center6', 'center60', 'center61', 'center62', 'center63', 'center64', 'center65', 'center66', 'center67', 'center68', 'center69', 'center7', 'center70', 'center71', 'center8', 'center9','source10', 'source11', 'source12', 'source13', 'source14', 'source15', 'source16', 'source17', 'source18', 'source19', 'source20', 'source21', 'source22', 'source23', 'source24', 'source25', 'source26', 'source27', 'source28', 'source29', 'source30', 'source31', 'source32', 'source33', 'source34', 'source35', 'source4', 'source42', 'source5', 'source6', 'source7', 'source8', 'source9','small10', 'small11', 'small12', 'small13', 'small14', 'small15', 'small16', 'small17', 'small18', 'small19', 'small20', 'small21', 'small22', 'small23', 'small24', 'small25', 'small26', 'small27', 'small3', 'small30', 'small31', 'small32', 'small33', 'small4', 'small5', 'small6', 'small7', 'small8', 'small9','noscript10', 'noscript11', 'noscript12', 'noscript13', 'noscript14', 'noscript15', 'noscript16', 'noscript17', 'noscript18', 'noscript19', 'noscript20', 'noscript21', 'noscript22', 'noscript23', 'noscript24', 'noscript25', 'noscript26', 'noscript27', 'noscript28', 'noscript29', 'noscript3', 'noscript30', 'noscript31', 'noscript32', 'noscript36', 'noscript4', 'noscript43', 'noscript5', 'noscript6', 'noscript7', 'noscript8', 'noscript9','em10', 'em11', 'em12', 'em13', 'em14', 'em15', 'em16', 'em17', 'em18', 'em19', 'em20', 'em21', 'em22', 'em23', 'em24', 'em25', 'em26', 'em27', 'em28', 'em29', 'em3', 'em30', 'em31', 'em32', 'em33', 'em34', 'em35', 'em37', 'em4', 'em42', 'em45', 'em5', 'em6', 'em69', 'em7', 'em71', 'em77', 'em78', 'em79', 'em8', 'em84', 'em85', 'em9','link10', 'link11', 'link12', 'link13', 'link14', 'link15', 'link16', 'link17', 'link18', 'link19', 'link20', 'link21', 'link22', 'link23', 'link24', 'link25', 'link27', 'link28', 'link3', 'link31', 'link34', 'link36', 'link39', 'link4', 'link43', 'link5', 'link6', 'link7', 'link8', 'link9','script10', 'script11', 'script116', 'script12', 'script13', 'script14', 'script15', 'script16', 'script17', 'script18', 'script19', 'script20', 'script21', 'script22', 'script23', 'script24', 'script25', 'script26', 'script27', 'script28', 'script29', 'script3', 'script30', 'script31', 'script32', 'script33', 'script34', 'script35', 'script36', 'script37', 'script38', 'script39', 'script4', 'script40', 'script41', 'script43', 'script47', 'script48', 'script49', 'script5', 'script50', 'script52', 'script53', 'script57', 'script6', 'script62', 'script7', 'script8', 'script9','strong10', 'strong11', 'strong12', 'strong13', 'strong14', 'strong15', 'strong16', 'strong17', 'strong18', 'strong19', 'strong20', 'strong21', 'strong215', 'strong216', 'strong22', 'strong23', 'strong24', 'strong25', 'strong26', 'strong27', 'strong28', 'strong29', 'strong3', 'strong30', 'strong31', 'strong32', 'strong33', 'strong34', 'strong35', 'strong36', 'strong37', 'strong38', 'strong39', 'strong4', 'strong40', 'strong41', 'strong42', 'strong43', 'strong44', 'strong45', 'strong46', 'strong47', 'strong49', 'strong5', 'strong52', 'strong53', 'strong54', 'strong55', 'strong56', 'strong57', 'strong58', 'strong59', 'strong6', 'strong61', 'strong62', 'strong66', 'strong7', 'strong70', 'strong73', 'strong75', 'strong76', 'strong79', 'strong8', 'strong9', 'strong93','img10', 'img108', 'img11', 'img12', 'img13', 'img14', 'img15', 'img16', 'img17', 'img18', 'img19', 'img20', 'img21', 'img22', 'img23', 'img24', 'img25', 'img26', 'img27', 'img28', 'img29', 'img3', 'img30', 'img31', 'img32', 'img33', 'img34', 'img35', 'img36', 'img37', 'img38', 'img39', 'img4', 'img40', 'img41', 'img42', 'img43', 'img44', 'img45', 'img46', 'img47', 'img48', 'img49', 'img5', 'img50', 'img51', 'img52', 'img53', 'img54', 'img55', 'img56', 'img57', 'img58', 'img59', 'img6', 'img60', 'img61', 'img63', 'img64', 'img65', 'img66', 'img67', 'img7', 'img74', 'img76', 'img8', 'img83', 'img9']         
        dictB=['var10', 'var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25', 'u26', 'u27', 'u28', 'u29', 'u3', 'u30', 'u31', 'u32', 'u33', 'u34', 'u35', 'u36', 'u37', 'u38', 'u39', 'u4', 'u42', 'u43', 'u44', 'u48', 'u5', 'u53', 'u6', 'u7', 'u8', 'u9', 'tt11', 'tt7', 'tt8', 'summary10', 'summary11', 'summary15', 'summary7', 'summary8', 'summary9', 'samp10', 'samp11', 'samp12', 'samp13', 'samp14', 'samp16', 'samp5', 'samp6', 'samp8', 'samp9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's31', 's32', 's4', 's40', 's5', 's7', 's8', 's9', 'pre10', 'pre11', 'pre12', 'pre13', 'pre14', 'pre15', 'pre16', 'pre17', 'pre19', 'pre20', 'pre22', 'pre3', 'pre30', 'pre4', 'pre5', 'pre6', 'pre7', 'pre8', 'pre9', 'mark10', 'mark11', 'mark12', 'mark13', 'mark16', 'mark17', 'mark18', 'mark21', 'mark23', 'mark6', 'mark7', 'mark8', 'mark9', 'kbd10', 'kbd11', 'kbd14', 'kbd16', 'kbd7', 'kbd9', 'ins10', 'ins11', 'ins12', 'ins13', 'ins14', 'ins15', 'ins16', 'ins17', 'ins18', 'ins19', 'ins20', 'ins21', 'ins22', 'ins23', 'ins24', 'ins25', 'ins26', 'ins27', 'ins28', 'ins29', 'ins3', 'ins30', 'ins31', 'ins32', 'ins33', 'ins34', 'ins35', 'ins36', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'del10', 'del11', 'del12', 'del13', 'del14', 'del15', 'del16', 'del17', 'del18', 'del19', 'del20', 'del21', 'del22', 'del23', 'del25', 'del26', 'del28', 'del29', 'del33', 'del35', 'del5', 'del6', 'del7', 'del8', 'del9', 'b10', 'b101', 'b106', 'b11', 'b110', 'b111', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b212', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b3', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b4', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b5', 'b50', 'b51', 'b52', 'b53', 'b55', 'b56', 'b57', 'b6', 'b61', 'b64', 'b65', 'b66', 'b67', 'b7', 'b70', 'b71', 'b76', 'b8', 'b81', 'b85', 'b86', 'b9', 'b91', 'b96', 'base10', 'base11', 'base12', 'base13', 'base14', 'base15', 'base16', 'base17', 'base18', 'base19', 'base21', 'base27', 'base3', 'base31', 'base4', 'base5', 'base6', 'base7', 'base8', 'base9', 'bdi10', 'bdi12', 'bdi13', 'bdi16', 'bdi17', 'bdi18', 'bdi19', 'bdi20', 'bdi21', 'bdi8', 'bdo24', 'bdo6', 'bdo8', 'big10', 'big100', 'big103', 'big105', 'big108', 'big11', 'big110', 'big12', 'big13', 'big14', 'big15', 'big16', 'big17', 'big18', 'big19', 'big20', 'big21', 'big22', 'big24', 'big25', 'big26', 'big27', 'big28', 'big3', 'big30', 'big33', 'big35', 'big38', 'big4', 'big40', 'big43', 'big45', 'big48', 'big5', 'big50', 'big53', 'big55', 'big58', 'big6', 'big60', 'big63', 'big65', 'big68', 'big7', 'big70', 'big73', 'big75', 'big78', 'big8', 'big80', 'big83', 'big85', 'big88', 'big9', 'big90', 'big93', 'big95', 'big98','code10', 'code11', 'code12', 'code13', 'code14', 'code15', 'code16', 'code17', 'code18', 'code21', 'code22', 'code3', 'code4', 'code5', 'code6', 'code7', 'code8', 'code9', 'href20', 'href6', 'href7', 'href8', 'href9', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i16', 'i17', 'i18', 'i19', 'i20', 'i21', 'i22', 'i23', 'i24', 'i25', 'i26', 'i27', 'i28', 'i29', 'i3', 'i30', 'i31', 'i32', 'i33', 'i34', 'i35', 'i36', 'i37', 'i38', 'i39', 'i4', 'i40', 'i42', 'i45', 'i5', 'i6', 'i7', 'i8', 'i9','thead10', 'thead11', 'thead12', 'thead13', 'thead14', 'thead15', 'thead16', 'thead17', 'thead18', 'thead19', 'thead20', 'thead21', 'thead22', 'thead23', 'thead24', 'thead25', 'thead26', 'thead27', 'thead28', 'thead29', 'thead32', 'thead34', 'thead4', 'thead5', 'thead6', 'thead7', 'thead8', 'thead9', 'tfoot10', 'tfoot11', 'tfoot12', 'tfoot13', 'tfoot14', 'tfoot15', 'tfoot16', 'tfoot18', 'tfoot21', 'tfoot5', 'tfoot6', 'tfoot7', 'tfoot8', 'tfoot9', 'sup10', 'sup11', 'sup12', 'sup13', 'sup14', 'sup15', 'sup16', 'sup17', 'sup18', 'sup19', 'sup20', 'sup21', 'sup22', 'sup23', 'sup24', 'sup25', 'sup26', 'sup27', 'sup29', 'sup33', 'sup35', 'sup36', 'sup37', 'sup38', 'sup4', 'sup5', 'sup6', 'sup7', 'sup8', 'sup9', 'sub10', 'sub11', 'sub12', 'sub13', 'sub14', 'sub15', 'sub16', 'sub19', 'sub5', 'sub6', 'sub7', 'sub8', 'sub9', 'legend10', 'legend11', 'legend12', 'legend13', 'legend14', 'legend15', 'legend16', 'legend17', 'legend18', 'legend19', 'legend20', 'legend21', 'legend22', 'legend23', 'legend24', 'legend25', 'legend26', 'legend27', 'legend28', 'legend29', 'legend30', 'legend31', 'legend5', 'legend6', 'legend7', 'legend8', 'legend9','details10', 'details11', 'details12', 'details13', 'details14', 'details16', 'details6', 'details7', 'details8', 'address10', 'address11', 'address12', 'address13', 'address14', 'address15', 'address16', 'address17', 'address18', 'address19', 'address20', 'address21', 'address22', 'address3', 'address34', 'address4', 'address5', 'address6', 'address7', 'address8', 'address9', 'caption10', 'caption11', 'caption12', 'caption13', 'caption14', 'caption15', 'caption16', 'caption17', 'caption18', 'caption19', 'caption20', 'caption21', 'caption22', 'caption23', 'caption25', 'caption27', 'caption5', 'caption6', 'caption7', 'caption8', 'caption9','wbr10', 'wbr11', 'wbr12', 'wbr13', 'wbr14', 'wbr15', 'wbr16', 'wbr17', 'wbr18', 'wbr19', 'wbr20', 'wbr21', 'wbr22', 'wbr23', 'wbr24', 'wbr25', 'wbr26', 'wbr27', 'wbr28', 'wbr29', 'wbr34', 'wbr36', 'wbr37', 'wbr7', 'wbr8', 'wbr9video10', 'video11', 'video12', 'video13', 'video14', 'video15', 'video16', 'video17', 'video18', 'video19', 'video20', 'video21', 'video23', 'video24', 'video25', 'video3', 'video4', 'video5', 'video6', 'video7', 'video8', 'video9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q18', 'q19', 'q7', 'q8', 'q9', 'progress12', 'progress5', 'progress9', 'param10', 'param11', 'param12', 'param13', 'param14', 'param15', 'param16', 'param17', 'param18', 'param19', 'param20', 'param21', 'param22', 'param23', 'param24', 'param25', 'param26', 'param27', 'param3', 'param30', 'param31', 'param32', 'param35', 'param37', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'object10', 'object11', 'object12', 'object13', 'object14', 'object15', 'object16', 'object17', 'object18', 'object19', 'object20', 'object21', 'object22', 'object23', 'object24', 'object25', 'object26', 'object29', 'object3', 'object30', 'object31', 'object34', 'object36', 'object4', 'object5', 'object6', 'object7', 'object8', 'object9', 'map10', 'map11', 'map12', 'map13', 'map14', 'map15', 'map16', 'map17', 'map18', 'map19', 'map20', 'map21', 'map22', 'map23', 'map24', 'map25', 'map28', 'map29', 'map3', 'map30', 'map31', 'map4', 'map43', 'map5', 'map6', 'map7', 'map8', 'map9', 'image10', 'image11', 'image12', 'image13', 'image14', 'image15', 'image16', 'image17', 'image18', 'image19', 'image20', 'image21', 'image22', 'image24', 'image25', 'image28', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'figcaption10', 'figcaption11', 'figcaption12', 'figcaption13', 'figcaption14', 'figcaption15', 'figcaption16', 'figcaption17', 'figcaption18', 'figcaption19', 'figcaption20', 'figcaption21', 'figcaption22', 'figcaption23', 'figcaption24', 'figcaption25', 'figcaption27', 'figcaption35', 'figcaption53', 'figcaption6', 'figcaption7', 'figcaption8', 'figcaption9', 'figure10', 'figure11', 'figure12', 'figure13', 'figure14', 'figure15', 'figure16', 'figure17', 'figure18', 'figure19', 'figure20', 'figure21', 'figure22', 'figure23', 'figure24', 'figure25', 'figure26', 'figure27', 'figure3', 'figure30', 'figure32', 'figure33', 'figure36', 'figure4', 'figure5', 'figure52', 'figure6', 'figure7', 'figure8', 'figure9', 'filter10', 'filter12', 'filter13', 'filter15', 'filter16', 'filter17', 'filter4', 'filter7', 'filter8', 'filter9', 'firure10', 'firure11', 'firure13', 'firure14', 'firure20', 'firure9', 'embed10', 'embed11', 'embed12', 'embed13', 'embed14', 'embed15', 'embed16', 'embed17', 'embed18', 'embed19', 'embed20', 'embed21', 'embed22', 'embed24', 'embed25', 'embed27', 'embed3', 'embed31', 'embed32', 'embed35', 'embed4', 'embed5', 'embed6', 'embed7', 'embed8', 'embed9', 'a10', 'a108', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a3', 'a30', 'a31', 'a32', 'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a4', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48', 'a49', 'a5', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a6', 'a60', 'a61', 'a62', 'a63', 'a64', 'a65', 'a67', 'a7', 'a73', 'a76', 'a8', 'a83', 'a88', 'a9', 'a90', 'a92', 'a94', 'a96', 'a98', 'abbr10', 'abbr11', 'abbr12', 'abbr13', 'abbr14', 'abbr15', 'abbr16', 'abbr17', 'abbr18', 'abbr19', 'abbr20', 'abbr21', 'abbr22', 'abbr23', 'abbr25', 'abbr3', 'abbr30', 'abbr32', 'abbr34', 'abbr35', 'abbr36', 'abbr37', 'abbr4', 'abbr5', 'abbr6', 'abbr7', 'abbr8', 'abbr9', 'acronym10', 'acronym11', 'acronym12', 'acronym13', 'acronym14', 'acronym15', 'acronym16', 'acronym17', 'acronym18', 'acronym20', 'acronym22', 'acronym33', 'acronym34', 'acronym5', 'acronym6', 'acronym7', 'acronym8', 'acronym9', 'applet15', 'applet7', 'applet8', 'aside10', 'aside11', 'aside12', 'aside13', 'aside14', 'aside15', 'aside16', 'aside17', 'aside18', 'aside19', 'aside20', 'aside21', 'aside22', 'aside24', 'aside25', 'aside3', 'aside30', 'aside4', 'aside5', 'aside6', 'aside7', 'aside8', 'aside9', 'audio10', 'audio11', 'audio12', 'audio13', 'audio14', 'audio15', 'audio16', 'audio17', 'audio20', 'audio21', 'audio24', 'audio3', 'audio31', 'audio4', 'audio41', 'audio5', 'audio6', 'audio7', 'audio8', 'audio9', 'blockquote10', 'blockquote11', 'blockquote12', 'blockquote13', 'blockquote14', 'blockquote15', 'blockquote16', 'blockquote17', 'blockquote18', 'blockquote19', 'blockquote20', 'blockquote21', 'blockquote22', 'blockquote23', 'blockquote25', 'blockquote3', 'blockquote4', 'blockquote5', 'blockquote6', 'blockquote7', 'blockquote8', 'blockquote9', 'cite10', 'cite11', 'cite12', 'cite13', 'cite14', 'cite15', 'cite16', 'cite17', 'cite18', 'cite24', 'cite4', 'cite6', 'cite7', 'cite8', 'cite9','textarea10', 'textarea11', 'textarea12', 'textarea13', 'textarea14', 'textarea15', 'textarea16', 'textarea17', 'textarea18', 'textarea19', 'textarea20', 'textarea21', 'textarea22', 'textarea23', 'textarea24', 'textarea25', 'textarea26', 'textarea27', 'textarea28', 'textarea29', 'textarea3', 'textarea32', 'textarea39', 'textarea4', 'textarea5', 'textarea53', 'textarea6', 'textarea7', 'textarea71', 'textarea8', 'textarea9', 'section10', 'section11', 'section12', 'section13', 'section14', 'section15', 'section16', 'section17', 'section18', 'section19', 'section20', 'section21', 'section22', 'section23', 'section24', 'section25', 'section26', 'section27', 'section29', 'section3', 'section30', 'section33', 'section4', 'section43', 'section5', 'section6', 'section7', 'section8', 'section9', 'p10', 'p11', 'p117', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p216', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p3', 'p30', 'p31', 'p314', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p4', 'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p5', 'p50', 'p52', 'p54', 'p55', 'p56', 'p58', 'p59', 'p6', 'p60', 'p61', 'p7', 'p8', 'p9', 'optgroup10', 'optgroup11', 'optgroup12', 'optgroup13', 'optgroup14', 'optgroup15', 'optgroup16', 'optgroup17', 'optgroup18', 'optgroup19', 'optgroup20', 'optgroup21', 'optgroup22', 'optgroup24', 'optgroup25', 'optgroup26', 'optgroup30', 'optgroup5', 'optgroup6', 'optgroup7', 'optgroup8', 'optgroup9', 'ol10', 'ol11', 'ol12', 'ol13', 'ol14', 'ol15', 'ol16', 'ol17', 'ol18', 'ol19', 'ol20', 'ol21', 'ol22', 'ol23', 'ol24', 'ol25', 'ol26', 'ol27', 'ol28', 'ol3', 'ol30', 'ol31', 'ol32', 'ol34', 'ol35', 'ol38', 'ol4', 'ol40', 'ol41', 'ol5', 'ol6', 'ol7', 'ol8', 'ol9', 'menu10', 'menu11', 'menu12', 'menu13', 'menu14', 'menu15', 'menu16', 'menu17', 'menu18', 'menu19', 'menu20', 'menu21', 'menu3', 'menu37', 'menu4', 'menu5', 'menu6', 'menu7', 'menu8', 'menu9', 'menuitem11', 'menuitem12', 'menuitem13', 'menuitem14', 'menuitem15', 'menuitem16', 'menuitem20', 'menuitem21', 'menuitem38', 'menuitem6', 'menuitem7', 'menuitem8', 'menuitem9','h03', 'h210', 'h211', 'h212', 'h213', 'h214', 'h215', 'h216', 'h217', 'h218', 'h219', 'h220', 'h221', 'h222', 'h223', 'h224', 'h225', 'h226', 'h227', 'h228', 'h229', 'h23', 'h230', 'h231', 'h232', 'h233', 'h234', 'h236', 'h237', 'h24', 'h241', 'h243', 'h244', 'h249', 'h25', 'h252', 'h253', 'h26', 'h27', 'h28', 'h29', 'h310', 'h311', 'h312', 'h313', 'h314', 'h315', 'h316', 'h317', 'h318', 'h319', 'h320', 'h321', 'h322', 'h323', 'h324', 'h325', 'h326', 'h327', 'h328', 'h329', 'h33', 'h330', 'h331', 'h3310', 'h332', 'h333', 'h334', 'h335', 'h337', 'h34', 'h341', 'h3410', 'h342', 'h35', 'h355', 'h359', 'h36', 'h37', 'h38', 'h39', 'h410', 'h411', 'h412', 'h413', 'h414', 'h415', 'h416', 'h417', 'h418', 'h419', 'h420', 'h421', 'h422', 'h423', 'h424', 'h425', 'h426', 'h427', 'h428', 'h429', 'h43', 'h430', 'h431', 'h432', 'h433', 'h434', 'h436', 'h44', 'h441', 'h442', 'h443', 'h444', 'h445', 'h45', 'h452', 'h46', 'h47', 'h48', 'h49', 'h510', 'h511', 'h512', 'h513', 'h514', 'h515', 'h516', 'h517', 'h518', 'h519', 'h520', 'h521', 'h522', 'h523', 'h524', 'h525', 'h526', 'h527', 'h528', 'h53', 'h530', 'h531', 'h536', 'h54', 'h55', 'h56', 'h57', 'h58', 'h59', 'h710', 'h711', 'h712', 'h713', 'h715', 'h77', 'h79', 'h8', 'h810', 'h815', 'h817', 'h85', 'h87', 'h921', 'dfn10', 'dfn11', 'dfn12', 'dfn13', 'dfn14', 'dfn15', 'dfn16', 'dfn17', 'dfn18', 'dfn5', 'dfn7', 'dfn8', 'dfn9', 'dialog13', 'dialog3', 'dialog5', 'dialog6', 'dl10', 'dl11', 'dl12', 'dl13', 'dl14', 'dl15', 'dl16', 'dl17', 'dl18', 'dl19', 'dl20', 'dl21', 'dl22', 'dl23', 'dl24', 'dl25', 'dl26', 'dl3', 'dl32', 'dl36', 'dl4', 'dl5', 'dl6', 'dl7', 'dl8', 'dl9', 'dt10', 'dt11', 'dt12', 'dt13', 'dt14', 'dt15', 'dt16', 'dt17', 'dt18', 'dt19', 'dt20', 'dt21', 'dt22', 'dt23', 'dt24', 'dt25', 'dt26', 'dt27', 'dt29', 'dt33', 'dt37', 'dt4', 'dt5', 'dt6', 'dt7', 'dt8', 'dt9', 'd11', 'd14', 'd17', 'd6', 'datalist11', 'datalist14', 'datalist6', 'datalist7', 'datalist9', 'dd10', 'dd11', 'dd12', 'dd13', 'dd14', 'dd15', 'dd16', 'dd17', 'dd18', 'dd19', 'dd20', 'dd21', 'dd22', 'dd23', 'dd24', 'dd25', 'dd26', 'dd27', 'dd33', 'dd4', 'dd5', 'dd6', 'dd7', 'dd8', 'dd9', 'body2', 'body5','col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col24', 'col28', 'col29', 'col36', 'col40', 'col5', 'col6', 'col7', 'col8', 'col9', 'colgroup10', 'colgroup11', 'colgroup12', 'colgroup13', 'colgroup14', 'colgroup15', 'colgroup16', 'colgroup17', 'colgroup18', 'colgroup19', 'colgroup20', 'colgroup23', 'colgroup27', 'colgroup28', 'colgroup35', 'colgroup39', 'colgroup4', 'colgroup5', 'colgroup6', 'colgroup7', 'colgroup8', 'colgroup9']

        strk="a10 a108 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29 a3 a30 a31 a32 a33 a34 a35 a36 a37 a38 a39 a4 a40 a41 a42 a43 a44 a45 a46 a47 a48 a49 a5 a50 a51 a52 a53 a54 a55 a56 a57 a58 a59 a6 a60 a61 a62 a63 a64 a65 a67 a7 a73 a76 a8 a83 a88 a9 a90 a92 a94 a96 a98 abbr10 abbr11 abbr12 abbr13 abbr14 abbr15 abbr16 abbr17 abbr18 abbr19 abbr20 abbr21 abbr22 abbr23 abbr25 abbr3 abbr30 abbr32 abbr34 abbr35 abbr36 abbr37 abbr4 abbr5 abbr6 abbr7 abbr8 abbr9 acronym10 acronym11 acronym12 acronym13 acronym14 acronym15 acronym16 acronym17 acronym18 acronym20 acronym22 acronym33 acronym34 acronym5 acronym6 acronym7 acronym8 acronym9 address10 address11 address12 address13 address14 address15 address16 address17 address18 address19 address20 address21 address22 address3 address34 address4 address5 address6 address7 address8 address9 applet15 applet7 applet8 area10 area11 area12 area13 area14 area15 area16 area17 area18 area19 area20 area21 area22 area23 area24 area25 area26 area29 area3 area30 area31 area32 area4 area44 area5 area6 area7 area8 area9 article10 article11 article12 article13 article14 article15 article16 article17 article18 article19 article20 article21 article22 article23 article24 article25 article26 article27 article28 article29 article3 article31 article4 article5 article6 article7 article8 article9 aside10 aside11 aside12 aside13 aside14 aside15 aside16 aside17 aside18 aside19 aside20 aside21 aside22 aside24 aside25 aside3 aside30 aside4 aside5 aside6 aside7 aside8 aside9 audio10 audio11 audio12 audio13 audio14 audio15 audio16 audio17 audio20 audio21 audio24 audio3 audio31 audio4 audio41 audio5 audio6 audio7 audio8 audio9 b10 b101 b106 b11 b110 b111 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b212 b22 b23 b24 b25 b26 b27 b28 b29 b3 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b4 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b5 b50 b51 b52 b53 b55 b56 b57 b6 b61 b64 b65 b66 b67 b7 b70 b71 b76 b8 b81 b85 b86 b9 b91 b96 base10 base11 base12 base13 base14 base15 base16 base17 base18 base19 base21 base27 base3 base31 base4 base5 base6 base7 base8 base9 bdi10 bdi12 bdi13 bdi16 bdi17 bdi18 bdi19 bdi20 bdi21 bdi8 bdo24 bdo6 bdo8 big10 big100 big103 big105 big108 big11 big110 big12 big13 big14 big15 big16 big17 big18 big19 big20 big21 big22 big24 big25 big26 big27 big28 big3 big30 big33 big35 big38 big4 big40 big43 big45 big48 big5 big50 big53 big55 big58 big6 big60 big63 big65 big68 big7 big70 big73 big75 big78 big8 big80 big83 big85 big88 big9 big90 big93 big95 big98 blockquote10 blockquote11 blockquote12 blockquote13 blockquote14 blockquote15 blockquote16 blockquote17 blockquote18 blockquote19 blockquote20 blockquote21 blockquote22 blockquote23 blockquote25 blockquote3 blockquote4 blockquote5 blockquote6 blockquote7 blockquote8 blockquote9 body2 body5 br10 br109 br11 br12 br13 br14 br15 br16 br17 br18 br19 br20 br21 br22 br23 br24 br25 br26 br27 br28 br29 br3 br30 br31 br32 br33 br34 br35 br36 br37 br38 br39 br4 br40 br41 br42 br43 br44 br45 br46 br47 br48 br49 br5 br50 br51 br52 br54 br55 br56 br57 br59 br6 br60 br62 br63 br7 br78 br8 br84 br9 button10 button11 button12 button13 button14 button15 button16 button17 button18 button19 button20 button21 button22 button23 button24 button25 button26 button27 button28 button29 button3 button30 button31 button32 button33 button37 button38 button4 button41 button42 button43 button44 button45 button5 button53 button6 button7 button8 button9 canvas10 canvas11 canvas12 canvas13 canvas14 canvas15 canvas16 canvas17 canvas18 canvas19 canvas20 canvas21 canvas22 canvas23 canvas24 canvas25 canvas27 canvas3 canvas31 canvas35 canvas36 canvas4 canvas5 canvas6 canvas7 canvas8 canvas9 caption10 caption11 caption12 caption13 caption14 caption15 caption16 caption17 caption18 caption19 caption20 caption21 caption22 caption23 caption25 caption27 caption5 caption6 caption7 caption8 caption9 center10 center11 center12 center13 center14 center15 center16 center17 center18 center19 center20 center21 center22 center23 center24 center25 center26 center27 center28 center29 center3 center30 center31 center32 center33 center34 center35 center36 center37 center38 center39 center4 center40 center41 center42 center43 center44 center45 center46 center47 center48 center49 center5 center50 center51 center52 center53 center54 center55 center56 center57 center58 center59 center6 center60 center61 center62 center63 center64 center65 center66 center67 center68 center69 center7 center70 center71 center8 center9 cite10 cite11 cite12 cite13 cite14 cite15 cite16 cite17 cite18 cite24 cite4 cite6 cite7 cite8 cite9 code10 code11 code12 code13 code14 code15 code16 code17 code18 code21 code22 code3 code4 code5 code6 code7 code8 code9 col10 col11 col12 col13 col14 col15 col16 col17 col18 col19 col20 col21 col24 col28 col29 col36 col40 col5 col6 col7 col8 col9 colgroup10 colgroup11 colgroup12 colgroup13 colgroup14 colgroup15 colgroup16 colgroup17 colgroup18 colgroup19 colgroup20 colgroup23 colgroup27 colgroup28 colgroup35 colgroup39 colgroup4 colgroup5 colgroup6 colgroup7 colgroup8 colgroup9 d11 d14 d17 d6 datalist11 datalist14 datalist6 datalist7 datalist9 dd10 dd11 dd12 dd13 dd14 dd15 dd16 dd17 dd18 dd19 dd20 dd21 dd22 dd23 dd24 dd25 dd26 dd27 dd33 dd4 dd5 dd6 dd7 dd8 dd9 del10 del11 del12 del13 del14 del15 del16 del17 del18 del19 del20 del21 del22 del23 del25 del26 del28 del29 del33 del35 del5 del6 del7 del8 del9 details10 details11 details12 details13 details14 details16 details6 details7 details8 dfn10 dfn11 dfn12 dfn13 dfn14 dfn15 dfn16 dfn17 dfn18 dfn5 dfn7 dfn8 dfn9 dialog13 dialog3 dialog5 dialog6 div10 div100 div101 div102 div103 div104 div105 div106 div107 div11 div12 div13 div14 div15 div16 div17 div18 div19 div20 div21 div22 div23 div24 div25 div26 div27 div28 div29 div3 div30 div31 div32 div33 div34 div35 div36 div37 div38 div39 div4 div40 div41 div42 div43 div44 div45 div46 div47 div48 div49 div5 div50 div51 div52 div53 div54 div55 div56 div57 div58 div59 div6 div60 div61 div62 div63 div64 div65 div66 div67 div68 div69 div7 div70 div71 div72 div73 div74 div75 div76 div77 div78 div79 div8 div80 div81 div82 div83 div84 div85 div86 div87 div88 div89 div9 div90 div91 div92 div93 div94 div95 div96 div97 div98 div99 dl10 dl11 dl12 dl13 dl14 dl15 dl16 dl17 dl18 dl19 dl20 dl21 dl22 dl23 dl24 dl25 dl26 dl3 dl32 dl36 dl4 dl5 dl6 dl7 dl8 dl9 dt10 dt11 dt12 dt13 dt14 dt15 dt16 dt17 dt18 dt19 dt20 dt21 dt22 dt23 dt24 dt25 dt26 dt27 dt29 dt33 dt37 dt4 dt5 dt6 dt7 dt8 dt9 em10 em11 em12 em13 em14 em15 em16 em17 em18 em19 em20 em21 em22 em23 em24 em25 em26 em27 em28 em29 em3 em30 em31 em32 em33 em34 em35 em37 em4 em42 em45 em5 em6 em69 em7 em71 em77 em78 em79 em8 em84 em85 em9 embed10 embed11 embed12 embed13 embed14 embed15 embed16 embed17 embed18 embed19 embed20 embed21 embed22 embed24 embed25 embed27 embed3 embed31 embed32 embed35 embed4 embed5 embed6 embed7 embed8 embed9 fieldset10 fieldset11 fieldset12 fieldset13 fieldset14 fieldset15 fieldset16 fieldset17 fieldset18 fieldset19 fieldset20 fieldset21 fieldset22 fieldset23 fieldset24 fieldset25 fieldset26 fieldset27 fieldset28 fieldset29 fieldset3 fieldset30 fieldset33 fieldset4 fieldset5 fieldset6 fieldset7 fieldset8 fieldset9 figcaption10 figcaption11 figcaption12 figcaption13 figcaption14 figcaption15 figcaption16 figcaption17 figcaption18 figcaption19 figcaption20 figcaption21 figcaption22 figcaption23 figcaption24 figcaption25 figcaption27 figcaption35 figcaption53 figcaption6 figcaption7 figcaption8 figcaption9 figure10 figure11 figure12 figure13 figure14 figure15 figure16 figure17 figure18 figure19 figure20 figure21 figure22 figure23 figure24 figure25 figure26 figure27 figure3 figure30 figure32 figure33 figure36 figure4 figure5 figure52 figure6 figure7 figure8 figure9 filter10 filter12 filter13 filter15 filter16 filter17 filter4 filter7 filter8 filter9 firure10 firure11 firure13 firure14 firure20 firure9 font10 font104 font109 font11 font12 font13 font14 font15 font16 font17 font18 font19 font20 font21 font22 font23 font24 font25 font26 font27 font28 font29 font3 font30 font31 font32 font33 font34 font35 font36 font37 font38 font39 font4 font40 font41 font42 font43 font44 font45 font46 font47 font48 font49 font5 font50 font51 font53 font54 font55 font56 font57 font58 font59 font6 font60 font61 font62 font63 font64 font65 font69 font7 font74 font79 font8 font84 font89 font9 font94 font99 foooter4 footer10 footer11 footer12 footer13 footer14 footer15 footer16 footer17 footer18 footer19 footer20 footer22 footer24 footer27 footer3 footer30 footer4 footer43 footer5 footer6 footer7 footer8 footer9 form10 form11 form12 form13 form14 form15 form16 form17 form18 form19 form20 form21 form22 form23 form24 form25 form26 form27 form28 form29 form3 form30 form31 form32 form34 form35 form36 form37 form4 form40 form44 form48 form49 form5 form50 form57 form6 form60 form62 form68 form7 form8 form9 h03 h1020 h110 h111 h112 h113 h114 h115 h116 h117 h118 h119 h120 h121 h122 h123 h124 h125 h126 h127 h128 h129 h13 h130 h131 h132 h133 h134 h135 h137 h138 h139 h14 h140 h141 h1413 h15 h16 h1613 h17 h18 h19 h210 h211 h212 h213 h214 h215 h216 h217 h218 h219 h220 h221 h222 h223 h224 h225 h226 h227 h228 h229 h23 h230 h231 h232 h233 h234 h236 h237 h24 h241 h243 h244 h249 h25 h252 h253 h26 h27 h28 h29 h310 h311 h312 h313 h314 h315 h316 h317 h318 h319 h320 h321 h322 h323 h324 h325 h326 h327 h328 h329 h33 h330 h331 h3310 h332 h333 h334 h335 h337 h34 h341 h3410 h342 h35 h355 h359 h36 h37 h38 h39 h410 h411 h412 h413 h414 h415 h416 h417 h418 h419 h420 h421 h422 h423 h424 h425 h426 h427 h428 h429 h43 h430 h431 h432 h433 h434 h436 h44 h441 h442 h443 h444 h445 h45 h452 h46 h47 h48 h49 h510 h511 h512 h513 h514 h515 h516 h517 h518 h519 h520 h521 h522 h523 h524 h525 h526 h527 h528 h53 h530 h531 h536 h54 h55 h56 h57 h58 h59 h610 h611 h612 h613 h614 h615 h616 h617 h618 h619 h620 h621 h622 h623 h624 h625 h626 h627 h628 h629 h630 h64 h65 h66 h67 h68 h69 h710 h711 h712 h713 h715 h77 h79 h8 h810 h815 h817 h85 h87 h921 header10 header11 header12 header13 header14 header15 header16 header17 header18 header19 header20 header21 header22 header23 header24 header26 header27 header29 header3 header30 header4 header5 header6 header7 header8 header9 hr10 hr11 hr12 hr13 hr14 hr15 hr16 hr17 hr18 hr19 hr20 hr21 hr22 hr23 hr24 hr25 hr26 hr27 hr28 hr29 hr3 hr30 hr31 hr34 hr35 hr37 hr38 hr4 hr40 hr41 hr43 hr45 hr5 hr54 hr55 hr56 hr6 hr7 hr8 hr9 href20 href6 href7 href8 href9 i10 i11 i12 i13 i14 i15 i16 i17 i18 i19 i20 i21 i22 i23 i24 i25 i26 i27 i28 i29 i3 i30 i31 i32 i33 i34 i35 i36 i37 i38 i39 i4 i40 i42 i45 i5 i6 i7 i8 i9 iframe10 iframe11 iframe12 iframe13 iframe14 iframe15 iframe16 iframe17 iframe18 iframe19 iframe20 iframe21 iframe22 iframe23 iframe24 iframe25 iframe26 iframe27 iframe28 iframe29 iframe3 iframe30 iframe31 iframe32 iframe33 iframe34 iframe35 iframe36 iframe37 iframe39 iframe4 iframe40 iframe43 iframe44 iframe5 iframe50 iframe6 iframe7 iframe8 iframe9 image10 image11 image12 image13 image14 image15 image16 image17 image18 image19 image20 image21 image22 image24 image25 image28 image4 image5 image6 image7 image8 image9 img10 img108 img11 img12 img13 img14 img15 img16 img17 img18 img19 img20 img21 img22 img23 img24 img25 img26 img27 img28 img29 img3 img30 img31 img32 img33 img34 img35 img36 img37 img38 img39 img4 img40 img41 img42 img43 img44 img45 img46 img47 img48 img49 img5 img50 img51 img52 img53 img54 img55 img56 img57 img58 img59 img6 img60 img61 img63 img64 img65 img66 img67 img7 img74 img76 img8 img83 img9 input10 input11 input12 input13 input14 input15 input16 input17 input18 input19 input20 input21 input22 input23 input24 input25 input26 input27 input28 input29 input3 input30 input31 input32 input33 input34 input35 input36 input37 input38 input39 input4 input40 input41 input42 input43 input44 input45 input46 input48 input49 input5 input51 input53 input54 input55 input57 input58 input6 input60 input61 input63 input69 input7 input71 input75 input8 input85 input87 input9 input93 ins10 ins11 ins12 ins13 ins14 ins15 ins16 ins17 ins18 ins19 ins20 ins21 ins22 ins23 ins24 ins25 ins26 ins27 ins28 ins29 ins3 ins30 ins31 ins32 ins33 ins34 ins35 ins36 ins4 ins5 ins6 ins7 ins8 ins9 item10 item11 item12 item13 item14 item15 item16 item5 item8 kbd10 kbd11 kbd14 kbd16 kbd7 kbd9 label10 label11 label12 label13 label14 label15 label16 label17 label18 label19 label20 label21 label22 label23 label24 label25 label26 label27 label28 label29 label3 label30 label31 label32 label33 label34 label35 label36 label37 label38 label39 label4 label40 label41 label42 label43 label46 label5 label6 label7 label8 label9 lable8 legend10 legend11 legend12 legend13 legend14 legend15 legend16 legend17 legend18 legend19 legend20 legend21 legend22 legend23 legend24 legend25 legend26 legend27 legend28 legend29 legend30 legend31 legend5 legend6 legend7 legend8 legend9 li10 li11 li12 li13 li14 li15 li16 li17 li18 li19 li20 li21 li210 li22 li23 li24 li25 li26 li27 li28 li29 li3 li30 li31 li32 li33 li34 li35 li36 li37 li39 li4 li40 li41 li42 li43 li44 li45 li46 li48 li49 li5 li50 li55 li57 li6 li60 li61 li62 li7 li8 li9 link10 link11 link12 link13 link14 link15 link16 link17 link18 link19 link20 link21 link22 link23 link24 link25 link27 link28 link3 link31 link34 link36 link39 link4 link43 link5 link6 link7 link8 link9 map10 map11 map12 map13 map14 map15 map16 map17 map18 map19 map20 map21 map22 map23 map24 map25 map28 map29 map3 map30 map31 map4 map43 map5 map6 map7 map8 map9 mark10 mark11 mark12 mark13 mark16 mark17 mark18 mark21 mark23 mark6 mark7 mark8 mark9 menu10 menu11 menu12 menu13 menu14 menu15 menu16 menu17 menu18 menu19 menu20 menu21 menu3 menu37 menu4 menu5 menu6 menu7 menu8 menu9 menuitem11 menuitem12 menuitem13 menuitem14 menuitem15 menuitem16 menuitem20 menuitem21 menuitem38 menuitem6 menuitem7 menuitem8 menuitem9 meta10 meta11 meta12 meta13 meta14 meta15 meta16 meta17 meta18 meta19 meta20 meta21 meta22 meta23 meta24 meta25 meta26 meta27 meta28 meta29 meta3 meta31 meta37 meta38 meta4 meta5 meta6 meta7 meta8 meta9 nav10 nav11 nav12 nav13 nav14 nav15 nav16 nav17 nav18 nav19 nav20 nav21 nav22 nav23 nav24 nav25 nav26 nav27 nav3 nav4 nav5 nav6 nav7 nav8 nav9 noscript10 noscript11 noscript12 noscript13 noscript14 noscript15 noscript16 noscript17 noscript18 noscript19 noscript20 noscript21 noscript22 noscript23 noscript24 noscript25 noscript26 noscript27 noscript28 noscript29 noscript3 noscript30 noscript31 noscript32 noscript36 noscript4 noscript43 noscript5 noscript6 noscript7 noscript8 noscript9 object10 object11 object12 object13 object14 object15 object16 object17 object18 object19 object20 object21 object22 object23 object24 object25 object26 object29 object3 object30 object31 object34 object36 object4 object5 object6 object7 object8 object9 ol10 ol11 ol12 ol13 ol14 ol15 ol16 ol17 ol18 ol19 ol20 ol21 ol22 ol23 ol24 ol25 ol26 ol27 ol28 ol3 ol30 ol31 ol32 ol34 ol35 ol38 ol4 ol40 ol41 ol5 ol6 ol7 ol8 ol9 optgroup10 optgroup11 optgroup12 optgroup13 optgroup14 optgroup15 optgroup16 optgroup17 optgroup18 optgroup19 optgroup20 optgroup21 optgroup22 optgroup24 optgroup25 optgroup26 optgroup30 optgroup5 optgroup6 optgroup7 optgroup8 optgroup9 option10 option11 option12 option13 option14 option15 option16 option17 option18 option19 option20 option21 option22 option23 option24 option25 option26 option27 option28 option29 option30 option31 option32 option33 option34 option35 option36 option37 option38 option4 option40 option42 option43 option45 option46 option5 option6 option7 option8 option9 p10 p11 p117 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p216 p22 p23 p24 p25 p26 p27 p28 p29 p3 p30 p31 p314 p32 p33 p34 p35 p36 p37 p38 p39 p4 p40 p41 p42 p43 p44 p45 p46 p47 p48 p5 p50 p52 p54 p55 p56 p58 p59 p6 p60 p61 p7 p8 p9 param10 param11 param12 param13 param14 param15 param16 param17 param18 param19 param20 param21 param22 param23 param24 param25 param26 param27 param3 param30 param31 param32 param35 param37 param4 param5 param6 param7 param8 param9 pre10 pre11 pre12 pre13 pre14 pre15 pre16 pre17 pre19 pre20 pre22 pre3 pre30 pre4 pre5 pre6 pre7 pre8 pre9 progress12 progress5 progress9 q10 q11 q12 q13 q14 q15 q16 q18 q19 q7 q8 q9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 s31 s32 s4 s40 s5 s7 s8 s9 samp10 samp11 samp12 samp13 samp14 samp16 samp5 samp6 samp8 samp9 script10 script11 script116 script12 script13 script14 script15 script16 script17 script18 script19 script20 script21 script22 script23 script24 script25 script26 script27 script28 script29 script3 script30 script31 script32 script33 script34 script35 script36 script37 script38 script39 script4 script40 script41 script43 script47 script48 script49 script5 script50 script52 script53 script57 script6 script62 script7 script8 script9 section10 section11 section12 section13 section14 section15 section16 section17 section18 section19 section20 section21 section22 section23 section24 section25 section26 section27 section29 section3 section30 section33 section4 section43 section5 section6 section7 section8 section9 select10 select11 select12 select13 select14 select15 select16 select17 select18 select19 select20 select21 select22 select23 select24 select25 select26 select27 select28 select29 select3 select30 select31 select32 select33 select34 select35 select36 select37 select39 select4 select41 select42 select44 select45 select5 select6 select7 select8 select9 small10 small11 small12 small13 small14 small15 small16 small17 small18 small19 small20 small21 small22 small23 small24 small25 small26 small27 small3 small30 small31 small32 small33 small4 small5 small6 small7 small8 small9 source10 source11 source12 source13 source14 source15 source16 source17 source18 source19 source20 source21 source22 source23 source24 source25 source26 source27 source28 source29 source30 source31 source32 source33 source34 source35 source4 source42 source5 source6 source7 source8 source9 span10 span100 span101 span102 span103 span104 span105 span106 span107 span108 span109 span11 span111 span112 span113 span12 span13 span14 span15 span16 span17 span18 span19 span20 span21 span22 span23 span24 span25 span26 span27 span28 span29 span3 span30 span31 span32 span33 span34 span35 span36 span37 span38 span39 span4 span40 span41 span42 span43 span44 span45 span46 span47 span48 span49 span5 span50 span51 span52 span53 span54 span55 span56 span57 span58 span59 span6 span60 span61 span62 span63 span64 span65 span66 span67 span68 span69 span7 span70 span71 span72 span73 span74 span75 span76 span77 span78 span79 span8 span80 span81 span82 span83 span84 span85 span86 span87 span88 span89 span9 span90 span91 span92 span93 span94 span95 span96 span97 span98 span99 strong10 strong11 strong12 strong13 strong14 strong15 strong16 strong17 strong18 strong19 strong20 strong21 strong215 strong216 strong22 strong23 strong24 strong25 strong26 strong27 strong28 strong29 strong3 strong30 strong31 strong32 strong33 strong34 strong35 strong36 strong37 strong38 strong39 strong4 strong40 strong41 strong42 strong43 strong44 strong45 strong46 strong47 strong49 strong5 strong52 strong53 strong54 strong55 strong56 strong57 strong58 strong59 strong6 strong61 strong62 strong66 strong7 strong70 strong73 strong75 strong76 strong79 strong8 strong9 strong93 style10 style11 style12 style13 style14 style15 style16 style17 style18 style19 style20 style21 style22 style23 style24 style25 style26 style27 style28 style29 style3 style30 style31 style34 style35 style36 style38 style39 style4 style46 style5 style6 style7 style8 style9 sub10 sub11 sub12 sub13 sub14 sub15 sub16 sub19 sub5 sub6 sub7 sub8 sub9 summary10 summary11 summary15 summary7 summary8 summary9 sup10 sup11 sup12 sup13 sup14 sup15 sup16 sup17 sup18 sup19 sup20 sup21 sup22 sup23 sup24 sup25 sup26 sup27 sup29 sup33 sup35 sup36 sup37 sup38 sup4 sup5 sup6 sup7 sup8 sup9 table10 table11 table12 table13 table14 table15 table16 table17 table18 table19 table20 table21 table22 table23 table24 table25 table26 table27 table28 table29 table3 table30 table31 table32 table33 table34 table35 table36 table37 table38 table39 table4 table40 table41 table42 table43 table44 table45 table46 table47 table48 table49 table5 table50 table51 table52 table53 table54 table55 table56 table57 table58 table6 table60 table61 table62 table63 table64 table65 table66 table67 table68 table69 table7 table70 table71 table72 table73 table74 table75 table76 table77 table79 table8 table80 table81 table83 table85 table89 table9 tbody10 tbody11 tbody12 tbody13 tbody14 tbody15 tbody16 tbody17 tbody18 tbody19 tbody20 tbody21 tbody22 tbody23 tbody24 tbody25 tbody26 tbody27 tbody28 tbody29 tbody30 tbody31 tbody32 tbody33 tbody34 tbody35 tbody36 tbody37 tbody38 tbody39 tbody4 tbody40 tbody41 tbody42 tbody43 tbody44 tbody45 tbody46 tbody47 tbody48 tbody49 tbody5 tbody50 tbody51 tbody52 tbody53 tbody54 tbody55 tbody56 tbody57 tbody58 tbody59 tbody6 tbody61 tbody62 tbody63 tbody64 tbody65 tbody66 tbody67 tbody68 tbody69 tbody7 tbody70 tbody71 tbody72 tbody73 tbody74 tbody75 tbody76 tbody77 tbody78 tbody8 tbody80 tbody81 tbody82 tbody84 tbody86 tbody9 tbody90 td10 td11 td12 td13 td14 td15 td16 td17 td18 td19 td20 td21 td22 td23 td24 td25 td26 td27 td28 td29 td30 td31 td32 td33 td34 td35 td36 td37 td38 td39 td4 td40 td41 td42 td43 td44 td45 td46 td47 td48 td49 td50 td51 td52 td53 td54 td55 td56 td57 td58 td59 td6 td60 td61 td63 td64 td65 td66 td67 td68 td69 td7 td70 td71 td72 td73 td74 td75 td76 td77 td78 td79 td8 td80 td82 td83 td84 td86 td88 td9 td92 textarea10 textarea11 textarea12 textarea13 textarea14 textarea15 textarea16 textarea17 textarea18 textarea19 textarea20 textarea21 textarea22 textarea23 textarea24 textarea25 textarea26 textarea27 textarea28 textarea29 textarea3 textarea32 textarea39 textarea4 textarea5 textarea53 textarea6 textarea7 textarea71 textarea8 textarea9 tfoot10 tfoot11 tfoot12 tfoot13 tfoot14 tfoot15 tfoot16 tfoot18 tfoot21 tfoot5 tfoot6 tfoot7 tfoot8 tfoot9 th10 th11 th12 th13 th14 th15 th16 th17 th18 th19 th20 th21 th22 th23 th24 th25 th26 th27 th28 th29 th30 th31 th32 th34 th36 th38 th6 th7 th8 th9 thead10 thead11 thead12 thead13 thead14 thead15 thead16 thead17 thead18 thead19 thead20 thead21 thead22 thead23 thead24 thead25 thead26 thead27 thead28 thead29 thead32 thead34 thead4 thead5 thead6 thead7 thead8 thead9 title10 title11 title12 title13 title14 title15 title16 title17 title18 title19 title20 title21 title22 title23 title24 title26 title28 title29 title3 title30 title31 title37 title38 title4 title40 title42 title5 title6 title7 title8 title9 tr10 tr11 tr12 tr13 tr14 tr15 tr16 tr17 tr18 tr19 tr20 tr21 tr22 tr23 tr24 tr25 tr26 tr27 tr28 tr29 tr30 tr31 tr32 tr33 tr34 tr35 tr36 tr37 tr38 tr39 tr40 tr41 tr42 tr43 tr44 tr45 tr46 tr47 tr48 tr49 tr5 tr50 tr51 tr52 tr53 tr54 tr55 tr56 tr57 tr58 tr59 tr6 tr60 tr62 tr63 tr64 tr65 tr66 tr67 tr68 tr69 tr7 tr70 tr71 tr72 tr73 tr74 tr75 tr76 tr77 tr78 tr79 tr8 tr81 tr82 tr83 tr85 tr87 tr9 tr91 tt11 tt7 tt8 u10 u11 u12 u13 u14 u15 u16 u17 u18 u19 u20 u21 u22 u23 u24 u25 u26 u27 u28 u29 u3 u30 u31 u32 u33 u34 u35 u36 u37 u38 u39 u4 u42 u43 u44 u48 u5 u53 u6 u7 u8 u9 ul10 ul11 ul12 ul13 ul14 ul15 ul16 ul17 ul18 ul19 ul20 ul21 ul22 ul23 ul24 ul25 ul26 ul27 ul28 ul29 ul3 ul30 ul31 ul32 ul33 ul34 ul35 ul36 ul38 ul39 ul4 ul40 ul41 ul42 ul43 ul44 ul45 ul47 ul48 ul49 ul5 ul53 ul54 ul56 ul59 ul6 ul60 ul61 ul7 ul8 ul9 var10 var11 var12 var13 var14 var15 var16 var3 var4 var5 var6 var7 var8 var9 video10 video11 video12 video13 video14 video15 video16 video17 video18 video19 video20 video21 video23 video24 video25 video3 video4 video5 video6 video7 video8 video9 wbr10 wbr11 wbr12 wbr13 wbr14 wbr15 wbr16 wbr17 wbr18 wbr19 wbr20 wbr21 wbr22 wbr23 wbr24 wbr25 wbr26 wbr27 wbr28 wbr29 wbr34 wbr36 wbr37 wbr7 wbr8 wbr9"
        veclistDict = strk.split(' ')#词典列表
        
        vectorizer=CountVectorizer(vocabulary=veclistDict)#
        transformer=TfidfTransformer()# 
        X1 = vectorizer.fit_transform(res) #词频矩阵
        X2 = X1.toarray()#词频矩阵
        tfidf=transformer.fit_transform(X1)
        weight=tfidf.toarray()# 
        ########
        #tagWeight=[]
        tagWeightIn=[]
         
        for i in range(len(veclistDict)):  #词典长度
                
                if veclistDict[i] in dictA:
                    tagWeightIn.append(0.7/0.3)
                    break
                if veclistDict[i] in dictB:
                    tagWeightIn.append(0.3/0.3)
                    break
                    
        listNew=np.array(tagWeightIn)    
        newWeight=weight*listNew
        ########
        
        fink = self.pca(newWeight,0.9508) #0.9674 0.9871 #0.96 #0.9603 #0.9697 #0.9817 #0.9508
        vec = fink.tolist()
        for i in range(len(vec)):
            vec[i].append(classRes[i]) 
            vec[i].append(lz78Res[i])
            
          
        return vec   #返回词频矩阵
    def fetchVectorListN(self, fnin, ffin = None):
        pageInfo = None
        with open(fnin, 'r') as fin:
            pageInfo = eval(fin.read())
        
        pageFilter = None
        if ffin:
            with open(ffin, 'r') as fin:
                pageFilter = eval(fin.read())    
        
        res = []
        classRes=[]
        lz78Res=[]  
        
        tagSeqList =[]
        
        for file, content in pageInfo.items():
            
            #ignore the files not in filter
            if pageFilter:
                if not file in pageFilter:
                    continue
            #######################
            
            vector = content['newtagseq']  #字典中的内容
            TestClass = content['classStyle'] #一个列表
            Testlz78 = content['tagseq'] #为了计算压缩
            
            veclist = vector.split(',')
            ls3 = ' '.join(veclist)
            res.append(ls3)  #tfidf的所有标签序列
            
            tagSeqList.append(veclist) #标签列表
            
            classRes.append(TestClass)  #得到所有网页的class集合
            lz78Res.append(Testlz78)
            
        dictA=['form10', 'form11', 'form12', 'form13', 'form14', 'form15', 'form16', 'form17', 'form18', 'form19', 'form20', 'form21', 'form22', 'form23', 'form24', 'form25', 'form26', 'form27', 'form28', 'form29', 'form3', 'form30', 'form31', 'form32', 'form34', 'form35', 'form36', 'form37', 'form4', 'form40', 'form44', 'form48', 'form49', 'form5', 'form50', 'form57', 'form6', 'form60', 'form62', 'form68', 'form7', 'form8', 'form9', 'div10', 'div100', 'div101', 'div102', 'div103', 'div104', 'div105', 'div106', 'div107', 'div11', 'div12', 'div13', 'div14', 'div15', 'div16', 'div17', 'div18', 'div19', 'div20', 'div21', 'div22', 'div23', 'div24', 'div25', 'div26', 'div27', 'div28', 'div29', 'div3', 'div30', 'div31', 'div32', 'div33', 'div34', 'div35', 'div36', 'div37', 'div38', 'div39', 'div4', 'div40', 'div41', 'div42', 'div43', 'div44', 'div45', 'div46', 'div47', 'div48', 'div49', 'div5', 'div50', 'div51', 'div52', 'div53', 'div54', 'div55', 'div56', 'div57', 'div58', 'div59', 'div6', 'div60', 'div61', 'div62', 'div63', 'div64', 'div65', 'div66', 'div67', 'div68', 'div69', 'div7', 'div70', 'div71', 'div72', 'div73', 'div74', 'div75', 'div76', 'div77', 'div78', 'div79', 'div8', 'div80', 'div81', 'div82', 'div83', 'div84', 'div85', 'div86', 'div87', 'div88', 'div89', 'div9', 'div90', 'div91', 'div92', 'div93', 'div94', 'div95', 'div96', 'div97', 'div98', 'div99', 'style10', 'style11', 'style12', 'style13', 'style14', 'style15', 'style16', 'style17', 'style18', 'style19', 'style20', 'style21', 'style22', 'style23', 'style24', 'style25', 'style26', 'style27', 'style28', 'style29', 'style3', 'style30', 'style31', 'style34', 'style35', 'style36', 'style38', 'style39', 'style4', 'style46', 'style5', 'style6', 'style7', 'style8', 'style9', 'span10', 'span100', 'span101', 'span102', 'span103', 'span104', 'span105', 'span106', 'span107', 'span108', 'span109', 'span11', 'span111', 'span112', 'span113', 'span12', 'span13', 'span14', 'span15', 'span16', 'span17', 'span18', 'span19', 'span20', 'span21', 'span22', 'span23', 'span24', 'span25', 'span26', 'span27', 'span28', 'span29', 'span3', 'span30', 'span31', 'span32', 'span33', 'span34', 'span35', 'span36', 'span37', 'span38', 'span39', 'span4', 'span40', 'span41', 'span42', 'span43', 'span44', 'span45', 'span46', 'span47', 'span48', 'span49', 'span5', 'span50', 'span51', 'span52', 'span53', 'span54', 'span55', 'span56', 'span57', 'span58', 'span59', 'span6', 'span60', 'span61', 'span62', 'span63', 'span64', 'span65', 'span66', 'span67', 'span68', 'span69', 'span7', 'span70', 'span71', 'span72', 'span73', 'span74', 'span75', 'span76', 'span77', 'span78', 'span79', 'span8', 'span80', 'span81', 'span82', 'span83', 'span84', 'span85', 'span86', 'span87', 'span88', 'span89', 'span9', 'span90', 'span91', 'span92', 'span93', 'span94', 'span95', 'span96', 'span97', 'span98', 'span99',  'td10', 'td11', 'td12', 'td13', 'td14', 'td15', 'td16', 'td17', 'td18', 'td19', 'td20', 'td21', 'td22', 'td23', 'td24', 'td25', 'td26', 'td27', 'td28', 'td29', 'td30', 'td31', 'td32', 'td33', 'td34', 'td35', 'td36', 'td37', 'td38', 'td39', 'td4', 'td40', 'td41', 'td42', 'td43', 'td44', 'td45', 'td46', 'td47', 'td48', 'td49', 'td50', 'td51', 'td52', 'td53', 'td54', 'td55', 'td56', 'td57', 'td58', 'td59', 'td6', 'td60', 'td61', 'td63', 'td64', 'td65', 'td66', 'td67', 'td68', 'td69', 'td7', 'td70', 'td71', 'td72', 'td73', 'td74', 'td75', 'td76', 'td77', 'td78', 'td79', 'td8', 'td80', 'td82', 'td83', 'td84', 'td86', 'td88', 'td9', 'td92','iframe10', 'iframe11', 'iframe12', 'iframe13', 'iframe14', 'iframe15', 'iframe16', 'iframe17', 'iframe18', 'iframe19', 'iframe20', 'iframe21', 'iframe22', 'iframe23', 'iframe24', 'iframe25', 'iframe26', 'iframe27', 'iframe28', 'iframe29', 'iframe3', 'iframe30', 'iframe31', 'iframe32', 'iframe33', 'iframe34', 'iframe35', 'iframe36', 'iframe37', 'iframe39', 'iframe4', 'iframe40', 'iframe43', 'iframe44', 'iframe5', 'iframe50', 'iframe6', 'iframe7', 'iframe8', 'iframe9', 'fieldset10', 'fieldset11', 'fieldset12', 'fieldset13', 'fieldset14', 'fieldset15', 'fieldset16', 'fieldset17', 'fieldset18', 'fieldset19', 'fieldset20', 'fieldset21', 'fieldset22', 'fieldset23', 'fieldset24', 'fieldset25', 'fieldset26', 'fieldset27', 'fieldset28', 'fieldset29', 'fieldset3', 'fieldset30', 'fieldset33', 'fieldset4', 'fieldset5', 'fieldset6', 'fieldset7', 'fieldset8', 'fieldset9', 'select10', 'select11', 'select12', 'select13', 'select14', 'select15', 'select16', 'select17', 'select18', 'select19', 'select20', 'select21', 'select22', 'select23', 'select24', 'select25', 'select26', 'select27', 'select28', 'select29', 'select3', 'select30', 'select31', 'select32', 'select33', 'select34', 'select35', 'select36', 'select37', 'select39', 'select4', 'select41', 'select42', 'select44', 'select45', 'select5', 'select6', 'select7', 'select8', 'select9', 'option10', 'option11', 'option12', 'option13', 'option14', 'option15', 'option16', 'option17', 'option18', 'option19', 'option20', 'option21', 'option22', 'option23', 'option24', 'option25', 'option26', 'option27', 'option28', 'option29', 'option30', 'option31', 'option32', 'option33', 'option34', 'option35', 'option36', 'option37', 'option38', 'option4', 'option40', 'option42', 'option43', 'option45', 'option46', 'option5', 'option6', 'option7', 'option8', 'option9', 'li10', 'li11', 'li12', 'li13', 'li14', 'li15', 'li16', 'li17', 'li18', 'li19', 'li20', 'li21', 'li210', 'li22', 'li23', 'li24', 'li25', 'li26', 'li27', 'li28', 'li29', 'li3', 'li30', 'li31', 'li32', 'li33', 'li34', 'li35', 'li36', 'li37', 'li39', 'li4', 'li40', 'li41', 'li42', 'li43', 'li44', 'li45', 'li46', 'li48', 'li49', 'li5', 'li50', 'li55', 'li57', 'li6', 'li60', 'li61', 'li62', 'li7', 'li8', 'li9', 'input10', 'input11', 'input12', 'input13', 'input14', 'input15', 'input16', 'input17', 'input18', 'input19', 'input20', 'input21', 'input22', 'input23', 'input24', 'input25', 'input26', 'input27', 'input28', 'input29', 'input3', 'input30', 'input31', 'input32', 'input33', 'input34', 'input35', 'input36', 'input37', 'input38', 'input39', 'input4', 'input40', 'input41', 'input42', 'input43', 'input44', 'input45', 'input46', 'input48', 'input49', 'input5', 'input51', 'input53', 'input54', 'input55', 'input57', 'input58', 'input6', 'input60', 'input61', 'input63', 'input69', 'input7', 'input71', 'input75', 'input8', 'input85', 'input87', 'input9', 'input93', 'button10', 'button11', 'button12', 'button13', 'button14', 'button15', 'button16', 'button17', 'button18', 'button19', 'button20', 'button21', 'button22', 'button23', 'button24', 'button25', 'button26', 'button27', 'button28', 'button29', 'button3', 'button30', 'button31', 'button32', 'button33', 'button37', 'button38', 'button4', 'button41', 'button42', 'button43', 'button44', 'button45', 'button5', 'button53', 'button6', 'button7', 'button8', 'button9','area10', 'area11', 'area12', 'area13', 'area14', 'area15', 'area16', 'area17', 'area18', 'area19', 'area20', 'area21', 'area22', 'area23', 'area24', 'area25', 'area26', 'area29', 'area3', 'area30', 'area31', 'area32', 'area4', 'area44', 'area5', 'area6', 'area7', 'area8', 'area9', 'table10', 'table11', 'table12', 'table13', 'table14', 'table15', 'table16', 'table17', 'table18', 'table19', 'table20', 'table21', 'table22', 'table23', 'table24', 'table25', 'table26', 'table27', 'table28', 'table29', 'table3', 'table30', 'table31', 'table32', 'table33', 'table34', 'table35', 'table36', 'table37', 'table38', 'table39', 'table4', 'table40', 'table41', 'table42', 'table43', 'table44', 'table45', 'table46', 'table47', 'table48', 'table49', 'table5', 'table50', 'table51', 'table52', 'table53', 'table54', 'table55', 'table56', 'table57', 'table58', 'table6', 'table60', 'table61', 'table62', 'table63', 'table64', 'table65', 'table66', 'table67', 'table68', 'table69', 'table7', 'table70', 'table71', 'table72', 'table73', 'table74', 'table75', 'table76', 'table77', 'table79', 'table8', 'table80', 'table81', 'table83', 'table85', 'table89', 'table9', 'tr10', 'tr11', 'tr12', 'tr13', 'tr14', 'tr15', 'tr16', 'tr17', 'tr18', 'tr19', 'tr20', 'tr21', 'tr22', 'tr23', 'tr24', 'tr25', 'tr26', 'tr27', 'tr28', 'tr29', 'tr30', 'tr31', 'tr32', 'tr33', 'tr34', 'tr35', 'tr36', 'tr37', 'tr38', 'tr39', 'tr40', 'tr41', 'tr42', 'tr43', 'tr44', 'tr45', 'tr46', 'tr47', 'tr48', 'tr49', 'tr5', 'tr50', 'tr51', 'tr52', 'tr53', 'tr54', 'tr55', 'tr56', 'tr57', 'tr58', 'tr59', 'tr6', 'tr60', 'tr62', 'tr63', 'tr64', 'tr65', 'tr66', 'tr67', 'tr68', 'tr69', 'tr7', 'tr70', 'tr71', 'tr72', 'tr73', 'tr74', 'tr75', 'tr76', 'tr77', 'tr78', 'tr79', 'tr8', 'tr81', 'tr82', 'tr83', 'tr85', 'tr87', 'tr9', 'tr91', 'br10', 'br109', 'br11', 'br12', 'br13', 'br14', 'br15', 'br16', 'br17', 'br18', 'br19', 'br20', 'br21', 'br22', 'br23', 'br24', 'br25', 'br26', 'br27', 'br28', 'br29', 'br3', 'br30', 'br31', 'br32', 'br33', 'br34', 'br35', 'br36', 'br37', 'br38', 'br39', 'br4', 'br40', 'br41', 'br42', 'br43', 'br44', 'br45', 'br46', 'br47', 'br48', 'br49', 'br5', 'br50', 'br51', 'br52', 'br54', 'br55', 'br56', 'br57', 'br59', 'br6', 'br60', 'br62', 'br63', 'br7', 'br78', 'br8', 'br84', 'br9',  'hr10', 'hr11', 'hr12', 'hr13', 'hr14', 'hr15', 'hr16', 'hr17', 'hr18', 'hr19', 'hr20', 'hr21', 'hr22', 'hr23', 'hr24', 'hr25', 'hr26', 'hr27', 'hr28', 'hr29', 'hr3', 'hr30', 'hr31', 'hr34', 'hr35', 'hr37', 'hr38', 'hr4', 'hr40', 'hr41', 'hr43', 'hr45', 'hr5', 'hr54', 'hr55', 'hr56', 'hr6', 'hr7', 'hr8', 'hr9',  'h1020', 'h110', 'h111', 'h112', 'h113', 'h114', 'h115', 'h116', 'h117', 'h118', 'h119', 'h120', 'h121', 'h122', 'h123', 'h124', 'h125', 'h126', 'h127', 'h128', 'h129', 'h13', 'h130', 'h131', 'h132', 'h133', 'h134', 'h135', 'h137', 'h138', 'h139', 'h14', 'h140', 'h141', 'h1413', 'h15', 'h16', 'h1613', 'h17', 'h18', 'h19', 'th10', 'th11', 'th12', 'th13', 'th14', 'th15', 'th16', 'th17', 'th18', 'th19', 'th20', 'th21', 'th22', 'th23', 'th24', 'th25', 'th26', 'th27', 'th28', 'th29', 'th30', 'th31', 'th32', 'th34', 'th36', 'th38', 'th6', 'th7', 'th8', 'th9', 'canvas10', 'canvas11', 'canvas12', 'canvas13', 'canvas14', 'canvas15', 'canvas16', 'canvas17', 'canvas18', 'canvas19', 'canvas20', 'canvas21', 'canvas22', 'canvas23', 'canvas24', 'canvas25', 'canvas27', 'canvas3', 'canvas31', 'canvas35', 'canvas36', 'canvas4', 'canvas5', 'canvas6', 'canvas7', 'canvas8', 'canvas9','h610', 'h611', 'h612', 'h613', 'h614', 'h615', 'h616', 'h617', 'h618', 'h619', 'h620', 'h621', 'h622', 'h623', 'h624', 'h625', 'h626', 'h627', 'h628', 'h629', 'h630', 'h64', 'h65', 'h66', 'h67', 'h68', 'h69', 'tbody10', 'tbody11', 'tbody12', 'tbody13', 'tbody14', 'tbody15', 'tbody16', 'tbody17', 'tbody18', 'tbody19', 'tbody20', 'tbody21', 'tbody22', 'tbody23', 'tbody24', 'tbody25', 'tbody26', 'tbody27', 'tbody28', 'tbody29', 'tbody30', 'tbody31', 'tbody32', 'tbody33', 'tbody34', 'tbody35', 'tbody36', 'tbody37', 'tbody38', 'tbody39', 'tbody4', 'tbody40', 'tbody41', 'tbody42', 'tbody43', 'tbody44', 'tbody45', 'tbody46', 'tbody47', 'tbody48', 'tbody49', 'tbody5', 'tbody50', 'tbody51', 'tbody52', 'tbody53', 'tbody54', 'tbody55', 'tbody56', 'tbody57', 'tbody58', 'tbody59', 'tbody6', 'tbody61', 'tbody62', 'tbody63', 'tbody64', 'tbody65', 'tbody66', 'tbody67', 'tbody68', 'tbody69', 'tbody7', 'tbody70', 'tbody71', 'tbody72', 'tbody73', 'tbody74', 'tbody75', 'tbody76', 'tbody77', 'tbody78', 'tbody8', 'tbody80', 'tbody81', 'tbody82', 'tbody84', 'tbody86', 'tbody9', 'tbody90', 'ul10', 'ul11', 'ul12', 'ul13', 'ul14', 'ul15', 'ul16', 'ul17', 'ul18', 'ul19', 'ul20', 'ul21', 'ul22', 'ul23', 'ul24', 'ul25', 'ul26', 'ul27', 'ul28', 'ul29', 'ul3', 'ul30', 'ul31', 'ul32', 'ul33', 'ul34', 'ul35', 'ul36', 'ul38', 'ul39', 'ul4', 'ul40', 'ul41', 'ul42', 'ul43', 'ul44', 'ul45', 'ul47', 'ul48', 'ul49', 'ul5', 'ul53', 'ul54', 'ul56', 'ul59', 'ul6', 'ul60', 'ul61', 'ul7', 'ul8', 'ul9',  'nav10', 'nav11', 'nav12', 'nav13', 'nav14', 'nav15', 'nav16', 'nav17', 'nav18', 'nav19', 'nav20', 'nav21', 'nav22', 'nav23', 'nav24', 'nav25', 'nav26', 'nav27', 'nav3', 'nav4', 'nav5', 'nav6', 'nav7', 'nav8', 'nav9',  'header10', 'header11', 'header12', 'header13', 'header14', 'header15', 'header16', 'header17', 'header18', 'header19', 'header20', 'header21', 'header22', 'header23', 'header24', 'header26', 'header27', 'header29', 'header3', 'header30', 'header4', 'header5', 'header6', 'header7', 'header8', 'header9','foooter4', 'footer10', 'footer11', 'footer12', 'footer13', 'footer14', 'footer15', 'footer16', 'footer17', 'footer18', 'footer19', 'footer20', 'footer22', 'footer24', 'footer27', 'footer3', 'footer30', 'footer4', 'footer43', 'footer5', 'footer6', 'footer7', 'footer8', 'footer9',  'label10', 'label11', 'label12', 'label13', 'label14', 'label15', 'label16', 'label17', 'label18', 'label19', 'label20', 'label21', 'label22', 'label23', 'label24', 'label25', 'label26', 'label27', 'label28', 'label29', 'label3', 'label30', 'label31', 'label32', 'label33', 'label34', 'label35', 'label36', 'label37', 'label38', 'label39', 'label4', 'label40', 'label41', 'label42', 'label43', 'label46', 'label5', 'label6', 'label7', 'label8', 'label9', 'lable8', 'font10', 'font104', 'font109', 'font11', 'font12', 'font13', 'font14', 'font15', 'font16', 'font17', 'font18', 'font19', 'font20', 'font21', 'font22', 'font23', 'font24', 'font25', 'font26', 'font27', 'font28', 'font29', 'font3', 'font30', 'font31', 'font32', 'font33', 'font34', 'font35', 'font36', 'font37', 'font38', 'font39', 'font4', 'font40', 'font41', 'font42', 'font43', 'font44', 'font45', 'font46', 'font47', 'font48', 'font49', 'font5', 'font50', 'font51', 'font53', 'font54', 'font55', 'font56', 'font57', 'font58', 'font59', 'font6', 'font60', 'font61', 'font62', 'font63', 'font64', 'font65', 'font69', 'font7', 'font74', 'font79', 'font8', 'font84', 'font89', 'font9', 'font94', 'font99', 'article10', 'article11', 'article12', 'article13', 'article14', 'article15', 'article16', 'article17', 'article18', 'article19', 'article20', 'article21', 'article22', 'article23', 'article24', 'article25', 'article26', 'article27', 'article28', 'article29', 'article3', 'article31', 'article4', 'article5', 'article6', 'article7', 'article8', 'article9', 'title10', 'title11', 'title12', 'title13', 'title14', 'title15', 'title16', 'title17', 'title18', 'title19', 'title20', 'title21', 'title22', 'title23', 'title24', 'title26', 'title28', 'title29', 'title3', 'title30', 'title31', 'title37', 'title38', 'title4', 'title40', 'title42', 'title5', 'title6', 'title7', 'title8', 'title9', 'meta10', 'meta11', 'meta12', 'meta13', 'meta14', 'meta15', 'meta16', 'meta17', 'meta18', 'meta19', 'meta20', 'meta21', 'meta22', 'meta23', 'meta24', 'meta25', 'meta26', 'meta27', 'meta28', 'meta29', 'meta3', 'meta31', 'meta37', 'meta38', 'meta4', 'meta5', 'meta6', 'meta7', 'meta8', 'meta9', 'center10', 'center11', 'center12', 'center13', 'center14', 'center15', 'center16', 'center17', 'center18', 'center19', 'center20', 'center21', 'center22', 'center23', 'center24', 'center25', 'center26', 'center27', 'center28', 'center29', 'center3', 'center30', 'center31', 'center32', 'center33', 'center34', 'center35', 'center36', 'center37', 'center38', 'center39', 'center4', 'center40', 'center41', 'center42', 'center43', 'center44', 'center45', 'center46', 'center47', 'center48', 'center49', 'center5', 'center50', 'center51', 'center52', 'center53', 'center54', 'center55', 'center56', 'center57', 'center58', 'center59', 'center6', 'center60', 'center61', 'center62', 'center63', 'center64', 'center65', 'center66', 'center67', 'center68', 'center69', 'center7', 'center70', 'center71', 'center8', 'center9','source10', 'source11', 'source12', 'source13', 'source14', 'source15', 'source16', 'source17', 'source18', 'source19', 'source20', 'source21', 'source22', 'source23', 'source24', 'source25', 'source26', 'source27', 'source28', 'source29', 'source30', 'source31', 'source32', 'source33', 'source34', 'source35', 'source4', 'source42', 'source5', 'source6', 'source7', 'source8', 'source9','small10', 'small11', 'small12', 'small13', 'small14', 'small15', 'small16', 'small17', 'small18', 'small19', 'small20', 'small21', 'small22', 'small23', 'small24', 'small25', 'small26', 'small27', 'small3', 'small30', 'small31', 'small32', 'small33', 'small4', 'small5', 'small6', 'small7', 'small8', 'small9','noscript10', 'noscript11', 'noscript12', 'noscript13', 'noscript14', 'noscript15', 'noscript16', 'noscript17', 'noscript18', 'noscript19', 'noscript20', 'noscript21', 'noscript22', 'noscript23', 'noscript24', 'noscript25', 'noscript26', 'noscript27', 'noscript28', 'noscript29', 'noscript3', 'noscript30', 'noscript31', 'noscript32', 'noscript36', 'noscript4', 'noscript43', 'noscript5', 'noscript6', 'noscript7', 'noscript8', 'noscript9','em10', 'em11', 'em12', 'em13', 'em14', 'em15', 'em16', 'em17', 'em18', 'em19', 'em20', 'em21', 'em22', 'em23', 'em24', 'em25', 'em26', 'em27', 'em28', 'em29', 'em3', 'em30', 'em31', 'em32', 'em33', 'em34', 'em35', 'em37', 'em4', 'em42', 'em45', 'em5', 'em6', 'em69', 'em7', 'em71', 'em77', 'em78', 'em79', 'em8', 'em84', 'em85', 'em9','link10', 'link11', 'link12', 'link13', 'link14', 'link15', 'link16', 'link17', 'link18', 'link19', 'link20', 'link21', 'link22', 'link23', 'link24', 'link25', 'link27', 'link28', 'link3', 'link31', 'link34', 'link36', 'link39', 'link4', 'link43', 'link5', 'link6', 'link7', 'link8', 'link9','script10', 'script11', 'script116', 'script12', 'script13', 'script14', 'script15', 'script16', 'script17', 'script18', 'script19', 'script20', 'script21', 'script22', 'script23', 'script24', 'script25', 'script26', 'script27', 'script28', 'script29', 'script3', 'script30', 'script31', 'script32', 'script33', 'script34', 'script35', 'script36', 'script37', 'script38', 'script39', 'script4', 'script40', 'script41', 'script43', 'script47', 'script48', 'script49', 'script5', 'script50', 'script52', 'script53', 'script57', 'script6', 'script62', 'script7', 'script8', 'script9','strong10', 'strong11', 'strong12', 'strong13', 'strong14', 'strong15', 'strong16', 'strong17', 'strong18', 'strong19', 'strong20', 'strong21', 'strong215', 'strong216', 'strong22', 'strong23', 'strong24', 'strong25', 'strong26', 'strong27', 'strong28', 'strong29', 'strong3', 'strong30', 'strong31', 'strong32', 'strong33', 'strong34', 'strong35', 'strong36', 'strong37', 'strong38', 'strong39', 'strong4', 'strong40', 'strong41', 'strong42', 'strong43', 'strong44', 'strong45', 'strong46', 'strong47', 'strong49', 'strong5', 'strong52', 'strong53', 'strong54', 'strong55', 'strong56', 'strong57', 'strong58', 'strong59', 'strong6', 'strong61', 'strong62', 'strong66', 'strong7', 'strong70', 'strong73', 'strong75', 'strong76', 'strong79', 'strong8', 'strong9', 'strong93','img10', 'img108', 'img11', 'img12', 'img13', 'img14', 'img15', 'img16', 'img17', 'img18', 'img19', 'img20', 'img21', 'img22', 'img23', 'img24', 'img25', 'img26', 'img27', 'img28', 'img29', 'img3', 'img30', 'img31', 'img32', 'img33', 'img34', 'img35', 'img36', 'img37', 'img38', 'img39', 'img4', 'img40', 'img41', 'img42', 'img43', 'img44', 'img45', 'img46', 'img47', 'img48', 'img49', 'img5', 'img50', 'img51', 'img52', 'img53', 'img54', 'img55', 'img56', 'img57', 'img58', 'img59', 'img6', 'img60', 'img61', 'img63', 'img64', 'img65', 'img66', 'img67', 'img7', 'img74', 'img76', 'img8', 'img83', 'img9']         
        dictB=['var10', 'var11', 'var12', 'var13', 'var14', 'var15', 'var16', 'var3', 'var4', 'var5', 'var6', 'var7', 'var8', 'var9', 'u10', 'u11', 'u12', 'u13', 'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23', 'u24', 'u25', 'u26', 'u27', 'u28', 'u29', 'u3', 'u30', 'u31', 'u32', 'u33', 'u34', 'u35', 'u36', 'u37', 'u38', 'u39', 'u4', 'u42', 'u43', 'u44', 'u48', 'u5', 'u53', 'u6', 'u7', 'u8', 'u9', 'tt11', 'tt7', 'tt8', 'summary10', 'summary11', 'summary15', 'summary7', 'summary8', 'summary9', 'samp10', 'samp11', 'samp12', 'samp13', 'samp14', 'samp16', 'samp5', 'samp6', 'samp8', 'samp9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's31', 's32', 's4', 's40', 's5', 's7', 's8', 's9', 'pre10', 'pre11', 'pre12', 'pre13', 'pre14', 'pre15', 'pre16', 'pre17', 'pre19', 'pre20', 'pre22', 'pre3', 'pre30', 'pre4', 'pre5', 'pre6', 'pre7', 'pre8', 'pre9', 'mark10', 'mark11', 'mark12', 'mark13', 'mark16', 'mark17', 'mark18', 'mark21', 'mark23', 'mark6', 'mark7', 'mark8', 'mark9', 'kbd10', 'kbd11', 'kbd14', 'kbd16', 'kbd7', 'kbd9', 'ins10', 'ins11', 'ins12', 'ins13', 'ins14', 'ins15', 'ins16', 'ins17', 'ins18', 'ins19', 'ins20', 'ins21', 'ins22', 'ins23', 'ins24', 'ins25', 'ins26', 'ins27', 'ins28', 'ins29', 'ins3', 'ins30', 'ins31', 'ins32', 'ins33', 'ins34', 'ins35', 'ins36', 'ins4', 'ins5', 'ins6', 'ins7', 'ins8', 'ins9', 'del10', 'del11', 'del12', 'del13', 'del14', 'del15', 'del16', 'del17', 'del18', 'del19', 'del20', 'del21', 'del22', 'del23', 'del25', 'del26', 'del28', 'del29', 'del33', 'del35', 'del5', 'del6', 'del7', 'del8', 'del9', 'b10', 'b101', 'b106', 'b11', 'b110', 'b111', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 'b18', 'b19', 'b20', 'b21', 'b212', 'b22', 'b23', 'b24', 'b25', 'b26', 'b27', 'b28', 'b29', 'b3', 'b30', 'b31', 'b32', 'b33', 'b34', 'b35', 'b36', 'b37', 'b38', 'b39', 'b4', 'b40', 'b41', 'b42', 'b43', 'b44', 'b45', 'b46', 'b47', 'b48', 'b49', 'b5', 'b50', 'b51', 'b52', 'b53', 'b55', 'b56', 'b57', 'b6', 'b61', 'b64', 'b65', 'b66', 'b67', 'b7', 'b70', 'b71', 'b76', 'b8', 'b81', 'b85', 'b86', 'b9', 'b91', 'b96', 'base10', 'base11', 'base12', 'base13', 'base14', 'base15', 'base16', 'base17', 'base18', 'base19', 'base21', 'base27', 'base3', 'base31', 'base4', 'base5', 'base6', 'base7', 'base8', 'base9', 'bdi10', 'bdi12', 'bdi13', 'bdi16', 'bdi17', 'bdi18', 'bdi19', 'bdi20', 'bdi21', 'bdi8', 'bdo24', 'bdo6', 'bdo8', 'big10', 'big100', 'big103', 'big105', 'big108', 'big11', 'big110', 'big12', 'big13', 'big14', 'big15', 'big16', 'big17', 'big18', 'big19', 'big20', 'big21', 'big22', 'big24', 'big25', 'big26', 'big27', 'big28', 'big3', 'big30', 'big33', 'big35', 'big38', 'big4', 'big40', 'big43', 'big45', 'big48', 'big5', 'big50', 'big53', 'big55', 'big58', 'big6', 'big60', 'big63', 'big65', 'big68', 'big7', 'big70', 'big73', 'big75', 'big78', 'big8', 'big80', 'big83', 'big85', 'big88', 'big9', 'big90', 'big93', 'big95', 'big98','code10', 'code11', 'code12', 'code13', 'code14', 'code15', 'code16', 'code17', 'code18', 'code21', 'code22', 'code3', 'code4', 'code5', 'code6', 'code7', 'code8', 'code9', 'href20', 'href6', 'href7', 'href8', 'href9', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i16', 'i17', 'i18', 'i19', 'i20', 'i21', 'i22', 'i23', 'i24', 'i25', 'i26', 'i27', 'i28', 'i29', 'i3', 'i30', 'i31', 'i32', 'i33', 'i34', 'i35', 'i36', 'i37', 'i38', 'i39', 'i4', 'i40', 'i42', 'i45', 'i5', 'i6', 'i7', 'i8', 'i9','thead10', 'thead11', 'thead12', 'thead13', 'thead14', 'thead15', 'thead16', 'thead17', 'thead18', 'thead19', 'thead20', 'thead21', 'thead22', 'thead23', 'thead24', 'thead25', 'thead26', 'thead27', 'thead28', 'thead29', 'thead32', 'thead34', 'thead4', 'thead5', 'thead6', 'thead7', 'thead8', 'thead9', 'tfoot10', 'tfoot11', 'tfoot12', 'tfoot13', 'tfoot14', 'tfoot15', 'tfoot16', 'tfoot18', 'tfoot21', 'tfoot5', 'tfoot6', 'tfoot7', 'tfoot8', 'tfoot9', 'sup10', 'sup11', 'sup12', 'sup13', 'sup14', 'sup15', 'sup16', 'sup17', 'sup18', 'sup19', 'sup20', 'sup21', 'sup22', 'sup23', 'sup24', 'sup25', 'sup26', 'sup27', 'sup29', 'sup33', 'sup35', 'sup36', 'sup37', 'sup38', 'sup4', 'sup5', 'sup6', 'sup7', 'sup8', 'sup9', 'sub10', 'sub11', 'sub12', 'sub13', 'sub14', 'sub15', 'sub16', 'sub19', 'sub5', 'sub6', 'sub7', 'sub8', 'sub9', 'legend10', 'legend11', 'legend12', 'legend13', 'legend14', 'legend15', 'legend16', 'legend17', 'legend18', 'legend19', 'legend20', 'legend21', 'legend22', 'legend23', 'legend24', 'legend25', 'legend26', 'legend27', 'legend28', 'legend29', 'legend30', 'legend31', 'legend5', 'legend6', 'legend7', 'legend8', 'legend9','details10', 'details11', 'details12', 'details13', 'details14', 'details16', 'details6', 'details7', 'details8', 'address10', 'address11', 'address12', 'address13', 'address14', 'address15', 'address16', 'address17', 'address18', 'address19', 'address20', 'address21', 'address22', 'address3', 'address34', 'address4', 'address5', 'address6', 'address7', 'address8', 'address9', 'caption10', 'caption11', 'caption12', 'caption13', 'caption14', 'caption15', 'caption16', 'caption17', 'caption18', 'caption19', 'caption20', 'caption21', 'caption22', 'caption23', 'caption25', 'caption27', 'caption5', 'caption6', 'caption7', 'caption8', 'caption9','wbr10', 'wbr11', 'wbr12', 'wbr13', 'wbr14', 'wbr15', 'wbr16', 'wbr17', 'wbr18', 'wbr19', 'wbr20', 'wbr21', 'wbr22', 'wbr23', 'wbr24', 'wbr25', 'wbr26', 'wbr27', 'wbr28', 'wbr29', 'wbr34', 'wbr36', 'wbr37', 'wbr7', 'wbr8', 'wbr9video10', 'video11', 'video12', 'video13', 'video14', 'video15', 'video16', 'video17', 'video18', 'video19', 'video20', 'video21', 'video23', 'video24', 'video25', 'video3', 'video4', 'video5', 'video6', 'video7', 'video8', 'video9', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q18', 'q19', 'q7', 'q8', 'q9', 'progress12', 'progress5', 'progress9', 'param10', 'param11', 'param12', 'param13', 'param14', 'param15', 'param16', 'param17', 'param18', 'param19', 'param20', 'param21', 'param22', 'param23', 'param24', 'param25', 'param26', 'param27', 'param3', 'param30', 'param31', 'param32', 'param35', 'param37', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9', 'object10', 'object11', 'object12', 'object13', 'object14', 'object15', 'object16', 'object17', 'object18', 'object19', 'object20', 'object21', 'object22', 'object23', 'object24', 'object25', 'object26', 'object29', 'object3', 'object30', 'object31', 'object34', 'object36', 'object4', 'object5', 'object6', 'object7', 'object8', 'object9', 'map10', 'map11', 'map12', 'map13', 'map14', 'map15', 'map16', 'map17', 'map18', 'map19', 'map20', 'map21', 'map22', 'map23', 'map24', 'map25', 'map28', 'map29', 'map3', 'map30', 'map31', 'map4', 'map43', 'map5', 'map6', 'map7', 'map8', 'map9', 'image10', 'image11', 'image12', 'image13', 'image14', 'image15', 'image16', 'image17', 'image18', 'image19', 'image20', 'image21', 'image22', 'image24', 'image25', 'image28', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'figcaption10', 'figcaption11', 'figcaption12', 'figcaption13', 'figcaption14', 'figcaption15', 'figcaption16', 'figcaption17', 'figcaption18', 'figcaption19', 'figcaption20', 'figcaption21', 'figcaption22', 'figcaption23', 'figcaption24', 'figcaption25', 'figcaption27', 'figcaption35', 'figcaption53', 'figcaption6', 'figcaption7', 'figcaption8', 'figcaption9', 'figure10', 'figure11', 'figure12', 'figure13', 'figure14', 'figure15', 'figure16', 'figure17', 'figure18', 'figure19', 'figure20', 'figure21', 'figure22', 'figure23', 'figure24', 'figure25', 'figure26', 'figure27', 'figure3', 'figure30', 'figure32', 'figure33', 'figure36', 'figure4', 'figure5', 'figure52', 'figure6', 'figure7', 'figure8', 'figure9', 'filter10', 'filter12', 'filter13', 'filter15', 'filter16', 'filter17', 'filter4', 'filter7', 'filter8', 'filter9', 'firure10', 'firure11', 'firure13', 'firure14', 'firure20', 'firure9', 'embed10', 'embed11', 'embed12', 'embed13', 'embed14', 'embed15', 'embed16', 'embed17', 'embed18', 'embed19', 'embed20', 'embed21', 'embed22', 'embed24', 'embed25', 'embed27', 'embed3', 'embed31', 'embed32', 'embed35', 'embed4', 'embed5', 'embed6', 'embed7', 'embed8', 'embed9', 'a10', 'a108', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'a21', 'a22', 'a23', 'a24', 'a25', 'a26', 'a27', 'a28', 'a29', 'a3', 'a30', 'a31', 'a32', 'a33', 'a34', 'a35', 'a36', 'a37', 'a38', 'a39', 'a4', 'a40', 'a41', 'a42', 'a43', 'a44', 'a45', 'a46', 'a47', 'a48', 'a49', 'a5', 'a50', 'a51', 'a52', 'a53', 'a54', 'a55', 'a56', 'a57', 'a58', 'a59', 'a6', 'a60', 'a61', 'a62', 'a63', 'a64', 'a65', 'a67', 'a7', 'a73', 'a76', 'a8', 'a83', 'a88', 'a9', 'a90', 'a92', 'a94', 'a96', 'a98', 'abbr10', 'abbr11', 'abbr12', 'abbr13', 'abbr14', 'abbr15', 'abbr16', 'abbr17', 'abbr18', 'abbr19', 'abbr20', 'abbr21', 'abbr22', 'abbr23', 'abbr25', 'abbr3', 'abbr30', 'abbr32', 'abbr34', 'abbr35', 'abbr36', 'abbr37', 'abbr4', 'abbr5', 'abbr6', 'abbr7', 'abbr8', 'abbr9', 'acronym10', 'acronym11', 'acronym12', 'acronym13', 'acronym14', 'acronym15', 'acronym16', 'acronym17', 'acronym18', 'acronym20', 'acronym22', 'acronym33', 'acronym34', 'acronym5', 'acronym6', 'acronym7', 'acronym8', 'acronym9', 'applet15', 'applet7', 'applet8', 'aside10', 'aside11', 'aside12', 'aside13', 'aside14', 'aside15', 'aside16', 'aside17', 'aside18', 'aside19', 'aside20', 'aside21', 'aside22', 'aside24', 'aside25', 'aside3', 'aside30', 'aside4', 'aside5', 'aside6', 'aside7', 'aside8', 'aside9', 'audio10', 'audio11', 'audio12', 'audio13', 'audio14', 'audio15', 'audio16', 'audio17', 'audio20', 'audio21', 'audio24', 'audio3', 'audio31', 'audio4', 'audio41', 'audio5', 'audio6', 'audio7', 'audio8', 'audio9', 'blockquote10', 'blockquote11', 'blockquote12', 'blockquote13', 'blockquote14', 'blockquote15', 'blockquote16', 'blockquote17', 'blockquote18', 'blockquote19', 'blockquote20', 'blockquote21', 'blockquote22', 'blockquote23', 'blockquote25', 'blockquote3', 'blockquote4', 'blockquote5', 'blockquote6', 'blockquote7', 'blockquote8', 'blockquote9', 'cite10', 'cite11', 'cite12', 'cite13', 'cite14', 'cite15', 'cite16', 'cite17', 'cite18', 'cite24', 'cite4', 'cite6', 'cite7', 'cite8', 'cite9','textarea10', 'textarea11', 'textarea12', 'textarea13', 'textarea14', 'textarea15', 'textarea16', 'textarea17', 'textarea18', 'textarea19', 'textarea20', 'textarea21', 'textarea22', 'textarea23', 'textarea24', 'textarea25', 'textarea26', 'textarea27', 'textarea28', 'textarea29', 'textarea3', 'textarea32', 'textarea39', 'textarea4', 'textarea5', 'textarea53', 'textarea6', 'textarea7', 'textarea71', 'textarea8', 'textarea9', 'section10', 'section11', 'section12', 'section13', 'section14', 'section15', 'section16', 'section17', 'section18', 'section19', 'section20', 'section21', 'section22', 'section23', 'section24', 'section25', 'section26', 'section27', 'section29', 'section3', 'section30', 'section33', 'section4', 'section43', 'section5', 'section6', 'section7', 'section8', 'section9', 'p10', 'p11', 'p117', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p20', 'p21', 'p216', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29', 'p3', 'p30', 'p31', 'p314', 'p32', 'p33', 'p34', 'p35', 'p36', 'p37', 'p38', 'p39', 'p4', 'p40', 'p41', 'p42', 'p43', 'p44', 'p45', 'p46', 'p47', 'p48', 'p5', 'p50', 'p52', 'p54', 'p55', 'p56', 'p58', 'p59', 'p6', 'p60', 'p61', 'p7', 'p8', 'p9', 'optgroup10', 'optgroup11', 'optgroup12', 'optgroup13', 'optgroup14', 'optgroup15', 'optgroup16', 'optgroup17', 'optgroup18', 'optgroup19', 'optgroup20', 'optgroup21', 'optgroup22', 'optgroup24', 'optgroup25', 'optgroup26', 'optgroup30', 'optgroup5', 'optgroup6', 'optgroup7', 'optgroup8', 'optgroup9', 'ol10', 'ol11', 'ol12', 'ol13', 'ol14', 'ol15', 'ol16', 'ol17', 'ol18', 'ol19', 'ol20', 'ol21', 'ol22', 'ol23', 'ol24', 'ol25', 'ol26', 'ol27', 'ol28', 'ol3', 'ol30', 'ol31', 'ol32', 'ol34', 'ol35', 'ol38', 'ol4', 'ol40', 'ol41', 'ol5', 'ol6', 'ol7', 'ol8', 'ol9', 'menu10', 'menu11', 'menu12', 'menu13', 'menu14', 'menu15', 'menu16', 'menu17', 'menu18', 'menu19', 'menu20', 'menu21', 'menu3', 'menu37', 'menu4', 'menu5', 'menu6', 'menu7', 'menu8', 'menu9', 'menuitem11', 'menuitem12', 'menuitem13', 'menuitem14', 'menuitem15', 'menuitem16', 'menuitem20', 'menuitem21', 'menuitem38', 'menuitem6', 'menuitem7', 'menuitem8', 'menuitem9','h03', 'h210', 'h211', 'h212', 'h213', 'h214', 'h215', 'h216', 'h217', 'h218', 'h219', 'h220', 'h221', 'h222', 'h223', 'h224', 'h225', 'h226', 'h227', 'h228', 'h229', 'h23', 'h230', 'h231', 'h232', 'h233', 'h234', 'h236', 'h237', 'h24', 'h241', 'h243', 'h244', 'h249', 'h25', 'h252', 'h253', 'h26', 'h27', 'h28', 'h29', 'h310', 'h311', 'h312', 'h313', 'h314', 'h315', 'h316', 'h317', 'h318', 'h319', 'h320', 'h321', 'h322', 'h323', 'h324', 'h325', 'h326', 'h327', 'h328', 'h329', 'h33', 'h330', 'h331', 'h3310', 'h332', 'h333', 'h334', 'h335', 'h337', 'h34', 'h341', 'h3410', 'h342', 'h35', 'h355', 'h359', 'h36', 'h37', 'h38', 'h39', 'h410', 'h411', 'h412', 'h413', 'h414', 'h415', 'h416', 'h417', 'h418', 'h419', 'h420', 'h421', 'h422', 'h423', 'h424', 'h425', 'h426', 'h427', 'h428', 'h429', 'h43', 'h430', 'h431', 'h432', 'h433', 'h434', 'h436', 'h44', 'h441', 'h442', 'h443', 'h444', 'h445', 'h45', 'h452', 'h46', 'h47', 'h48', 'h49', 'h510', 'h511', 'h512', 'h513', 'h514', 'h515', 'h516', 'h517', 'h518', 'h519', 'h520', 'h521', 'h522', 'h523', 'h524', 'h525', 'h526', 'h527', 'h528', 'h53', 'h530', 'h531', 'h536', 'h54', 'h55', 'h56', 'h57', 'h58', 'h59', 'h710', 'h711', 'h712', 'h713', 'h715', 'h77', 'h79', 'h8', 'h810', 'h815', 'h817', 'h85', 'h87', 'h921', 'dfn10', 'dfn11', 'dfn12', 'dfn13', 'dfn14', 'dfn15', 'dfn16', 'dfn17', 'dfn18', 'dfn5', 'dfn7', 'dfn8', 'dfn9', 'dialog13', 'dialog3', 'dialog5', 'dialog6', 'dl10', 'dl11', 'dl12', 'dl13', 'dl14', 'dl15', 'dl16', 'dl17', 'dl18', 'dl19', 'dl20', 'dl21', 'dl22', 'dl23', 'dl24', 'dl25', 'dl26', 'dl3', 'dl32', 'dl36', 'dl4', 'dl5', 'dl6', 'dl7', 'dl8', 'dl9', 'dt10', 'dt11', 'dt12', 'dt13', 'dt14', 'dt15', 'dt16', 'dt17', 'dt18', 'dt19', 'dt20', 'dt21', 'dt22', 'dt23', 'dt24', 'dt25', 'dt26', 'dt27', 'dt29', 'dt33', 'dt37', 'dt4', 'dt5', 'dt6', 'dt7', 'dt8', 'dt9', 'd11', 'd14', 'd17', 'd6', 'datalist11', 'datalist14', 'datalist6', 'datalist7', 'datalist9', 'dd10', 'dd11', 'dd12', 'dd13', 'dd14', 'dd15', 'dd16', 'dd17', 'dd18', 'dd19', 'dd20', 'dd21', 'dd22', 'dd23', 'dd24', 'dd25', 'dd26', 'dd27', 'dd33', 'dd4', 'dd5', 'dd6', 'dd7', 'dd8', 'dd9', 'body2', 'body5','col10', 'col11', 'col12', 'col13', 'col14', 'col15', 'col16', 'col17', 'col18', 'col19', 'col20', 'col21', 'col24', 'col28', 'col29', 'col36', 'col40', 'col5', 'col6', 'col7', 'col8', 'col9', 'colgroup10', 'colgroup11', 'colgroup12', 'colgroup13', 'colgroup14', 'colgroup15', 'colgroup16', 'colgroup17', 'colgroup18', 'colgroup19', 'colgroup20', 'colgroup23', 'colgroup27', 'colgroup28', 'colgroup35', 'colgroup39', 'colgroup4', 'colgroup5', 'colgroup6', 'colgroup7', 'colgroup8', 'colgroup9']

        strk="a10 a108 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20 a21 a22 a23 a24 a25 a26 a27 a28 a29 a3 a30 a31 a32 a33 a34 a35 a36 a37 a38 a39 a4 a40 a41 a42 a43 a44 a45 a46 a47 a48 a49 a5 a50 a51 a52 a53 a54 a55 a56 a57 a58 a59 a6 a60 a61 a62 a63 a64 a65 a67 a7 a73 a76 a8 a83 a88 a9 a90 a92 a94 a96 a98 abbr10 abbr11 abbr12 abbr13 abbr14 abbr15 abbr16 abbr17 abbr18 abbr19 abbr20 abbr21 abbr22 abbr23 abbr25 abbr3 abbr30 abbr32 abbr34 abbr35 abbr36 abbr37 abbr4 abbr5 abbr6 abbr7 abbr8 abbr9 acronym10 acronym11 acronym12 acronym13 acronym14 acronym15 acronym16 acronym17 acronym18 acronym20 acronym22 acronym33 acronym34 acronym5 acronym6 acronym7 acronym8 acronym9 address10 address11 address12 address13 address14 address15 address16 address17 address18 address19 address20 address21 address22 address3 address34 address4 address5 address6 address7 address8 address9 applet15 applet7 applet8 area10 area11 area12 area13 area14 area15 area16 area17 area18 area19 area20 area21 area22 area23 area24 area25 area26 area29 area3 area30 area31 area32 area4 area44 area5 area6 area7 area8 area9 article10 article11 article12 article13 article14 article15 article16 article17 article18 article19 article20 article21 article22 article23 article24 article25 article26 article27 article28 article29 article3 article31 article4 article5 article6 article7 article8 article9 aside10 aside11 aside12 aside13 aside14 aside15 aside16 aside17 aside18 aside19 aside20 aside21 aside22 aside24 aside25 aside3 aside30 aside4 aside5 aside6 aside7 aside8 aside9 audio10 audio11 audio12 audio13 audio14 audio15 audio16 audio17 audio20 audio21 audio24 audio3 audio31 audio4 audio41 audio5 audio6 audio7 audio8 audio9 b10 b101 b106 b11 b110 b111 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b212 b22 b23 b24 b25 b26 b27 b28 b29 b3 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b4 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b5 b50 b51 b52 b53 b55 b56 b57 b6 b61 b64 b65 b66 b67 b7 b70 b71 b76 b8 b81 b85 b86 b9 b91 b96 base10 base11 base12 base13 base14 base15 base16 base17 base18 base19 base21 base27 base3 base31 base4 base5 base6 base7 base8 base9 bdi10 bdi12 bdi13 bdi16 bdi17 bdi18 bdi19 bdi20 bdi21 bdi8 bdo24 bdo6 bdo8 big10 big100 big103 big105 big108 big11 big110 big12 big13 big14 big15 big16 big17 big18 big19 big20 big21 big22 big24 big25 big26 big27 big28 big3 big30 big33 big35 big38 big4 big40 big43 big45 big48 big5 big50 big53 big55 big58 big6 big60 big63 big65 big68 big7 big70 big73 big75 big78 big8 big80 big83 big85 big88 big9 big90 big93 big95 big98 blockquote10 blockquote11 blockquote12 blockquote13 blockquote14 blockquote15 blockquote16 blockquote17 blockquote18 blockquote19 blockquote20 blockquote21 blockquote22 blockquote23 blockquote25 blockquote3 blockquote4 blockquote5 blockquote6 blockquote7 blockquote8 blockquote9 body2 body5 br10 br109 br11 br12 br13 br14 br15 br16 br17 br18 br19 br20 br21 br22 br23 br24 br25 br26 br27 br28 br29 br3 br30 br31 br32 br33 br34 br35 br36 br37 br38 br39 br4 br40 br41 br42 br43 br44 br45 br46 br47 br48 br49 br5 br50 br51 br52 br54 br55 br56 br57 br59 br6 br60 br62 br63 br7 br78 br8 br84 br9 button10 button11 button12 button13 button14 button15 button16 button17 button18 button19 button20 button21 button22 button23 button24 button25 button26 button27 button28 button29 button3 button30 button31 button32 button33 button37 button38 button4 button41 button42 button43 button44 button45 button5 button53 button6 button7 button8 button9 canvas10 canvas11 canvas12 canvas13 canvas14 canvas15 canvas16 canvas17 canvas18 canvas19 canvas20 canvas21 canvas22 canvas23 canvas24 canvas25 canvas27 canvas3 canvas31 canvas35 canvas36 canvas4 canvas5 canvas6 canvas7 canvas8 canvas9 caption10 caption11 caption12 caption13 caption14 caption15 caption16 caption17 caption18 caption19 caption20 caption21 caption22 caption23 caption25 caption27 caption5 caption6 caption7 caption8 caption9 center10 center11 center12 center13 center14 center15 center16 center17 center18 center19 center20 center21 center22 center23 center24 center25 center26 center27 center28 center29 center3 center30 center31 center32 center33 center34 center35 center36 center37 center38 center39 center4 center40 center41 center42 center43 center44 center45 center46 center47 center48 center49 center5 center50 center51 center52 center53 center54 center55 center56 center57 center58 center59 center6 center60 center61 center62 center63 center64 center65 center66 center67 center68 center69 center7 center70 center71 center8 center9 cite10 cite11 cite12 cite13 cite14 cite15 cite16 cite17 cite18 cite24 cite4 cite6 cite7 cite8 cite9 code10 code11 code12 code13 code14 code15 code16 code17 code18 code21 code22 code3 code4 code5 code6 code7 code8 code9 col10 col11 col12 col13 col14 col15 col16 col17 col18 col19 col20 col21 col24 col28 col29 col36 col40 col5 col6 col7 col8 col9 colgroup10 colgroup11 colgroup12 colgroup13 colgroup14 colgroup15 colgroup16 colgroup17 colgroup18 colgroup19 colgroup20 colgroup23 colgroup27 colgroup28 colgroup35 colgroup39 colgroup4 colgroup5 colgroup6 colgroup7 colgroup8 colgroup9 d11 d14 d17 d6 datalist11 datalist14 datalist6 datalist7 datalist9 dd10 dd11 dd12 dd13 dd14 dd15 dd16 dd17 dd18 dd19 dd20 dd21 dd22 dd23 dd24 dd25 dd26 dd27 dd33 dd4 dd5 dd6 dd7 dd8 dd9 del10 del11 del12 del13 del14 del15 del16 del17 del18 del19 del20 del21 del22 del23 del25 del26 del28 del29 del33 del35 del5 del6 del7 del8 del9 details10 details11 details12 details13 details14 details16 details6 details7 details8 dfn10 dfn11 dfn12 dfn13 dfn14 dfn15 dfn16 dfn17 dfn18 dfn5 dfn7 dfn8 dfn9 dialog13 dialog3 dialog5 dialog6 div10 div100 div101 div102 div103 div104 div105 div106 div107 div11 div12 div13 div14 div15 div16 div17 div18 div19 div20 div21 div22 div23 div24 div25 div26 div27 div28 div29 div3 div30 div31 div32 div33 div34 div35 div36 div37 div38 div39 div4 div40 div41 div42 div43 div44 div45 div46 div47 div48 div49 div5 div50 div51 div52 div53 div54 div55 div56 div57 div58 div59 div6 div60 div61 div62 div63 div64 div65 div66 div67 div68 div69 div7 div70 div71 div72 div73 div74 div75 div76 div77 div78 div79 div8 div80 div81 div82 div83 div84 div85 div86 div87 div88 div89 div9 div90 div91 div92 div93 div94 div95 div96 div97 div98 div99 dl10 dl11 dl12 dl13 dl14 dl15 dl16 dl17 dl18 dl19 dl20 dl21 dl22 dl23 dl24 dl25 dl26 dl3 dl32 dl36 dl4 dl5 dl6 dl7 dl8 dl9 dt10 dt11 dt12 dt13 dt14 dt15 dt16 dt17 dt18 dt19 dt20 dt21 dt22 dt23 dt24 dt25 dt26 dt27 dt29 dt33 dt37 dt4 dt5 dt6 dt7 dt8 dt9 em10 em11 em12 em13 em14 em15 em16 em17 em18 em19 em20 em21 em22 em23 em24 em25 em26 em27 em28 em29 em3 em30 em31 em32 em33 em34 em35 em37 em4 em42 em45 em5 em6 em69 em7 em71 em77 em78 em79 em8 em84 em85 em9 embed10 embed11 embed12 embed13 embed14 embed15 embed16 embed17 embed18 embed19 embed20 embed21 embed22 embed24 embed25 embed27 embed3 embed31 embed32 embed35 embed4 embed5 embed6 embed7 embed8 embed9 fieldset10 fieldset11 fieldset12 fieldset13 fieldset14 fieldset15 fieldset16 fieldset17 fieldset18 fieldset19 fieldset20 fieldset21 fieldset22 fieldset23 fieldset24 fieldset25 fieldset26 fieldset27 fieldset28 fieldset29 fieldset3 fieldset30 fieldset33 fieldset4 fieldset5 fieldset6 fieldset7 fieldset8 fieldset9 figcaption10 figcaption11 figcaption12 figcaption13 figcaption14 figcaption15 figcaption16 figcaption17 figcaption18 figcaption19 figcaption20 figcaption21 figcaption22 figcaption23 figcaption24 figcaption25 figcaption27 figcaption35 figcaption53 figcaption6 figcaption7 figcaption8 figcaption9 figure10 figure11 figure12 figure13 figure14 figure15 figure16 figure17 figure18 figure19 figure20 figure21 figure22 figure23 figure24 figure25 figure26 figure27 figure3 figure30 figure32 figure33 figure36 figure4 figure5 figure52 figure6 figure7 figure8 figure9 filter10 filter12 filter13 filter15 filter16 filter17 filter4 filter7 filter8 filter9 firure10 firure11 firure13 firure14 firure20 firure9 font10 font104 font109 font11 font12 font13 font14 font15 font16 font17 font18 font19 font20 font21 font22 font23 font24 font25 font26 font27 font28 font29 font3 font30 font31 font32 font33 font34 font35 font36 font37 font38 font39 font4 font40 font41 font42 font43 font44 font45 font46 font47 font48 font49 font5 font50 font51 font53 font54 font55 font56 font57 font58 font59 font6 font60 font61 font62 font63 font64 font65 font69 font7 font74 font79 font8 font84 font89 font9 font94 font99 foooter4 footer10 footer11 footer12 footer13 footer14 footer15 footer16 footer17 footer18 footer19 footer20 footer22 footer24 footer27 footer3 footer30 footer4 footer43 footer5 footer6 footer7 footer8 footer9 form10 form11 form12 form13 form14 form15 form16 form17 form18 form19 form20 form21 form22 form23 form24 form25 form26 form27 form28 form29 form3 form30 form31 form32 form34 form35 form36 form37 form4 form40 form44 form48 form49 form5 form50 form57 form6 form60 form62 form68 form7 form8 form9 h03 h1020 h110 h111 h112 h113 h114 h115 h116 h117 h118 h119 h120 h121 h122 h123 h124 h125 h126 h127 h128 h129 h13 h130 h131 h132 h133 h134 h135 h137 h138 h139 h14 h140 h141 h1413 h15 h16 h1613 h17 h18 h19 h210 h211 h212 h213 h214 h215 h216 h217 h218 h219 h220 h221 h222 h223 h224 h225 h226 h227 h228 h229 h23 h230 h231 h232 h233 h234 h236 h237 h24 h241 h243 h244 h249 h25 h252 h253 h26 h27 h28 h29 h310 h311 h312 h313 h314 h315 h316 h317 h318 h319 h320 h321 h322 h323 h324 h325 h326 h327 h328 h329 h33 h330 h331 h3310 h332 h333 h334 h335 h337 h34 h341 h3410 h342 h35 h355 h359 h36 h37 h38 h39 h410 h411 h412 h413 h414 h415 h416 h417 h418 h419 h420 h421 h422 h423 h424 h425 h426 h427 h428 h429 h43 h430 h431 h432 h433 h434 h436 h44 h441 h442 h443 h444 h445 h45 h452 h46 h47 h48 h49 h510 h511 h512 h513 h514 h515 h516 h517 h518 h519 h520 h521 h522 h523 h524 h525 h526 h527 h528 h53 h530 h531 h536 h54 h55 h56 h57 h58 h59 h610 h611 h612 h613 h614 h615 h616 h617 h618 h619 h620 h621 h622 h623 h624 h625 h626 h627 h628 h629 h630 h64 h65 h66 h67 h68 h69 h710 h711 h712 h713 h715 h77 h79 h8 h810 h815 h817 h85 h87 h921 header10 header11 header12 header13 header14 header15 header16 header17 header18 header19 header20 header21 header22 header23 header24 header26 header27 header29 header3 header30 header4 header5 header6 header7 header8 header9 hr10 hr11 hr12 hr13 hr14 hr15 hr16 hr17 hr18 hr19 hr20 hr21 hr22 hr23 hr24 hr25 hr26 hr27 hr28 hr29 hr3 hr30 hr31 hr34 hr35 hr37 hr38 hr4 hr40 hr41 hr43 hr45 hr5 hr54 hr55 hr56 hr6 hr7 hr8 hr9 href20 href6 href7 href8 href9 i10 i11 i12 i13 i14 i15 i16 i17 i18 i19 i20 i21 i22 i23 i24 i25 i26 i27 i28 i29 i3 i30 i31 i32 i33 i34 i35 i36 i37 i38 i39 i4 i40 i42 i45 i5 i6 i7 i8 i9 iframe10 iframe11 iframe12 iframe13 iframe14 iframe15 iframe16 iframe17 iframe18 iframe19 iframe20 iframe21 iframe22 iframe23 iframe24 iframe25 iframe26 iframe27 iframe28 iframe29 iframe3 iframe30 iframe31 iframe32 iframe33 iframe34 iframe35 iframe36 iframe37 iframe39 iframe4 iframe40 iframe43 iframe44 iframe5 iframe50 iframe6 iframe7 iframe8 iframe9 image10 image11 image12 image13 image14 image15 image16 image17 image18 image19 image20 image21 image22 image24 image25 image28 image4 image5 image6 image7 image8 image9 img10 img108 img11 img12 img13 img14 img15 img16 img17 img18 img19 img20 img21 img22 img23 img24 img25 img26 img27 img28 img29 img3 img30 img31 img32 img33 img34 img35 img36 img37 img38 img39 img4 img40 img41 img42 img43 img44 img45 img46 img47 img48 img49 img5 img50 img51 img52 img53 img54 img55 img56 img57 img58 img59 img6 img60 img61 img63 img64 img65 img66 img67 img7 img74 img76 img8 img83 img9 input10 input11 input12 input13 input14 input15 input16 input17 input18 input19 input20 input21 input22 input23 input24 input25 input26 input27 input28 input29 input3 input30 input31 input32 input33 input34 input35 input36 input37 input38 input39 input4 input40 input41 input42 input43 input44 input45 input46 input48 input49 input5 input51 input53 input54 input55 input57 input58 input6 input60 input61 input63 input69 input7 input71 input75 input8 input85 input87 input9 input93 ins10 ins11 ins12 ins13 ins14 ins15 ins16 ins17 ins18 ins19 ins20 ins21 ins22 ins23 ins24 ins25 ins26 ins27 ins28 ins29 ins3 ins30 ins31 ins32 ins33 ins34 ins35 ins36 ins4 ins5 ins6 ins7 ins8 ins9 item10 item11 item12 item13 item14 item15 item16 item5 item8 kbd10 kbd11 kbd14 kbd16 kbd7 kbd9 label10 label11 label12 label13 label14 label15 label16 label17 label18 label19 label20 label21 label22 label23 label24 label25 label26 label27 label28 label29 label3 label30 label31 label32 label33 label34 label35 label36 label37 label38 label39 label4 label40 label41 label42 label43 label46 label5 label6 label7 label8 label9 lable8 legend10 legend11 legend12 legend13 legend14 legend15 legend16 legend17 legend18 legend19 legend20 legend21 legend22 legend23 legend24 legend25 legend26 legend27 legend28 legend29 legend30 legend31 legend5 legend6 legend7 legend8 legend9 li10 li11 li12 li13 li14 li15 li16 li17 li18 li19 li20 li21 li210 li22 li23 li24 li25 li26 li27 li28 li29 li3 li30 li31 li32 li33 li34 li35 li36 li37 li39 li4 li40 li41 li42 li43 li44 li45 li46 li48 li49 li5 li50 li55 li57 li6 li60 li61 li62 li7 li8 li9 link10 link11 link12 link13 link14 link15 link16 link17 link18 link19 link20 link21 link22 link23 link24 link25 link27 link28 link3 link31 link34 link36 link39 link4 link43 link5 link6 link7 link8 link9 map10 map11 map12 map13 map14 map15 map16 map17 map18 map19 map20 map21 map22 map23 map24 map25 map28 map29 map3 map30 map31 map4 map43 map5 map6 map7 map8 map9 mark10 mark11 mark12 mark13 mark16 mark17 mark18 mark21 mark23 mark6 mark7 mark8 mark9 menu10 menu11 menu12 menu13 menu14 menu15 menu16 menu17 menu18 menu19 menu20 menu21 menu3 menu37 menu4 menu5 menu6 menu7 menu8 menu9 menuitem11 menuitem12 menuitem13 menuitem14 menuitem15 menuitem16 menuitem20 menuitem21 menuitem38 menuitem6 menuitem7 menuitem8 menuitem9 meta10 meta11 meta12 meta13 meta14 meta15 meta16 meta17 meta18 meta19 meta20 meta21 meta22 meta23 meta24 meta25 meta26 meta27 meta28 meta29 meta3 meta31 meta37 meta38 meta4 meta5 meta6 meta7 meta8 meta9 nav10 nav11 nav12 nav13 nav14 nav15 nav16 nav17 nav18 nav19 nav20 nav21 nav22 nav23 nav24 nav25 nav26 nav27 nav3 nav4 nav5 nav6 nav7 nav8 nav9 noscript10 noscript11 noscript12 noscript13 noscript14 noscript15 noscript16 noscript17 noscript18 noscript19 noscript20 noscript21 noscript22 noscript23 noscript24 noscript25 noscript26 noscript27 noscript28 noscript29 noscript3 noscript30 noscript31 noscript32 noscript36 noscript4 noscript43 noscript5 noscript6 noscript7 noscript8 noscript9 object10 object11 object12 object13 object14 object15 object16 object17 object18 object19 object20 object21 object22 object23 object24 object25 object26 object29 object3 object30 object31 object34 object36 object4 object5 object6 object7 object8 object9 ol10 ol11 ol12 ol13 ol14 ol15 ol16 ol17 ol18 ol19 ol20 ol21 ol22 ol23 ol24 ol25 ol26 ol27 ol28 ol3 ol30 ol31 ol32 ol34 ol35 ol38 ol4 ol40 ol41 ol5 ol6 ol7 ol8 ol9 optgroup10 optgroup11 optgroup12 optgroup13 optgroup14 optgroup15 optgroup16 optgroup17 optgroup18 optgroup19 optgroup20 optgroup21 optgroup22 optgroup24 optgroup25 optgroup26 optgroup30 optgroup5 optgroup6 optgroup7 optgroup8 optgroup9 option10 option11 option12 option13 option14 option15 option16 option17 option18 option19 option20 option21 option22 option23 option24 option25 option26 option27 option28 option29 option30 option31 option32 option33 option34 option35 option36 option37 option38 option4 option40 option42 option43 option45 option46 option5 option6 option7 option8 option9 p10 p11 p117 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p216 p22 p23 p24 p25 p26 p27 p28 p29 p3 p30 p31 p314 p32 p33 p34 p35 p36 p37 p38 p39 p4 p40 p41 p42 p43 p44 p45 p46 p47 p48 p5 p50 p52 p54 p55 p56 p58 p59 p6 p60 p61 p7 p8 p9 param10 param11 param12 param13 param14 param15 param16 param17 param18 param19 param20 param21 param22 param23 param24 param25 param26 param27 param3 param30 param31 param32 param35 param37 param4 param5 param6 param7 param8 param9 pre10 pre11 pre12 pre13 pre14 pre15 pre16 pre17 pre19 pre20 pre22 pre3 pre30 pre4 pre5 pre6 pre7 pre8 pre9 progress12 progress5 progress9 q10 q11 q12 q13 q14 q15 q16 q18 q19 q7 q8 q9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 s31 s32 s4 s40 s5 s7 s8 s9 samp10 samp11 samp12 samp13 samp14 samp16 samp5 samp6 samp8 samp9 script10 script11 script116 script12 script13 script14 script15 script16 script17 script18 script19 script20 script21 script22 script23 script24 script25 script26 script27 script28 script29 script3 script30 script31 script32 script33 script34 script35 script36 script37 script38 script39 script4 script40 script41 script43 script47 script48 script49 script5 script50 script52 script53 script57 script6 script62 script7 script8 script9 section10 section11 section12 section13 section14 section15 section16 section17 section18 section19 section20 section21 section22 section23 section24 section25 section26 section27 section29 section3 section30 section33 section4 section43 section5 section6 section7 section8 section9 select10 select11 select12 select13 select14 select15 select16 select17 select18 select19 select20 select21 select22 select23 select24 select25 select26 select27 select28 select29 select3 select30 select31 select32 select33 select34 select35 select36 select37 select39 select4 select41 select42 select44 select45 select5 select6 select7 select8 select9 small10 small11 small12 small13 small14 small15 small16 small17 small18 small19 small20 small21 small22 small23 small24 small25 small26 small27 small3 small30 small31 small32 small33 small4 small5 small6 small7 small8 small9 source10 source11 source12 source13 source14 source15 source16 source17 source18 source19 source20 source21 source22 source23 source24 source25 source26 source27 source28 source29 source30 source31 source32 source33 source34 source35 source4 source42 source5 source6 source7 source8 source9 span10 span100 span101 span102 span103 span104 span105 span106 span107 span108 span109 span11 span111 span112 span113 span12 span13 span14 span15 span16 span17 span18 span19 span20 span21 span22 span23 span24 span25 span26 span27 span28 span29 span3 span30 span31 span32 span33 span34 span35 span36 span37 span38 span39 span4 span40 span41 span42 span43 span44 span45 span46 span47 span48 span49 span5 span50 span51 span52 span53 span54 span55 span56 span57 span58 span59 span6 span60 span61 span62 span63 span64 span65 span66 span67 span68 span69 span7 span70 span71 span72 span73 span74 span75 span76 span77 span78 span79 span8 span80 span81 span82 span83 span84 span85 span86 span87 span88 span89 span9 span90 span91 span92 span93 span94 span95 span96 span97 span98 span99 strong10 strong11 strong12 strong13 strong14 strong15 strong16 strong17 strong18 strong19 strong20 strong21 strong215 strong216 strong22 strong23 strong24 strong25 strong26 strong27 strong28 strong29 strong3 strong30 strong31 strong32 strong33 strong34 strong35 strong36 strong37 strong38 strong39 strong4 strong40 strong41 strong42 strong43 strong44 strong45 strong46 strong47 strong49 strong5 strong52 strong53 strong54 strong55 strong56 strong57 strong58 strong59 strong6 strong61 strong62 strong66 strong7 strong70 strong73 strong75 strong76 strong79 strong8 strong9 strong93 style10 style11 style12 style13 style14 style15 style16 style17 style18 style19 style20 style21 style22 style23 style24 style25 style26 style27 style28 style29 style3 style30 style31 style34 style35 style36 style38 style39 style4 style46 style5 style6 style7 style8 style9 sub10 sub11 sub12 sub13 sub14 sub15 sub16 sub19 sub5 sub6 sub7 sub8 sub9 summary10 summary11 summary15 summary7 summary8 summary9 sup10 sup11 sup12 sup13 sup14 sup15 sup16 sup17 sup18 sup19 sup20 sup21 sup22 sup23 sup24 sup25 sup26 sup27 sup29 sup33 sup35 sup36 sup37 sup38 sup4 sup5 sup6 sup7 sup8 sup9 table10 table11 table12 table13 table14 table15 table16 table17 table18 table19 table20 table21 table22 table23 table24 table25 table26 table27 table28 table29 table3 table30 table31 table32 table33 table34 table35 table36 table37 table38 table39 table4 table40 table41 table42 table43 table44 table45 table46 table47 table48 table49 table5 table50 table51 table52 table53 table54 table55 table56 table57 table58 table6 table60 table61 table62 table63 table64 table65 table66 table67 table68 table69 table7 table70 table71 table72 table73 table74 table75 table76 table77 table79 table8 table80 table81 table83 table85 table89 table9 tbody10 tbody11 tbody12 tbody13 tbody14 tbody15 tbody16 tbody17 tbody18 tbody19 tbody20 tbody21 tbody22 tbody23 tbody24 tbody25 tbody26 tbody27 tbody28 tbody29 tbody30 tbody31 tbody32 tbody33 tbody34 tbody35 tbody36 tbody37 tbody38 tbody39 tbody4 tbody40 tbody41 tbody42 tbody43 tbody44 tbody45 tbody46 tbody47 tbody48 tbody49 tbody5 tbody50 tbody51 tbody52 tbody53 tbody54 tbody55 tbody56 tbody57 tbody58 tbody59 tbody6 tbody61 tbody62 tbody63 tbody64 tbody65 tbody66 tbody67 tbody68 tbody69 tbody7 tbody70 tbody71 tbody72 tbody73 tbody74 tbody75 tbody76 tbody77 tbody78 tbody8 tbody80 tbody81 tbody82 tbody84 tbody86 tbody9 tbody90 td10 td11 td12 td13 td14 td15 td16 td17 td18 td19 td20 td21 td22 td23 td24 td25 td26 td27 td28 td29 td30 td31 td32 td33 td34 td35 td36 td37 td38 td39 td4 td40 td41 td42 td43 td44 td45 td46 td47 td48 td49 td50 td51 td52 td53 td54 td55 td56 td57 td58 td59 td6 td60 td61 td63 td64 td65 td66 td67 td68 td69 td7 td70 td71 td72 td73 td74 td75 td76 td77 td78 td79 td8 td80 td82 td83 td84 td86 td88 td9 td92 textarea10 textarea11 textarea12 textarea13 textarea14 textarea15 textarea16 textarea17 textarea18 textarea19 textarea20 textarea21 textarea22 textarea23 textarea24 textarea25 textarea26 textarea27 textarea28 textarea29 textarea3 textarea32 textarea39 textarea4 textarea5 textarea53 textarea6 textarea7 textarea71 textarea8 textarea9 tfoot10 tfoot11 tfoot12 tfoot13 tfoot14 tfoot15 tfoot16 tfoot18 tfoot21 tfoot5 tfoot6 tfoot7 tfoot8 tfoot9 th10 th11 th12 th13 th14 th15 th16 th17 th18 th19 th20 th21 th22 th23 th24 th25 th26 th27 th28 th29 th30 th31 th32 th34 th36 th38 th6 th7 th8 th9 thead10 thead11 thead12 thead13 thead14 thead15 thead16 thead17 thead18 thead19 thead20 thead21 thead22 thead23 thead24 thead25 thead26 thead27 thead28 thead29 thead32 thead34 thead4 thead5 thead6 thead7 thead8 thead9 title10 title11 title12 title13 title14 title15 title16 title17 title18 title19 title20 title21 title22 title23 title24 title26 title28 title29 title3 title30 title31 title37 title38 title4 title40 title42 title5 title6 title7 title8 title9 tr10 tr11 tr12 tr13 tr14 tr15 tr16 tr17 tr18 tr19 tr20 tr21 tr22 tr23 tr24 tr25 tr26 tr27 tr28 tr29 tr30 tr31 tr32 tr33 tr34 tr35 tr36 tr37 tr38 tr39 tr40 tr41 tr42 tr43 tr44 tr45 tr46 tr47 tr48 tr49 tr5 tr50 tr51 tr52 tr53 tr54 tr55 tr56 tr57 tr58 tr59 tr6 tr60 tr62 tr63 tr64 tr65 tr66 tr67 tr68 tr69 tr7 tr70 tr71 tr72 tr73 tr74 tr75 tr76 tr77 tr78 tr79 tr8 tr81 tr82 tr83 tr85 tr87 tr9 tr91 tt11 tt7 tt8 u10 u11 u12 u13 u14 u15 u16 u17 u18 u19 u20 u21 u22 u23 u24 u25 u26 u27 u28 u29 u3 u30 u31 u32 u33 u34 u35 u36 u37 u38 u39 u4 u42 u43 u44 u48 u5 u53 u6 u7 u8 u9 ul10 ul11 ul12 ul13 ul14 ul15 ul16 ul17 ul18 ul19 ul20 ul21 ul22 ul23 ul24 ul25 ul26 ul27 ul28 ul29 ul3 ul30 ul31 ul32 ul33 ul34 ul35 ul36 ul38 ul39 ul4 ul40 ul41 ul42 ul43 ul44 ul45 ul47 ul48 ul49 ul5 ul53 ul54 ul56 ul59 ul6 ul60 ul61 ul7 ul8 ul9 var10 var11 var12 var13 var14 var15 var16 var3 var4 var5 var6 var7 var8 var9 video10 video11 video12 video13 video14 video15 video16 video17 video18 video19 video20 video21 video23 video24 video25 video3 video4 video5 video6 video7 video8 video9 wbr10 wbr11 wbr12 wbr13 wbr14 wbr15 wbr16 wbr17 wbr18 wbr19 wbr20 wbr21 wbr22 wbr23 wbr24 wbr25 wbr26 wbr27 wbr28 wbr29 wbr34 wbr36 wbr37 wbr7 wbr8 wbr9"
        veclistDict = strk.split(' ')
        vectorizer=CountVectorizer(vocabulary=veclistDict)#
        transformer=TfidfTransformer()#每个词语的tf-idf权值  
        X1 = vectorizer.fit_transform(res) #词频矩阵
        X2 = X1.toarray()#词频矩阵
        tfidf=transformer.fit_transform(X1)
        weight=tfidf.toarray()#元素a[i][j] 
        ########
        #tagWeight=[]
        tagWeightIn=[]
        
        for i in range(len(veclistDict)):  #词典的长度
                if veclistDict[i] in dictA:
                    tagWeightIn.append(0.7/0.3)
                    break
                elif veclistDict[i] in dictB:
                    tagWeightIn.append(0.3/0.3)
                    break
                 
        listNew=np.array(tagWeightIn)    
        newWeight=weight*listNew
        ########
        
        fink = self.pca(newWeight,0.81) #0.9084 #0.83  #0.83(tag+class)#0.8544 #0.81
        vec = fink.tolist()
        for i in range(len(vec)):
            vec[i].append(classRes[i])
            vec[i].append(lz78Res[i])
            
          
        return vec   #词频矩阵
    
    '''
    #观察测试集是否在差异范围之内
    def isFPDiff(self,point,centerSet,threshold):
        #在聚好类之后，对测试集中的网页与中心点集合进行相似度计算，从而计算出评价指标
        #print 'point:',point
        if threshold > 1:
            isPercent = False
        else:
            isPercent = True
        seqClasss=','.join(point[-2])+','    
        item=fingerprint(point[-1]+seqClasss)   #testdata and centerSet
        
        for centre in centerSet:
            seqClass=','.join(centre[-2])+','
            centrePoint=fingerprint(centre[-1]+seqClass)
            #print centrePoint
            if self.calculateFPDiff(item, centrePoint, isPercent=isPercent) <= 0.25:
                #tagSeq是列表中的最后一个元素
                #print 'True'
                return True
        
        return False
    '''

    def secondFP(self,tagseqFP1):#二次压缩
        j=1
        num=1
        listNum=[]
        listSeq=[]
        for i in range(len(tagseqFP1)):
            #for j in range(len(tagseqFP2)):
                j=i+1
                if j == len((tagseqFP1)):
                    listNum.append(1)
                    listSeq.append(tagseqFP1[j-1])
                    break
                if tagseqFP1[i] == tagseqFP1[j]:
                    #j += 1
                    num+=1
                else:
                    listNum.append(num)
                    listSeq.append(tagseqFP1[i])
                    num=1
        listSec=[]         
        for it in range(len(listNum)):
            if listNum[it] != 1:
                listSec.append(listSeq[it])
                listSec.append(listNum[it])
            else:    
                listSec.append(listSeq[it])
                
        return (listSec)  #返回新的压缩序列
        
    def isFPDiff(self,point,centerSet,threshold):
        
        if threshold > 1:
            isPercent = False
        else:
            isPercent = True
            
        seqClasss=','.join(point[-2])+','    
        item = fingerprint(point[-1])   #testdata and centerSet
        item = self.secondFP(item)       #二次压缩
        for centre in centerSet:
            
            seqClass = ','.join(centre[-2])+','
            centrePoint = fingerprint(centre[-1])
            centrePoint = self.secondFP(centrePoint)  #二次压缩
            #print centrePoint
            if self.calculateFPDiff(item, centrePoint, isPercent=isPercent) <= 0.2:
                #tagSeq是列表中的最后一个元素
                return True
        
        return False
    #比较指纹，二次压缩的新的算法
    def calculateFPDiff(self, vector1, vector2, splitSym = ',', isPercent = True):
        #sepcial case
        veclist1 = vector1
        veclist2 = vector2
        diffCnt = 0
        validCnt = 0
    
        maxVal=max(len(veclist1),len(veclist2))
        minVal=min(len(veclist1),len(veclist2))  
        for i in range(minVal):    
            if veclist1[i] != veclist2[i]:
                diffCnt += 1  
            #
        diffCnt=diffCnt+(maxVal-minVal)
        #validCnt=validCnt+(maxVal-minVal)
        validCnt= maxVal       
        if isPercent: 
            diffCnt = float(diffCnt) / float(validCnt)
        return diffCnt
    '''
    def calculateFPDiff(self, vector1, vector2, splitSym = ',', isPercent = True):
        #sepcial case

        if vector1 == vector2:
            return 0
        veclist1 = vector1
        veclist2 = vector2
        diffCnt = 0
        validCnt = 0
        #for i in range(len(veclist1)):
        maxVal=max(len(veclist1),len(veclist2))
        minVal=min(len(veclist1),len(veclist2))            
        for i in range(min(len(veclist1),len(veclist2))):    
            if veclist1[i] != veclist2[i]:
                diffCnt += 1  
            if veclist1[i] != "0" or veclist2[i] != "0":
                validCnt += 1
        if isPercent: 
            diffCnt = float(diffCnt+maxVal-minVal) / float(max(len(veclist1),len(veclist2)))
    
        return diffCnt
    '''
    ####    
    def isHitHRange(self, vector, centreSet, threshold):
        
        if threshold > 1:
            isPercent = False
        else:
            isPercent = True
        
        for centre in centreSet:
            if self.calculateVectorDiff(vector, centre, isPercent=isPercent) <= threshold:
                return True
        
        return False
        
    #used for chain clustering checking
    def isHitCRange(self, vector, groupList, threshold):
    
        if threshold > 1:
            isPercent = False
        else:
            isPercent = True
            
        for group in groupList:
            #remove duplicate vector
            #group = set(group)
            group = group
            
            for elem in group:
                if self.calculateVectorDiff(vector, elem, isPercent=isPercent) <= threshold:
                    return True, group
        
        return False, None
    '''
    def tfidf(self,res):
        vectorizer=CountVectorizer()#矩阵元素a[i][j] 表示j词在i类文本下的词频  
        transformer=TfidfTransformer()#统计每个词语的tf-idf权值  
        tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵  
        word=vectorizer.get_feature_names()#获取词袋模型中的所有词语  
        print ('标签种类：',list(word))
        weight=tfidf.toarray()#元素a[i][j]表示j词在i类文本中的tf-idf权重  
        #for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
            
            #for j in range(len(word)):  
                #print ('word:',word[j])
            #print('weight:',list(weight[i]))
        return weight   #返回词频矩阵
    '''
    ################################################# 
    def jaccard_similarity(self,set1, set2):
        set1 = set(set1)
        set2 = set(set2)
        intersection = len(set1 & set2)
    
        if len(set1) == 0 and len(set2) == 0:
            return 0

        denominator = len(set1) + len(set2) - intersection
        return 1.0-(float(intersection) / float(max(denominator, 0.000001)))
    '''
    def calculateVectorDiff(self,vector1, vector2,splitSym = ',', isPercent = True):  #计算两个向量之间的余弦相似度
        #
        list1=copy.copy(vector1)
        list2=copy.copy(vector2)
        del list1[-1]
        del list1[-1]
        del list2[-1]
        del list2[-1]
        #print 'list1:',list1
        #veclist1 = np.array(list1)  
        #veclist2 = np.array(list2) 
        vector_a = np.mat(list1)
        #print vector_a
        vector_b = np.mat(list2)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        
        classA=set(vector1[-2])   #先是tfidf 然后是class 最后是tagseq
        classB=set(vector2[-2])
        #print classA
        
        diffSeqCnt=1.0 - sim
        diffClassCnt=self.jaccard_similarity(classA, classB)  
        #print('----------begin-------------------------')
        diffCnt=0.2*diffClassCnt+0.8*diffSeqCnt
        #print('----------end-------------------------')
    
        #print diffCnt
        return diffCnt
    '''
    def calculateVectorDiff(self, vector1, vector2, splitSym = ',', isPercent = True):
        #
        list1=copy.copy(vector1)
        list2=copy.copy(vector2)
        
        del list1[-1]
        del list1[-1]
        
        del list2[-1]
        del list2[-1]
        #print 'list1:',list1
        a=[]
        veclist1 = np.array(list1)  
        veclist2 = np.array(list2) 
        #a.append(veclist1)
        #a.append(veclist2)
        #vector_a = np.mat(veclist1)
        #vector_b = np.mat(veclist2)
        #num = (vector_a * vector_b.T)
        #denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        #cos = float(num.real) / float(denom.real)
        ##sim = 1.0-(0.5 + 0.5 * cos)
        classA=set(vector1[-2])   #先是tfidf 然后是class 最后是tagseq
        classB=set(vector2[-2])
        
        diffSeqCnt=np.sum(np.abs(veclist1-veclist2))

        diffClassCnt=self.jaccard_similarity(classA, classB)  

        diffCnt=0.8*(float(diffSeqCnt)/12.0)+0.2*(diffClassCnt)

        
            
        return diffCnt
        
    #calculate the max distance of pair in one group    
    def calculateMaxDistance(self, vecGroup, isPercent):
        maxDis = 0
        
        for i in range(len(vecGroup)):
            vector1 = vecGroup[i]
            
            for j in range(i + 1, len(vecGroup)):
                vector2 = vecGroup[j]
            
                dis = self.calculateVectorDiff(vector1, vector2, isPercent=isPercent)
                
                if dis > maxDis:
                    maxDis = dis
        
        return maxDis
    
    #find the max distrance between poinis in group and centre point
    def calculateMaxCentreDistance(self, centrePoint, vecGroup, isPercent):
        maxDis = 0
        
        vector1 = centrePoint
        
        for j in range(0, len(vecGroup)):
            vector2 = vecGroup[j]
        
            dis = self.calculateVectorDiff(vector1, vector2, isPercent=isPercent)
            
            if dis > maxDis:
                maxDis = dis
        
        return maxDis
    
    #find centre point in one group
    def findBestCenter(self, vecGroup, isPercent):
        
        minToken = None
        minDistance = 0
        
        for i in range(len(vecGroup)):
            sum = 0
            vector1 = vecGroup[i]
            
            for j in range(len(vecGroup)):
                if (i != j):
                    vector2 = vecGroup[j]
                    
                    sum += self.calculateVectorDiff(vector1, vector2, isPercent=isPercent)
            
            #find better one or first one
            if sum < minDistance or minDistance == 0:
                minDistance = sum
                minToken = vecGroup[i]
                   
        return  (minToken, minDistance)
            
    def chainCluster(self, trainingset, threshold):
        startGroup = []
        groupList = []
        cnt = 0
        
        if threshold > 1:
            isPercent = False
        else:
            isPercent = True
        
        dataset = list(trainingset)
        
        while len(dataset) != 0:
            removeMask = [0] * len(dataset)
            
            #random select one point
            if not startGroup:
                #print("dataset length is %d" % len(dataset))
                #print('select new point from dataset')
                i = random.randrange(0, len(dataset))
                startPoint = dataset[i]
                
                #group = set()
                group=[]
                
                isNewGroup = True
                removeMask[i] = 1
                
                #for debug
                '''
                if cnt == 1:
                    pprint(len(groupList))
                    raise Exception("Debug here")
                cnt += 1
                '''
            else:
                #here we consider it's a queue, each time select the first point
                #print('select new point from start group')
                startPoint = startGroup[0]
                
                group = groupList[-1]
                
                isNewGroup = False
                del startGroup[0]
            
            for j, point in enumerate(dataset):
                diff = self.calculateVectorDiff(startPoint, point, isPercent=isPercent)
                
                if diff <= threshold:
                    #group.add(point)
                    group.append(point)
                    removeMask[j] = 1
                    
                    #if distance is not zero, add it into start group
                    if diff > 0:
                        startGroup.append(point)
                
            if isNewGroup:
                groupList.append(group)
            
            #update the dataset, and remove the useless elements
            dataNew = [dataset[i] for i in range(len(dataset)) if removeMask[i] == 0]
            dataset = dataNew
        
        #conver all the set to list
        groupList = list(map(list, groupList))
        
        return groupList
    
    def compactCluster(self, trainingset, threshold):
    
        centreSet = []
        groupPList = []
        n = 0
        
        if threshold > 1:
            isPercent = False
        else:
            isPercent = True
        
        while True:
        
            dataset = list(trainingset)
            groupList = []
            centreTmp = []
            
            #make a copy of centreSet
            centreCopy = list(centreSet)
                    
            while len(dataset) != 0:
                
                if len(centreCopy) == 0:
                    index = random.randrange(0, len(dataset))
                    centreElem = dataset[index]
                else:
                    index = random.randrange(0, len(centreCopy))
                    centreElem = centreCopy[index]
                    centreCopy.remove(centreElem)
                    
                group = []
                removeMask = [0] * len(dataset)
                
                for i in range(len(dataset)):
                    
                    checkElem = dataset[i]
                    
                    vector1 = centreElem
                    vector2 = checkElem
                    
                    #caluclate distance
                    diff = self.calculateVectorDiff(vector1, vector2, isPercent=isPercent)
                    
                    #find a qualified point
                    if diff <= threshold:
                        group.append(checkElem)
                        
                        #remove element from centreSet
                        if checkElem in centreCopy:
                            #print('Remove one element %s from centreSet' % checkElem)
                            centreCopy.remove(checkElem)
                        
                        removeMask[i] = 1
                
                groupList.append(group)
                
                #find new centre point
                token, distance = self.findBestCenter(group, isPercent=isPercent)
                
                centreTmp.append(token)
                
                #print(len(dataset))
                #print(len(removeMask))
                
                dataNew = [dataset[i] for i in range(len(dataset)) if removeMask[i] == 0]
                
                dataset = dataNew
            
            #print("Group Number change from %d to %d" % (n, len(groupList)))
            
            if len(groupList) >= n and n != 0:
                break
            else:
                n = len(groupList)
                #update centreSet
                centreSet = centreTmp
                
                #update groupList
                groupPList = list(groupList)
                
        return groupPList, centreSet
    
    #genererate fold-N data
    def pageClusterGenMLData(self, N, fnin):
        
        vecList = self.fetchVectorList(fnin)
        
        print(len(vecList))
        
        mlData = self.rg.generateNFold(N, vecList)
        
        print("Finish ML Data Generated!")
        
        return mlData
    
    #do one fold-N testing
    def pageClusterML(self, N, fnnin, ffnin, fnout, threshold = 3, cType='H'):
        mlData = self.pageClusterGenMLData(N, ffnin)
        dupRes = []
        accRes = []
        fpRes = []
        falseCase = defaultdict(set)
        res = {}
        
        summary = mlData['summary']
        details = mlData['details']
        
        N = summary['foldnum']
        
        trainingsetList = details['training']
        testsetList = details['test']
        
        testsetN = self.fetchVectorListN(fnnin)
        
        for i in range(N):
            trainingset = trainingsetList[i]
            testsetF = testsetList[i]
            
            #hierarchy cluster
            if cType == 'H':
                resGroup, centreSet = self.compactCluster(trainingset, threshold)
                
                hitNum = 0
                for testdata in testsetF:
                    if self.isHitHRange(testdata, centreSet, threshold):
                    #if self.isFPDiff(testdata, centreSet, threshold):    
                        hitNum += 1
                
                accuracy = float(hitNum) * 100 / float(len(testsetF))
                
                print("Group %d accuracy result is %.2f" % (i, accuracy))
                
                hitNum = 0
                for testdata in testsetN:
                    if self.isHitHRange(testdata, centreSet, threshold):
                    #if self.isFPDiff(testdata, centreSet, threshold):
                        hitNum += 1
                
                falseP = float(hitNum) * 100 / float(len(testsetN))
                b=hitNum
                
                print("Group %d false positive result is %.2f" % (i, falseP))
                    
            #chain cluster                
            elif cType == 'C':
                resGroup = self.chainCluster(trainingset, threshold)
                hitNum = 0
                for testdata in testsetF:
                    hitRes, group = self.isHitCRange(testdata, resGroup, threshold)
                    
                    if hitRes:
                        hitNum += 1
                
                accuracy = float(float(hitNum) * 100 / float(len(testsetF)))
                a=hitNum
                
                print("Group %d accuracy result is %.2f" % (i, accuracy))
                
                hitNum = 0
                for testdata in testsetN:
                    hitRes, group = self.isHitCRange(testdata, resGroup, threshold)
                    
                    if hitRes:
                        hitNum += 1
                        #falseCase[testdata] = falseCase[testdata].union(set(group))
                        '''
                        print('find one false normal case')
                        pprint(testdata)
                        try:
                            falseCase[testdata] = falseCase[testdata].union(set(group))
                        except Exception as e:
                            print(e)
                            raise(e)
                        '''
                falseP = float(float(hitNum) * 100 / float(len(testsetN)))
                print("Group %d false positive result is %.2f" % (i, falseP))
                b=hitNum
                
                
                #pre=float(a) * 100/float(a+b)
                #print("Group %d false precision result is %.2f" % (i, pre))
            else:
                raise Exception("Unsupported type %s" % cType)
                #pprint(falseCase)
            print ('len(resGroup):',len(resGroup))
            #duplicate =  (1 - len(resGroup) / len(trainingset)) * 100
            duplicate =  (1.0 - float(len(resGroup)) / float(len(trainingset))) * 100
            print("Group %d duplicate result is %.2f" % (i, duplicate))
            
            accRes.append(accuracy)
            dupRes.append(duplicate)
            fpRes.append(falseP)
        #pprint(accRes)
        #pprint(fpRes)
        #pprint(dupRes)
        
        res['accuracy'] = accRes
        res['falsepositive'] = fpRes
        res['duplicate'] = dupRes
        #res['falsecase'] = falseCase
                
        with open(fnout, 'w') as fout:
        
            acc = res['accuracy']
            fp = res['falsepositive']
            dup = res['duplicate']
            
            fout.write("<duplicates>\t<accuracy>\t<falsepositive>\n")
            for j in range(len(acc)):
                fout.write("group%d:\t%f\t%f\t%f\n" % (j, dup[j],acc[j], fp[j]))
            fout.write("---------------------\n")
        
            #fcase = res['falsecase']
            #fout.write("False Positive Case Details {<false vector>:<hit cluster>, ...}\n")
            #pprint(fcase, fout)
            fout.write("---------------------\n")
    
    #do chain clustering from file
    def pageChainClusterFromData(self, fnin, fnout, threshold = 0.4):
        dataset = self.fetchVectorList(fnin)
        
        groupList = self.chainCluster(dataset, threshold)
        
        if threshold > 1:
            isPercent = False
        else:
            isPercent = True
        
        print('Get %d cluster' % len(groupList))
        
        groupList = sorted(groupList, key=lambda d:len(d), reverse = True)
        
        with open(fnout, 'w', encoding='utf-8') as fout:
            pprint(groupList, fout)
    
    #do hierarchy clustering from file
    def pageCompactClusterFromData(self, fnin, fnout, threshold = 3):
        
        pageReport = None
        with open(fnin, 'r') as fin:
            pageReport = eval(fin.read())
        
        vecList = self.fetchVectorList(fnin)
        
        #hierarchy clustering
        groupList, centreSet = self.compactCluster(vecList, threshold)
        
        print('Get %d cluster' % len(groupList))        
        
        groupList = sorted(groupList, key=lambda d:len(d), reverse = True)
        
        with open(fnout, 'w') as fout:
            pprint(groupList, fout)
    #----------------------------------------#
    
    #report functions
    #----------------------------------------#
    #give page general information report
    def analysePageStrcture(self, fnin, fnout):
        pageData = None
        with open(fnin, 'r', encoding='utf-8') as fin:
            pageData = eval(fin.read())
        
        summary = {'sequence':{}, 'vector':{}, 'hash':{}}
        seqDetails = {}
        vecDetails = {}
        hashDetails = {}
        
        res = {}
        totalNum = 0
        
        for file,pageInfo in pageData.items():
            totalNum += 1
            url = pageInfo['url']
            
            content = pageInfo['tagseq']
            vector = pageInfo['tagvec']
            hash = pageInfo['hash']
            title = pageInfo['title']
            
            #convert vector to string
            vector = [str(item) for item in vector]
            vector = ",".join(vector)
            
            #analyse hash data
            if hash in summary['hash'] :
                summary['hash'][hash] += 1
                hashDetails[hash].append([url, file, title])
            else:
                summary['hash'][hash] = 1
                hashDetails[hash] = [[url, file, title]]
            
            #analyse sequence data
            if content in summary['sequence'] :
                summary['sequence'][content] += 1
                seqDetails[content].append([url, file, title])
            else:
                summary['sequence'][content] = 1
                seqDetails[content] = [[url, file, title]]
            
            #analyse vector data
            if vector in summary['vector'] :
                summary['vector'][vector] += 1
                vecDetails[vector].append([url, file, title])
            else:
                summary['vector'][vector] = 1
                vecDetails[vector] = [[url, file, title]]
        
        summary['sequence'] = sorted(summary['sequence'].items(), key=lambda d:d[1], reverse = True)
        
        summary['vector'] = sorted(summary['vector'].items(), key=lambda d:d[1], reverse = True)
        
        summary['hash'] = sorted(summary['hash'].items(), key=lambda d:d[1], reverse = True)
        
        res['summary'] = summary
        res['vecdetails'] = vecDetails
        res['seqdetails'] = seqDetails
        res['hashdetails'] = hashDetails
                
        res['summary']['totalpagenum'] = totalNum
        res['summary']['totalseqnum'] = len(summary['sequence'])
        res['summary']['totalvecnum'] = len(summary['vector'])
        res['summary']['totalseqnum'] = len(summary['sequence'])
        res['summary']['totalhashnum'] = len(summary['hash'])
        
        with open(fnout, 'w', encoding='utf-8') as fout:
            pprint(res, fout)
    #----------------------------------------#        
       
            
if __name__ == '__main__':
    
    taskType = ""
    fnnin = ""
    ffnin = ""
    fnin = ""
    fnout = ""
    distance = 0
    ipFile = None
    N = 0
    
    errorFlag = False
    
    supportedType = {3:['hostdup', 'ipdup', 'report'], 4:['ipdup', 'cluster'], 6:['nfold']}
    
    if len(sys.argv) == 4:
        taskType = sys.argv[1]
        fnin = sys.argv[2]
        fnout = sys.argv[3]
        
        if not taskType in supportedType[3]:
            errorFlag = True
            
    elif len(sys.argv) == 5:
        taskType = sys.argv[1]
        fnin = sys.argv[2]
        fnout = sys.argv[3]
        
        if taskType == 'ipdup':
            ipFile = sys.argv[4]
        elif taskType == 'cluster':
            distance = float(sys.argv[4])
        
        if not taskType in supportedType[4]:
            errorFlag = True
    
    elif len(sys.argv) == 7:
        taskType = sys.argv[1]
        N = int(sys.argv[2])
        fnnin = sys.argv[3]
        ffnin = sys.argv[4]
        fnout = sys.argv[5]
        distance = float(sys.argv[6])
        
        if not taskType in supportedType[6] or N <= 0:
            errorFlag = True
    else:
        errorFlag = True
    
    if errorFlag:
        print("Usage: \n \
        \t python FingerPrintCluster.py hostdup <fin> <fout> \n \
        \t python FingerPrintCluster.py ipdup <fin> <fout> [<ip file>] \n \
        \t python FingerPrintCluster.py report <fin> <fout> \n \
        \t python FingerPrintCluster.py nfold <normal info> <phishing info> <fout> <distance threshold> <N> \n \
        \t python FingerPrintCluster.py cluster <fin> <fout> <distance threshold>")
        exit()
    
    htmlCA = FingerPrintCluster(ipFile)
    
    if taskType == 'hostdup':
        #first generate report
        htmlCA.analysePageStrcture(fnin,'./tmp/%s_report_tmp' % fnin)
        
        #find duplicates
        htmlCA.findHashDupByHost('./tmp/%s_report_tmp' % fnin, './tmp/%s_noduplicate_host_tmp' % fnin)
        
        #remove from original file
        htmlCA.removeDuplication(fnin, './tmp/%s_noduplicate_host_tmp' % fnin, fnout, 'keep')
    
    elif taskType == 'ipdup':
        #first generate report
        htmlCA.analysePageStrcture(fnin,'./tmp/%s_report_tmp' % fnin)
        
        #find duplicates
        htmlCA.findHashDupByIP('./tmp/%s_report_tmp' % fnin, './tmp/%s_noduplicate_ip_tmp' % fnin)
        
        #remove from original file
        htmlCA.removeDuplication(fnin, './tmp/%s_noduplicate_ip_tmp' % fnin, fnout, 'keep')
    
    elif taskType == 'report':
        htmlCA.analysePageStrcture(fnin, fnout)
    
    elif taskType == 'cluster':
        htmlCA.pageChainClusterFromData(fnin, fnout, distance)
    
    elif taskType == 'nfold':
        htmlCA.pageClusterML(N, fnnin, ffnin, fnout, distance)
    #plt.show()
