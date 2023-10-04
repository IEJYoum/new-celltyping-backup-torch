# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:51:43 2023

@author: youm
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm
import os
import seaborn as sns
import allcolors as allc


NITERS = 10**8
MIN = 10


def main(df,obs,dfxy=None,sep="ask"):
    if sep == "ask":
        ch,uch = obMenu(obs,"category whose subcategories will be individually annotated")
        sep = obs.columns[ch]

    obs["orthoThresh phenotype " + sep] = ''
    for i,uo in enumerate(sorted(list(obs.loc[:,sep].unique()))):

        print(uo)
        key = obs.loc[:,sep] == uo
        tdf = df.loc[key,:].apply(zscore)
        tobs = obs.loc[key,:]

        cm = corMat(tdf)
        #showHeatmap(cm)
        #getThresh(cm,df,obs)
        thresho = getThresh(cm,tdf,tobs,t2 = uo)
        for i,biom in enumerate(df.columns):
            keyt = df.loc[:,biom] > thresho[i]
            tobs.loc[keyt,"orthoThresh phenotype " + sep] += biom
        obs.loc[key,:] = tobs
    return(df,obs,dfxy)



def obMenu(obs,title="choose category:"):
    for i,col in enumerate(obs.columns):
        print(i,col)
    ch = int(input(title)) #multiboxplot needs this to trigger an error if non int sent
    uch = sorted(list(obs[obs.columns[ch]].unique()))
    return(ch,uch)


def getThresh(cm,df,obs,t2=''):
    antiS = cm.idxmin(axis=1)
    thresho = []
    for i,b in enumerate(df.columns):
        abn = antiS.iloc[i]
        switch = 0
        badS = ["R0","DAPI","R5","R6","R7"]
        for bs in badS:
            if bs in b:
                switch = 1
        if switch == 1:
            thresho.append(9999)
            continue
        biom = torch.tensor(df.loc[:,b])
        #makeHist(biom,title=b)
        abiom = torch.tensor(df.loc[:,antiS.iloc[i]])



        thresh = torch.tensor(1.0,requires_grad = True)
        optim = torch.optim.SGD([thresh], lr=1e-7)
        lloss = 999999999#torch.tensor(999999999)
        loss = 0
        for i in range(NITERS):
            sbiom = (biom - thresh) * 10
            sbiom = 2 * torch.sigmoid(sbiom) - 1
            pos = 2*( torch.sigmoid((sbiom+1) * 10) - .5)
            neg = 2*( torch.sigmoid((sbiom-1) * -10) - .5)
            ABB = (neg * abiom).sum()
            ABA = (pos * abiom).sum()
            BEA = (pos * biom).sum()
            BEB = (neg * biom).sum()
            NC = neg.sum()
            PC = pos.sum()
            loss = (ABA**3 + BEB+ (NC+PC)**2)/((ABB+BEA)*(NC*PC+1)**.5)
            #loss = (ABA + BEB+ (NC+PC)**2)/((ABB+BEA)*NC*PC)
            #loss = (ABA + BEB)/(ABB+BEA)
            #loss = (ABA + BEB)/(ABB+BEA)
            #loss = (ABA + BEB+ (NC+PC)**2)/((ABB+BEA)*NC*PC) #lr = 1 works but is too heavily drawn towards median
            #loss = (ABA + BEB+ 10000000)/((ABB+BEA)*NC*PC)
            #makeHist(pos,title=b)
            if i % 5000 == 4999:
                try:
                    print(float(loss.detach()),float(lloss.detach()))
                    print(lloss/loss)
                    print(lloss/loss <  1 + 10 ** -9)
                except:
                    print(float(loss.detach()),lloss)

            if lloss > loss or i < MIN:
                #print(float((lloss/loss).detach()))
                lloss = loss
                loss.backward()
                optim.step()
            else:
                #print(float((lloss/loss).detach()))
                thr = float(thresh.detach())
                thresho.append(thr)
                print(float(BEB.detach()),"BEB")
                print(round(float((lloss/loss).detach()),3),"   lloss/loss", lloss>loss,round(float((lloss-loss).detach()),3),float(lloss.detach())>float(loss.detach()))
                print(float(thresh.detach()),b,i,"done!\n")

                makeHist(biom,title=b,vline=thr,t2=t2)
                scatterplot(df,obs,b,abn,vline = thr,t2=t2)
                break

            #break
    return(thresho)



def scatterplot(df,obs,biom,abiom,vline=None,t2=''): #biom and abiom are names, not series (as they are in loss fn)
        fig = plt.figure()
        ax = fig.add_subplot()
        colors = allc.colors
        for i,uo in enumerate(sorted(list(obs.loc[:,"slide_scene"].unique()))):
            key = obs.loc[:,"slide_scene"] == uo
            #print(key)
            color = colors[i]
            ax.plot(df.loc[key,biom],df.loc[key,abiom], color=color, marker='x', linestyle='none', markersize=.25)
            if type(vline) != type(None):
                ax.vlines(x=vline,ymin=min(df.loc[key,abiom]),ymax=max(df.loc[key,abiom]),color="b")
        ax.set_xlabel(biom)
        ax.set_ylabel(abiom)
        ax.set_title(t2)
        plt.show()



def makeHist(series,title='',vline=None,t2=''):
    try:
        series = pd.Series(series.detach().numpy())
    except Exception as e:
        print(e)
    nbin = int(series.shape[0]/100)
    mi,ma = series.min(),series.max()
    intensity = []
    count = []

    for i in np.linspace(mi,ma,nbin):
        intensity.append(i)
        key = series <= i
        count.append(key.sum())
        series = series.loc[~key]
        #print(series.shape)
    hist = pd.DataFrame([intensity,count]).transpose()
    hist.columns = ["intensity","count"]
    fig,ax = plt.subplots()
    ax.plot(hist["intensity"],hist["count"])
    if type(vline) != type(None):
        print("VL",vline)
        ax.vlines(x=vline,ymin=0,ymax=max(count),color="r")
    ax.set_title(title+' - '+t2)
    plt.show()

    return(hist)

def showHeatmap(cdf):
    f, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cdf,xticklabels=cdf.columns,yticklabels=cdf.index,center=np.mean(cdf.values))
    plt.show()


def corMat(df):
    cm = []
    for b1 in df.columns:
        cl = []
        for b2 in df.columns:
            cl.append((df.loc[:,b1] * df.loc[:,b2]).sum()/df.shape[0])
        cm.append(cl)
    return(pd.DataFrame(cm,columns=df.columns,index = df.columns))


def getLoss(bT,wT,tT): #,dcols
    tT = tT ** 2
    ts = torch.sum(tT,axis=1,keepdim=True)
    #print(ts,ts.shape)
    tT = tT/ts
    #print('\n',tT)


    loss = 0
    for i,bSer in enumerate(bT):
        we = wT[i]
        if we.sum() == 0:
            continue
        lmat = np.matmul(bSer,we)
        #print(lmat)
        ltot = lmat * tT
        #print(ltot.shape)
        loss += ltot.sum()
    return(loss)



if __name__ == "__main__":
    #folder = r'C:\Users\youm\Desktop\src\BR MFC7 GL data pre 230808 pre vietnam'#r"C:\Users\youm\Desktop\src\zzzzzzzzzzz_current/"
    stem = 'tiny_pTMA1' #'196_MCF7'#'95_GL'#'95_GL'#'z3_GL631'#'zzz_hta14'#'hta14bx1_ck7'#'recent june 3 23/PIPELINE_hta14_bx1_with_svm'#'PIPELINE_94'#'PIPELINE_hta14_bx1_99'#'93_hta14_no_neigh'#'93_hta14'#87_LC-4'##'89_LC-4_withN'#''96_LC'#cl56_depth_study_H12'#'96_LC'#'97_mtma2'#'93_hta14'###'96_hta14_primary'#'97_hta14bx1_primary_celltype'#'99_hta14'#"temp"#"zzz_hta1499"#"zzz14bx1_97"#"hta14bx1 dgram"#folder+"14_both"##"tempHta14_200"#"HTA14f"#"zzzz_hta1498_neighborhoodsOnly"#"hta1415Baf1"#"HTA15f"#"0086 HTA14+15"#"99HTA14"#"z99_ROIs_5bx_HTA1415"#"temp"#"z99_ROIs_5bx_HTA1415"#<-this one has old celltyping no TN #"0084 HTA14+15" #"HTA9-14Bx1-7 only"#"0.93 TNP-TMA-28"#"0.94.2 TNP-TMA-28 primaries"#"1111 96 TNP-28" #'0093 HTA14+15'#"0094.7 manthreshsub primaries HTA14+15"#"0094 HTA14+15" #"096 2021-11-21 px only" #'095.08 primaries only manthreshsub 2021-11-21 px only'#"094 manthreshsub 2021-11-21 px only" #  '095.1 primaries only manthreshsub 2021-11-21 px only'#
    print("trim outliers before calculating orthos?")
    #stem = folder + "/" + stem
    print(stem)
    df = pd.read_csv(stem+"_df.csv",index_col=0)
    obs = pd.read_csv(stem+"_obs.csv",index_col=0).astype(str)
    #dfxy = pd.read_csv(stem+"_dfxy.csv",index_col=0)
    dfxy = None
    #lw = pd.read_csv("loss_weights.csv",index_col=0)
    main(df,obs,dfxy)




    '''
    tT = np.ones((df.shape[0],5))
    print(tT)
    tT = torch.tensor(tT,requires_grad=True)

    optim = torch.optim.SGD([tT], lr=1e-4)
    #for i in tqdm(range(NITERS)):
    for i in range(NITERS):
        loss = getLoss(bT,wT,tT) #,df.columns
        print(loss)
        loss.backward()
        optim.step()

    obs.to_csv("tiny_pTMA1_obs.csv")
    '''