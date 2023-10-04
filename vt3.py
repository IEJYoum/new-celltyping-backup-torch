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


NITERS = 4000
STROM_W = -.05
CONF = .6

def main(df,obs,lw,dfxy=None):
    #print(lw)
    print(df.shape)
    df = df.apply(zscore)
    #lw.iloc[:,-1] += STROM_W
    #print(lw)
    bT,wT = makeBT(df,lw)

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
    tT = tT ** 2
    ts = torch.sum(tT,axis=1,keepdim=True)
    #print(ts,ts.shape)
    tT = tT/ts
    print('\n',tT)
    tdf = pd.DataFrame(tT.detach(),columns = lw.columns,index = df.index)
    tyMx = tdf.max(axis = 1)
    ukey = tyMx < CONF
    tyS = tdf.idxmax(axis=1)
    tyS.loc[ukey] = "6 Unidentified"
    tdf.to_csv("predicted celltypes.csv")
    obs['vector celltype_1'] = tyS
    obs.to_csv("tiny_pTMA1_obs.csv")

def makeBT(df,lw):
    bT = []
    wT = []
    for biom in df.columns:
        b = torch.tensor(df.loc[:,biom].values.reshape(-1,1))
        bT.append(b)
        we = torch.tensor(lw.loc[biom,:].values.reshape(1,-1))
        wT.append(we)
    return(bT,wT)

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
    print("axis labels %s on barplots more ticks/lines")
    #stem = folder + "/" + stem
    print(stem)
    df = pd.read_csv(stem+"_df.csv",index_col=0)
    obs = pd.read_csv(stem+"_obs.csv",index_col=0).astype(str)
    #dfxy = pd.read_csv(stem+"_dfxy.csv",index_col=0)
    dfxy = None
    lw = pd.read_csv("loss_weights.csv",index_col=0)
    main(df,obs,lw,dfxy)