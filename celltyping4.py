# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:22:20 2023

@author: youm
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm


TQ = False
NITERS = 1000
L4 = torch.tensor([np.sqrt(1/(np.sqrt(2*np.pi))),1,.5])
L5 = torch.tensor([np.sqrt(1/(np.sqrt(2*np.pi))),-1,.5])
ca,cb,cc = False, True, True

def main(df,obs,dfxy):
    global L4, L5, ca,cb,cc

    df = df.apply(zscore)
    #print(df)
    for i in range(5):
        #i = -4
        #for i in range(df.shape[1]):
        cn = df.columns[i]
        ser = df.iloc[:,i]
        hist = makeHist(ser)
        hist = normalize(hist)




        a = np.sqrt(1/(np.sqrt(2*np.pi)))
        ga = torch.tensor([a,a],requires_grad=True)
        gb = torch.tensor([-1.,1.],requires_grad=True)
        gc = torch.tensor([.5,.5],requires_grad=True)
        gaus = [ga,gb,gc]
        optim = torch.optim.SGD(gaus, lr=1e-6)#, momentum=0.1)

        #print(gaus)
        loss = 0
        if TQ:
            for i in tqdm(range(1000),cn):
                loss = getLoss(gaus,hist)
                loss.backward()
                optim.step()
        else:
            for i in range(NITERS):
                if i % 200 == 0 :
                    showGaus(gaus,data=hist,cn=cn+' '+str(i))
                    for gau in gaus:
                        try:
                            print(gau.detach().numpy())
                        except:
                            print([float(gau[0].detach()),float(gau[1].detach())])
                    print(loss,"\n")
                loss = getLoss(gaus,hist)
                loss.backward()
                optim.step()
                '''
                if i == int(NITERS/3) and False:
                    L4[1] = float(gaus[1][0].detach())
                    L5[1] = float(gaus[1][1].detach())
                    gaus = [ga,gaus[1],gc]
                    cb = False
                    cc = True
                    print("calculating c")
                '''
                if i == 2*int(NITERS/3):
                    L4[1] = float(gaus[1][0].detach())
                    L5[1] = float(gaus[1][1].detach())
                    L4[2] = float(gaus[2][0].detach())
                    L5[2] = float(gaus[2][1].detach())
                    ca = True
                    cb = False
                    cc = False
                    gaus = [ga,gaus[1],gaus[2]]
                    print("calculating a")
        showGaus(gaus,data=hist,cn=cn+' final')
        print(gaus)
        print(loss,cn,"\n\n")



def getLoss(gaus,data):
    #a,b,c = gaus[0,0],gaus[0,1],gaus[0,2]
    #A,B,C = gaus[1,0],gaus[1,1],gaus[1,2]

    #a,b,c = torch.max(torch.tensor([gaus[0,0],0],requires_grad=True))     ,gaus[0,1],gaus[0,2]
    #A,B,C = torch.max(torch.tensor([gaus[1,0],0],requires_grad=True))     ,gaus[1,1],gaus[1,2]

    '''
    bs = gaus[:,1]
    mb,Mb = min([b,B]),max([b,B])
    print(mb,Mb)
    mr,Mr = mb-1,Mb+1
    '''
    l1 = [ca,cb,cc]
    for i, cal in enumerate(l1):
        if not cal:
            gaus[i] = [L4[i],L5[i]]

    loss = 0
    for i in range(data.shape[0]):
        x = data.loc[:,'intensity'].iloc[i]
        y = data.loc[:,'count'].iloc[i]
        #g = a*torch.exp(-(x-b)**2/(2*c)**2) + A*torch.exp(-(x-B)**2/(2*C)**2)
        #gap = (y - g) ** 8
        #loss += ((.5-A)**2 + (.5-a)**2 + 1) * gap
        #loss -= torch.min(torch.tensor([a,A,0],requires_grad=True))
        g = calcGaus(gaus,x)
        loss += ((y - g) ** 2)
    #loss = loss * ((10-(4*A)**3)**2 + (10-(4*a)**3)**2 + 1)


    #loss -= torch.min(torch.tensor([a,A,0],requires_grad=True))
    return(loss)





def normalize(df):
    x,y = df.iloc[:,0].values,df.iloc[:,1].values
    area = np.trapz(y,x)
    df.iloc[:,1] = df.iloc[:,1] / area
    return(df)


def normalize1(ser):
    mi,ma = ser.min(),ser.max()
    ser = (ser - mi) / (ma - mi)
    su = ser.sum()
    return(ser/su)



def makeHist(series):
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
    return(hist)




def showGaus(gaus,mi=-20,ma=20,data=None,cn=''):
    #gaus = gaus.detach().numpy()
    fig,ax = plt.subplots()
    if type(data) != type(None):
        mi,ma = data["intensity"].min(),data["intensity"].max()
        ax.plot(data["intensity"],data["count"])

    #print(mi,ma,"min max")

    #a,b,c = np.max([gaus[0,0],0]),gaus[0,1],gaus[0,2]
    #A,B,C = np.max([gaus[1,0],0]),gaus[1,1],gaus[1,2]
    xs,ys = [],[]

    for x in np.linspace(mi,ma,101):
        xs.append(x)
        y = calcGaus(gaus,x)
        #y = a*np.exp(-(x-b)**2/(2*c)**2) + (1-a)*np.exp(-(x-B)**2/(2*C)**2)
        #for if area under hists sums to 1
        ys.append(y.detach())

    ax.plot(xs,ys)
    ax.set_title(cn)
    plt.show()


def calcGaus(gaus,x):
    #l4 and l5 are globals
    a,A = gaus[0][0],gaus[0][1]
    b,B = gaus[1][0],gaus[1][1]
    c,C = gaus[2][0],gaus[2][1]

    #print(l1,l2,l3)

    #y = a*torch.exp(-(x-b)**2/(2*c)**2) + A*torch.exp(-(x-B)**2/(2*C)**2)
    y = a**2*torch.exp(-(x-b)**2/(2*c)**2) + A**2*torch.exp(-(x-B)**2/(2*C)**2)
    #for i in range(10**8):
        #j = i + 1
    #y = a**2*torch.exp(-(x-b)**2/(2*c)**2) + A**2*torch.exp(-(x-B**2)**2/(2*C)**2)

    return(y)

def calcGaus1(gaus,x):
    a,b,c = gaus[0,0],gaus[0,1],gaus[0,2]
    A,B,C = gaus[1,0],gaus[1,1],gaus[1,2]
    #y = a**2*torch.exp(-(x-b)**2/(2*c)**2) + A**2*torch.exp(-(x-B**2)**2/(2*C)**2)
    #has square a square 2nd B to constrain positive
    #y = a*torch.exp(-(x-b)**2/(2*c)**2) + A*torch.exp(-(x-B)**2/(2*C)**2)
    #y = a**2*torch.exp(-(x-b)**2/(2*c)**2) + A**2*torch.exp(-(x-B)**2/(2*C)**2)
    fa = np.sqrt(1/(np.sqrt(2*np.pi)))
    y = fa**2*torch.exp(-(x-b)**2/(2*c)**2) + fa**2*torch.exp(-(x-B)**2/(2*C)**2)
    return(y)


if __name__ == "__main__":
    folder = r'C:\Users\youm\Desktop\src\BR MFC7 GL data pre 230808 pre vietnam'#r"C:\Users\youm\Desktop\src\zzzzzzzzzzz_current/"
    stem = '196_MCF7'#'95_GL'#'95_GL'#'z3_GL631'#'zzz_hta14'#'hta14bx1_ck7'#'recent june 3 23/PIPELINE_hta14_bx1_with_svm'#'PIPELINE_94'#'PIPELINE_hta14_bx1_99'#'93_hta14_no_neigh'#'93_hta14'#87_LC-4'##'89_LC-4_withN'#''96_LC'#cl56_depth_study_H12'#'96_LC'#'97_mtma2'#'93_hta14'###'96_hta14_primary'#'97_hta14bx1_primary_celltype'#'99_hta14'#"temp"#"zzz_hta1499"#"zzz14bx1_97"#"hta14bx1 dgram"#folder+"14_both"##"tempHta14_200"#"HTA14f"#"zzzz_hta1498_neighborhoodsOnly"#"hta1415Baf1"#"HTA15f"#"0086 HTA14+15"#"99HTA14"#"z99_ROIs_5bx_HTA1415"#"temp"#"z99_ROIs_5bx_HTA1415"#<-this one has old celltyping no TN #"0084 HTA14+15" #"HTA9-14Bx1-7 only"#"0.93 TNP-TMA-28"#"0.94.2 TNP-TMA-28 primaries"#"1111 96 TNP-28" #'0093 HTA14+15'#"0094.7 manthreshsub primaries HTA14+15"#"0094 HTA14+15" #"096 2021-11-21 px only" #'095.08 primaries only manthreshsub 2021-11-21 px only'#"094 manthreshsub 2021-11-21 px only" #  '095.1 primaries only manthreshsub 2021-11-21 px only'#
    print("axis labels %s on barplots more ticks/lines")
    stem = folder + "/" + stem
    print(stem)
    df = pd.read_csv(stem+"_df.csv",index_col=0)
    obs = pd.read_csv(stem+"_obs.csv",index_col=0).astype(str)
    dfxy = pd.read_csv(stem+"_dfxy.csv",index_col=0)
    main(df,obs,dfxy)



'''
    print(hist)
    hist['count'] = softmax(hist['count'])
    print(hist)

def softmax(x):
    mx = x.max()
    exp_x = np.exp(x-mx)
    ts = exp_x.sum()
    val = exp_x / ts
    return(val)



def softmax1(x):
    mx = torch.max(x, -1,keepdim=True)[0]
    #print(x.shape,mx.shape)
    exp_x = torch.exp(x - mx)
    ts = torch.sum(exp_x, -1,keepdim=True)
    #print(exp_x.shape,ts.shape,"\n")
    val = exp_x / ts
    return val

'''