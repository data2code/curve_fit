#!/usr/bin/env python
import sys
import re
import pandas as pd
import numpy as np
#import util as util
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

COLORS=[
    "#4572A7", "#AA4643", "#88A44D", "#6E548D", "#3D96AE", "#DB843D", "#8EA5CB", "#D19392", "#B9CD96", "#A99BBD",
    "#0000FF", "#FF00FF", "#008000", "#FF0000", "#D6954F", "#8080FF", "#FFC800", "#404040", "#00FFFF", "#80FF00",
    "#C0C0C0", "#000000", "#330066", "#0099CC", "#FF99CC", "#FF3366", "#9999CC", "#996633", "#FF9933", "#CC6699",
    "#669966", "#99CCCC", "#993366", "#6666CC", "#CCCC66", "#99CC66", "#FF6666", "#0066CC", "#FFFF00", "#663366",
    "#0A0A0A", "#CA8AAA", "#2A2A6A", "#EAAA8A", "#4A4A4A", "#8ACAEA", "#6A6A2A", "#AAEACA", "#8A0A0A", "#6A8AAA"
]

class IC50Plot:

    def __init__(self):
        sns.set()
        sns.set_style("darkgrid", {"axes.facecolor": "#EFEDF5"})
        self.width=self.height=-1

    def plot_init(self, width=200, height=150, popup=False):
        #sns.set()
        plt.close('all')
        plt.clf()
        #if popup:
        #    sns.set_style("ticks")
        my_dpi=96
        #plt.clf()
        fig=plt.gcf()
        #self.width, self.height=fig.get_size_inches()*fig.dpi
        my_dpi=96
        plt.figure(figsize=(width/my_dpi, height/my_dpi), dpi=my_dpi)
        SCALE=max(width, height)/200
        fontsize=min(max(8*SCALE, 5), 14)
        labelsize=max(min(fontsize*0.85, 12), 5)
        matplotlib.rc('xtick', labelsize=fontsize*0.85)
        matplotlib.rc('ytick', labelsize=fontsize*0.85)
        matplotlib.rc('axes', titlesize=fontsize)
        matplotlib.rc('axes', labelsize=fontsize)
        plt.gca().ticklabel_format(style='scientific')
        self.width, self.height = width, height

    def plot_one(self, data, fn_out, width=200, height=150, popup=False, scale=False):
        data=dict(data)
        data["Concentration"]=data["Concentration"].split(",")
        data["Response"]=data["Response"].split(",")
        data['TYPE']='IC50' if data['param_D']<0 else 'EC50'
        if pd.isnull(data['Fuzzy']) or data['Fuzzy']=='': data['Fuzzy']='='
        #if data['Fuzzy']!='=': return
        data=pd.DataFrame([data])
        data.columns=[x.upper() for x in data.header()]
        # GRP_ID can be used to control color
        data['GRP_ID']=0
        return self.plot(data, fn_out, width, height, popup, scale)

    def plot(self, data, fn_out, width=200, height=150, popup=False, scale=False):
        MAX_CURVES=16
        SCALE=max(width, height)/200
        LINEWIDTH=1*SCALE
        MARKERSIZE=max(3*SCALE, 3)

        def f(X, b, t, i, s):
            return b+(t-b)/(1+np.power(i/X, s))

        self.plot_init(width, height, popup=popup)
        #sw.check("plot_init")
        if data is None: return plt.gcf()
        plt.xscale('log')
        clr_by_grp=data.GRP_ID.max()>0
        #colors=sns.color_palette('dark', 6) if clr_by_grp else sns.color_palette('PuBuGn_d', MAX_CURVES)
        colors=COLORS if clr_by_grp else sns.color_palette('PuBuGn_d', MAX_CURVES)[::-1]
        try:
            x_min, x_max=0.0, 1.0
            conc=[]
            res=[]
            for i,r in data.iterrows():
                mask_x=np.array([ x!='' for x in r['CONCENTRATION']])
                mask_y=np.array([ x!='' for x in r['RESPONSE']])
                mask=mask_x & mask_y
                conc.append(np.array([float(x) for j,x in enumerate(r['CONCENTRATION']) if mask[j]]))
                y=np.array([float(x) for j,x in enumerate(r['RESPONSE']) if mask[j]])
                if scale and ('EC50' in r['TYPE']) and pd.notnull(r['FC_REF']):
                    d=(r['PARAM_A']*r['FC_REF'])
                    if d>0:
                        y=(y-r['PARAM_A'])/d
                        data.loc[i, 'PARAM_B']=(r['PARAM_B']-r['PARAM_A'])/d
                        data.loc[i, 'PARAM_A']=0
                        if pd.notnull(r['PARAM_A_STD_ERROR']): data.loc[i, 'PARAM_A_STD_ERROR']=r['PARAM_A_STD_ERROR']/d
                        if pd.notnull(r['PARAM_B_STD_ERROR']): data.loc[i, 'PARAM_B_STD_ERROR']=r['PARAM_B_STD_ERROR']/d
                res.append(y)
                x_min=min(x_min, round(np.log10(np.nanmin(conc[-1]))-0.5))
                x_max=max(x_max, round(np.log10(np.nanmax(conc[-1]))+0.5))
            #plt.gca().xaxis.set_ticks(np.power(10, np.arange(x_min, x_max)))
            X=np.logspace(x_min, x_max, 50 if popup else 30)
            #sw.check("Xmin/Xmax")
            for i,r in data[::-1].iterrows():
                # plot newer data last
                clr= colors[ min(r['GRP_ID'], len(colors)-1) ] if clr_by_grp else colors[i]
                if r['FUZZY']=='=' and not pd.isnull(r['PARAM_C']):
                    Y=f(X, float(r['PARAM_A']), float(r['PARAM_B']), float(r['PARAM_C']), float(r['PARAM_D']))
                    plt.plot(X, Y, color=clr, linewidth=LINEWIDTH)
                    if popup: # draw error bars
                        if not pd.isnull(r['PARAM_C_STD_ERROR']):
                            y_ic50=f(r['PARAM_C'], float(r['PARAM_A']), float(r['PARAM_B']), float(r['PARAM_C']), float(r['PARAM_D']))
                            d_x=np.exp(r['PARAM_C_STD_ERROR']/r['PARAM_C'])
                            plt.plot([r['PARAM_C']/d_x, r['PARAM_C']*d_x], [y_ic50, y_ic50], color=clr, linewidth=LINEWIDTH)
                        if not pd.isnull(r['PARAM_A_STD_ERROR']):
                            x_b=10**(x_min+np.random.rand()*0.2+0.05) if ('EC50' in r['TYPE']) else 10**(x_max-np.random.rand()*0.2-0.05)
                            y_b=f(x_b, float(r['PARAM_A']), float(r['PARAM_B']), float(r['PARAM_C']), float(r['PARAM_D']))
                            plt.plot([x_b, x_b], [y_b-r['PARAM_A_STD_ERROR'], y_b+r['PARAM_A_STD_ERROR']], color=clr, linewidth=LINEWIDTH)
                        if not pd.isnull(r['PARAM_B_STD_ERROR']):
                            x_t=10**(x_max-np.random.rand()*0.2-0.05) if ('EC50' in r['TYPE']) else 10**(x_min+np.random.rand()*0.2+0.05)
                            y_t=f(x_t, float(r['PARAM_A']), float(r['PARAM_B']), float(r['PARAM_C']), float(r['PARAM_D']))
                            plt.plot([x_t, x_t], [y_t-r['PARAM_B_STD_ERROR'], y_t+r['PARAM_B_STD_ERROR']], color=clr, linewidth=LINEWIDTH)
                if popup and not pd.isnull(r['OUTLIER']) and r['OUTLIER']!='':
                    outliers=[int(j)-1 for j in r['OUTLIER'].split(" ") if int(j)>0]
                    conc_=[x for j,x in enumerate(conc[i]) if j not in outliers]
                    res_=[x for j,x in enumerate(res[i]) if j not in outliers]
                    plt.plot(conc_, res_, color=clr, linestyle='', marker='o', markersize=MARKERSIZE, clip_on=False)
                    conc_=[x for j,x in enumerate(conc[i]) if j in outliers]
                    res_=[x for j,x in enumerate(res[i]) if j in outliers]
                    plt.plot(conc_, res_, color=clr, marker='x', linestyle='', markersize=MARKERSIZE, clip_on=False)
                else:
                    plt.plot(conc[i], res[i], color=clr, linestyle='', marker='o', markersize=MARKERSIZE, clip_on=False)
            #sw.check('Plot entries')
            plt.xlim(10**x_min, 10**x_max)
            ticks=np.power(10.0, np.arange(x_min, x_max+1))
            if len(ticks)>=8: # too many ticks
                ticks=ticks[::2]
            plt.minorticks_off()
            plt.xticks(ticks)
            ylim=list(plt.ylim())
            if 'EC50' in data.TYPE.tolist():
                ylim[0]=min(ylim[0], 0 if scale else 1)
            else:
                ylim[0]=min(ylim[0], 0)
                ylim[1]=max(ylim[1], 1)
            plt.ylim(ylim[0], ylim[1])
            if popup:
                plt.xlabel('Concentration (uM)')
                plt.ylabel('Response')
            #sw.check('before layout')
            plt.tight_layout(pad=0.1) #, w_pad=0, h_pad=0)
            #if not popup:
            #    plt.subplots_adjust(bottom=0.25, left=0.25)
            #sw.check('Plot post')
            plt.savefig(fn_out)
        except Exception as e:
           print(str(e))

