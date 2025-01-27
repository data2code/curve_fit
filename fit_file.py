#!/usr/bin/env python
import pandas as pd
import curve_fit.util as util
import numpy as np
import re
import curve_fit
import copy
import datetime as dt
import os
from curve_fit.dose_response_fit import SmartDoseResponseFit,Bounds
from curve_fit.plot_dr import IC50Plot
import json

class IO:

    @staticmethod
    def load_model(file_name):
        opt={'MODEL':'A', 'MONTE_CARLO':8, 'OUTLIER':True, 'MIN_FC':0.01, 'TRIM_FUZZY':True}
        model_id=0
        auc_min=auc_max=bounds=None
        toxicity=average=None

        data=json.loads(util.read_string(file_name))

        if 'model' in data:
            opt['MODEL']=data['model']
            if type(opt['MODEL']) is int:
                if opt['MODEL']==1:
                    opt['MODEL']='A'
                    model_id=1
                elif opt['MODEL']==4:
                    opt['MODEL']='A'
                    model_id=4
                elif opt['MODEL']==2:
                    opt['model']='B'
                    model_id=2
                else:
                    util.error_msg(f"Bad model in the json file: {data['model']}")
            else:
                if opt['MODEL']=='A':
                    model_id=1
                elif opt['MODEL']=='B':
                    model_id=2
                else:
                    util.error_msg(f"Bad model in the json file: {data['model']}")

        if 'bounds' in data:
            bounds=list(zip(*data['bounds']))
            LB,UB=Bounds.fix_bounds(bounds, model_id)
            if model_id==4: # CYP INHIBITION
                LB[0]=UB[0]=0.0
                LB[1]=UB[1]=1.0
            opt['LB']=LB
            opt['UB']=UB

        if 'nMC' in data:
            opt['MONTE_CARLO']=int(data['nMC'])
        if 'outlier' in data:
            opt['OUTLIER']=int(data['outlier'])>0
        if 'toxicity' in data:
            toxicity=int(data['toxicity'])>0
        if 'average_points' in data:
            average=int(data['average_points'])>0
        if 'auc_min' in data:
            if data['auc_min'] not in ('', 'X', null):
                auc_min=float(data['auc_min'])
        if 'auc_max' in data:
            if data['auc_min'] not in ('', 'X', null):
                auc_min=float(data['auc_max'])
        if toxicity is None:
            toxicity=opt['MODEL']=='B'
        if auc_min is not None and auc_max is not None:
            opt['AUC']=[(auc_min, auc_max)]
        else:
            opt['AUC']=[]
        opt['TOXICITY']=toxicity
        opt['AVERAGE_POINTS']=average
        return opt

    @staticmethod
    def run_lines(data, default_opt, fmt, cpd_list, plot, auto_qc):
        S_out=[]
        l_cpd_list=cpd_list is not None and len(cpd_list)>0
        if cpd_list is None: cpd_list=[]
        if l_cpd_list:
            data=data[data.CPD.isin(cpd_list)]
        pg=util.Progress(len(data))
        for i_,r in data.iterrows():
            cpd,outlier,tox,avg,min_fc,auc,bounds=str(r['CPD']),r['OUTLIER'],r['TOXICITY'],r['AVERAGE_POINTS'],r['MIN_FC'],str(r['AUC']).strip(),str(r['BOUNDS']).strip()
            cpd='NA' if cpd=='' else cpd
            if cpd.startswith('#'): continue
            opt=copy.deepcopy(default_opt) # need deep copy as there are array elements
            if opt['DEBUG']:
                print(f"Processing compound: {cpd}")
            opt['CPD']=cpd
            if outlier!="": opt['OUTLIER']=int(outlier)>0
            if tox!="": opt['TOXICITY']=int(tox)>0
            if avg!="": opt['AVERAGE_POINTS']=int(avg)>0
            if min_fc!="" and min_fc!="X": opt['MIN_FC']=float(min_fc)
            if auc!="" and auc!="X":
                S2=auc.split(":")
                opt['AUC'].extend([ (float(S2[i]),float(S2[i+1])) for i in range(0, len(S2), 2) ])
            if bounds!="":
                S2=bounds.split(":")
                for i in range(0, len(S2), 2):
                    j=i//2
                    if j==2: j=3 # the 3rd is for slope
                    opt['LB'][j]=float(S2[i])
                    opt['UB'][j]=float(S2[i+1])
            opt['LOCK']=np.zeros(4, dtype=bool)
            for i in range(4):
                if opt['LB'][i]==opt['UB'][i]: opt['LOCK'][i]=True

            concentration=r['CONCENTRATION']
            response=r['RESPONSE']
            #if cpd=='29760660': break
            #if cpd!='3421266': continue
            res=SmartDoseResponseFit.fit_a_cpd(util.sarray2rarray(concentration.split(",")), util.sarray2rarray(response.split(",")), opt)
            S_out.append(IO.report(res, fmt, auto_qc=auto_qc))
            S_out[-1]+=f",\"{concentration}\",\"{response}\""
            if curve_fit.curve_fit.DEBUG:
                res.print(brief=True)
                res.savefig(os.path.join('debug', re.sub(r'\W', '_', cpd)+".png"))
            pg.check(i_+1)

        return S_out

    @staticmethod
    def run(file_name, model_file=None, out_name=None, fmt="CSV", plot=False, cpd_list=None, auto_qc=True):
        """If cpd_list is provided, only calculate compounds in the list"""
        data=pd.read_csv(file_name, low_memory=False, dtype='string', keep_default_na=False)
        if model_file is not None:
            default_opt=IO.load_model(model_file)
        else:
            default_opt={}
        default_opt['DEBUG']=curve_fit.curve_fit.DEBUG
        #print(default_opt)
        S_out=[IO.header(fmt)]
        if plot:
            os.makedirs(plot, exist_ok=True)
        if curve_fit.curve_fit.DEBUG:
            os.makedirs('debug', exist_ok=True)
        # use one CPU
        S_out.extend(IO.run_lines(data, default_opt, fmt, cpd_list, plot, auto_qc))
        if out_name is not None:
            util.save_list(out_name, S_out, s_end="\n")
        if plot is not None:
            print("Plotting ...")
            cols=['CPD','Fuzzy','outlier','lock','comment','ignore','Mask','Note','Concentration','Response']
            t=pd.read_csv(out_name, low_memory=False, dtype={x:'string' for x in cols})
            for x in cols: t[x]=t[x].fillna('')
            plt=IC50Plot()
            pg=util.Progress(len(t))
            for i,r in t.iterrows():
                plt.plot_one(r, f"{plot}/{r['CPD']}.png", width=400, height=300, popup=True)
                pg.check(i+1)

    @staticmethod
    def header(format='CSV'):
        if format=='CSV':
            #return 'CPD,bottom,top,fuzzy,IC50,slope,d_bottom,d_top,d_IC50,d_slope,R2,outlier,lock,comment,firstPct,lastPct,outlierSensitivity,FC,ignore,AUC'
            return 'CPD,param_A,param_B,Fuzzy,param_C,param_D,param_A_std_error,param_B_std_error,param_C_std_error,param_D_std_error,R2,outlier,lock,comment,firstPct,lastPct,outlierSensitivity,FC,ignore,AUC,Mask,Note,Concentration,Response'
        else:
            return '<RESULT>CPD,param_A,param_B,Fuzzy,param_C,param_D,param_A_std_error,param_B_std_error,param_C_std_error,param_D_std_error,R2,outlier,lock,comment,firstPct,lastPct,outlierSensitivity,FC,ignore,AUC,Mask,Note,Concentration,Response</RESULT>'

    @staticmethod
    def report(res, format="CSV", auto_qc=False):
        X, Y=res._dd.X.values, res._dd.Y.values
        ic50=np.power(10, res._d['p'][2])
        d_ic50=np.log(10)*ic50*res._d['perr'][2]
        n=len(X)
        outliers=ignores=''
        if 'OUTLIER' in res._dd.header():
            #res._dd.display()
            #print(res._dd.ID[res._dd['OUTLIER']])
            outliers=" ".join([str(x) for x in res._dd.ID[res._dd['OUTLIER']]])
        if 'TOXICITY' in res._dd.header():
            ignores=" ".join([str(x) for x in res._dd.ID[res._dd['TOXICITY']]])
        locks=" ".join(["1" if x else "0" for x in res._d['locks']])
        AUCs=" ".join(['{:.4g}'.format(x) for x in res._d['AUC']])
        fuzzy=res._d['fuzzy']
        if fuzzy=='=': fuzzy=''
        if auto_qc:
            res.auto_qc()
        else:
            res._d['mask']=res._d['note']=0
        s_out=f"""{res._d['cpd']},{res._d['p'][0]:.4g},{res._d['p'][1]:.4g},{fuzzy},{ic50:.4g},{res._d['p'][3]:.4g},{res._d['perr'][0]:.4g},{res._d['perr'][1]:.4g},{d_ic50:.4g},{res._d['perr'][3]:.4g},{res._d['r2']:.4g},{outliers},{locks},{res._d['comment']},{res._d['firstPct']:.4g},{res._d['lastPct']:.4g},{res._d['outlier_sensitivity']:.4g},{res._d['fc']:.4g},{ignores},{AUCs},{res._d['mask']},{res._d['note']}"""
        s_out=re.sub(r'(?<=,)nan(?=,)', 'NaN', s_out)
        if format=="CSV":
            return s_out
        else:
            return "<RESULT>{}</RESULT>".format(s_out)

if __name__=="__main__":
    # for example, use build_example_data.py (need to modify the job_id inside)
    #
    # ./build_example_data.py > example.in
    #
    # modify example.m as needed (fitting parameter file)
    # then run with multiple CPUs, -n 10 in the example below
    #
    # ./fit_file.py -i example.in -m example.m -o output.csv -n 10
    #

    import argparse as arg
    opt=arg.ArgumentParser(description='Dose-Response Curve Fitting on Files')
    opt.add_argument('-i', '--input', type=str, default=None, help='input data file')
    opt.add_argument('-o','--output', type=str, default=None, help='output data file')
    opt.add_argument('-m','--model', type=str, default=None, help='model parameter file')
    opt.add_argument('-p','--plot', default=None, help='output folder name contains figures')
    opt.add_argument('-d','--debug', default=False, action='store_true', help='print debug message, save debug images into a folder called debug.')
    opt.add_argument('-c','--cpd', type=str, default=None, help='a compound ID to process')
    opt.add_argument('-x','--xml', default=False, action='store_true', help='print output within XML tags')
    opt.add_argument('-a','--auto_qc', default=True, action='store_false', help='apply Auto QC, will add additional columns: Mask, Note.')
    args=opt.parse_args()
    if args.input is None:
        util.error_msg('Missing input file')
    if args.output is None:
        util.error_msg('Missing output file')
    if args.model is None:
        util.error_msg('Missing model setting file')

    sw=util.StopWatch()
    curve_fit.curve_fit.DEBUG=args.debug
    cpd_list=None
    if args.cpd is not None:
        cpd_list=[args.cpd]
    fmt='CSV'
    if args.xml: fmt='XML'
    io=IO.run(args.input, args.model, args.output, plot=args.plot, cpd_list=cpd_list, auto_qc=args.auto_qc, fmt=fmt)
    sw.check('DONE')
