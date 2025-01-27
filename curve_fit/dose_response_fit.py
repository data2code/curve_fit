#!/usr/bin/env python
import pandas as pd
import curve_fit.util as util
import numpy as np
import scipy.stats as ss
from scipy.optimize import least_squares
from scipy.linalg import svd
import re
from .curve_fit import DEBUG, CurveFit, Result, debug
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from scipy.integrate import quad
import datetime as dt
#import parallel
from pprint import pprint

#minimal acceptable solution
LEAST_R2=0.5
#minimal acceptable R2 for an outlier solution, otherwise, ignore the outlier
LEAST_OUTLIER_R2=0.6
#if outlier is sensitive and original solution is >0.8, use the original solution
ACCEPTABLE_R2=0.8
# at least six points to trigger outlier detection
MIN_POINTS_OUTLIER=6
# Acceptable unstable solution
ACCEPTABLE_UNSTABLE_R2=0.8
# outlier solution has to have p value less than this to consider better
# is_better()
OUTLIER_IS_BETTER_P=0.2


class DoseResponseFit(CurveFit):
    """Set basic fitting capability, but no Fuzzy logic"""

    def __init__(self, model='A', **kw):
        """model A/B"""
        self.model=model
        self.extra={}
        self.set_extra(**kw)
        super(DoseResponseFit, self).__init__(res_cls=DoseResponseResult)

    def set_extra(self, **kw):
        """Utility function to store opt_lb and opt_ub,
        which will be passed into DoseResponseResult object"""
        for k,v in kw.items():
            self.extra[k]=v

    def get_extra(self, s):
        return self.extra.get(s)

    def f(self, X, p):
        return p[0]+(p[1]-p[0])/(1+np.power(10.0, (p[2]-X)*p[3]))

    def jacobi_wrapper(self, X, Y, p0, locks):
        # p is only for unlocked part, p0 has all values
        # return a matrix, where each row is a data point
        # each col is a unlocked fitting parameter
        def jac(p):
            # J_{i,k} = \frac{\partial{f(x_i)}{\partial{p_k}}
            p_=np.array(p0)
            p_[~locks]=p
            IC_x=p_[2]-X
            IC_x_s=np.power(10.0, IC_x*p_[3])
            IC_x_s_1=1/(1+IC_x_s)
            IC2=IC_x_s_1*IC_x_s_1*IC_x_s*np.log(10)*(p_[0]-p_[1])
            J=np.vstack((1-IC_x_s_1, IC_x_s_1, IC2*p_[3], IC2*IC_x ))
            return J[~locks].T
        return jac

    def p_init(self, data, lb, ub):
        """Initial guess"""
        X, Y=data[~data.OUTLIER].X.values, data[~data.OUTLIER].Y.values
        if len(X)<2:
            return np.array([np.nan, np.nan, np.nan, np.nan])
        p0=np.zeros(4)
        p0[0]=np.min(Y)
        p0[1]=np.max(Y)
        n=len(X)
        if self.model=='B' and n>1:
            # for EC50, min(Y) may due to toxicity
            p0[0]=max(p0[0], np.mean(Y[-2:]))
        if lb is not None and ub is not None and len(lb)>1 and len(ub)>1:
            p0[0]=np.clip(p0[0], lb[0], ub[0])
            p0[1]=np.clip(p0[1], lb[1], ub[1])
        IDX=np.argsort(np.abs(Y-(p0[0]+p0[1])/2))
        p0[2]=np.mean(X[IDX[:3]])
        p0[3]=-1 if self.model=='A' else 1
        if lb is not None and ub is not None and len(lb)>1 and len(ub)>1:
            p0=np.clip(p0, lb, ub)
        return p0

    def fit(self, data, p0=None, locks=None, lb=None, ub=None):
        # prevent side effect
        if p0 is not None: p0=p0.copy()
        if locks is not None: locks=locks.copy()
        if lb is not None: lb=lb.copy()
        if ub is not None: ub=ub.copy()
        data=data.copy()
        res=super().fit(data, p0, locks, lb, ub)
        res._d['model']=self.model
        # need to pass in opt_lb opt_ub for post_calc()
        opt_lb=self.get_extra('opt_lb')
        opt_ub=self.get_extra('opt_ub')
        res.set_bounds(opt_lb, opt_ub)
        res.post_calc()
        #debug('curve_fit.DoseReponse.fit():', [res.data['success'], res.data['p']])
        if not res._d['success'] and self.model=='B':
            debug('DoseResponseFit.fit(): retry and fix top')
            # If fails, for EC50, it may not have plateau at the top, tries to lock the top and try again
            mylocks=res._d['locks'].copy()
            myp0=res._d['p'].copy()
            LOCK_FACTOR=1.0 # it was 2.0, changed on 1/15/2010
            if not mylocks[1]:
                mylocks[1]=True
                # this seems to be a bug 9/13/2012, should not have -1.0 at the end (probably came from when LOCK_FACTOR was set to 2.0 initially
                #my_x(2)=max(lb(2), max(ydata(~my_outlier))*LOCK_FACTOR-1.0);

                # we need to use opt_lb, opt_ub, the reason is this could be within an outlier loop
                # if an outlier masked was the highest point, the original ub passed in was not right
                # as that ub was set to be (0.9*y_max, y_max)
                #myp0[1]=min(ub[1], max(lb[1], res.max_Y()*LOCK_FACTOR))
                myp0[1]=np.clip(res.max_Y()*LOCK_FACTOR, opt_lb[1], opt_ub[1])
                debug('DoseResponseFit.fit(), lock top and fit again.')
                lb[1]=ub[1]=myp0[1]
                myres=super().fit(data, myp0, mylocks, lb, ub)
                #debug("**********************************", lb, ub)
                #myres.print()
                myres._d['model']=self.model
                myres.set_bounds(opt_lb, opt_ub)
                myres.post_calc()
                if myres._d['success']:
                    res=myres.set_previous(res, 'DoseResponse.fit(): Refit by locking top in model B')
                # if still fails, lock the bottom to 1.0 and give it one more chance (may not plateau at the bottom)
            if not res._d['success'] and not mylocks[0]:
                mylocks[0]=True
                #myp0[0]=max(lb[0], min(ub[0], 1))
                myp0[0]=np.clip(1., opt_lb[0], opt_ub[0])
                lb[0]=ub[0]=myp0[0]
                debug('DoseResponseFit.fit(), lock bottom and fit again.')
                myres=super().fit(data, myp0, mylocks, lb, ub)
                #debug("**********************************", lb, ub)
                #myres.print()
                myres._d['model']=self.model
                myres.set_bounds(opt_lb, opt_ub)
                myres.post_calc()
                if myres._d['success']:
                    res=myres.set_previous(res, 'DoseResponse.fit(): Refit by locking bottom in model B')
        # bottom is higher (when the contrains overlap, the curve may be flipped)
        res.validate(inplace=True)

        return res

    def fit_outlier(self, data, p0=None, locks=None, lb=None, ub=None):
        """outlier identification routine if r2<GOOD_R2
            GOOD_R2 is defined in CurveFit.GOOD_R2, default to 0.9
        """
        MAX_SENSITIVITY_INDEX=2.0

        #data=data.copy()
        res0=self.fit(data, p0, locks, lb, ub)
        if res0._d['success'] and res0._d['r2']>=self.GOOD_R2:
            return res0
        # leave-one-out outlier identification
        n=len(data)
        data.index=range(n)
        best=res0
        debug(f"DoseResponseFit.fit_outlier(): Outlier testing, original solution r2: {res0._d['r2']:.4g}, has_fit:{res0.has_fit()}")
        ic50s=[]
        if res0.has_fit(): ic50s.append(res0._d['p'][2])
        # keep a copy for each trial, so they are not isolated from other trials
        for i in range(n):
            data['OUTLIER']=False
            data.loc[i, 'OUTLIER']=True
            res=self.fit(data, p0, locks, lb, ub)
            debug(f"Try outlier i={i}, has_fit={res.has_fit()}, r2={res._d['r2']:.4g}, p=", res._d['p'])
            if res.has_fit(): ic50s.append(res._d['p'][2])
            print(res._d['success'], best._d['r2'], res._d['r2'])
            if res._d['success'] and res._d['r2']>=self.LEAST_OUTLIER_R2 and res._d['r2']>best._d['r2']:
                debug(f"DoseResponseFit.fit_outlier(): Better candidate found {i+1} of {n}: r2={res._d['r2']:.4g}, ic50={res._d['p'][2]:.4g}")
                #res._dd.display()
                best = res
        ic50s=np.array(ic50s)

        # see if the outlier solution is better than original res0
        is_better=best.is_better(res0, OUTLIER_IS_BETTER_P)
        debug(f"Is outlier better? success={best._d['success']}, is better={is_better}")
        if (best._d['success'] and is_better):
            debug('DoseResponseFit.fit_outlier(): outlier solution might be better')
            #% calculate outlierSensitivity
            # the following logic only works with MULTIPLE_OUTLIER=False, as we haven't really use MULTIPLE mode
            outlier_sensitivity = np.max(np.abs(ic50s-best._d['p'][2]))/best._d['perr'][2]

            #% most likely a wrong outlier, we reset it to no outlier case
            # if the original solution is really bad, then we will take this outlier solution
            if (outlier_sensitivity > MAX_SENSITIVITY_INDEX) and res0.has_fit() and (res0._d['r2']>=ACCEPTABLE_R2):
                debug(f'DoseResponseFit.fit_outlier(): outlier too sensitive {outlier_sensitivity:.4g}, use initial solution', \
                    [res0._d['p'], best._d['p'], res0._d['perr']])
                return res0

            debug(f'DoseResponseFit.fit_outlier(): Outlier solution is accepted, outlier_sensitivity: {outlier_sensitivity:.4g}')
            #% check if there are at least two data points in between an outlier and IC50, if yes, the outlier has not much impact
            #% on the IC50 calculation, we set outlierSensitivity to a negative value
            critical=False
            best._dd.sort_values('X', ascending=False, inplace=True)
            tmp=best._dd.OUTLIER[best._dd.X<best._d['p'][2]]
            if np.sum(tmp[:2])>0:
                critical=True
            tmp=best._dd.OUTLIER[best._dd.X>best._d['p'][2]]
            if np.sum(tmp[-2:])>0:
                critical=True
            if not critical:
                outlier_sensitivity=-outlier_sensitivity
            debug('DoseResponseFit.fit_outlier(): accept outlier solution.')
            if DEBUG:
                debug("vvv Outlier solution *****************************")
                best.print()
                debug("^^^ Outlier solution *****************************")
            best._d['outlier_sensitivity']=outlier_sensitivity
            return best

        debug(f'DoseResponseFit.fit_outlier(): stick to original solution')
        res0._d['outlier_sensitivity']=np.nan
        return res0

    #def trivial_solution(self, data, lb=None, ub=None):
    #    res=DoseResponseResult({'data': data, 'lb':lb, 'ub':ub})
    #    return res.trivial_solution()

class DoseResponseResult(Result):
    """Store the DR fitting result, also all the data and parameters used to produce that result,
    as well as keep the fitting history.
    This enable us to take a result and reprocess it with another way, as it contains all the info.
    """

    def __init__(self, kw1, **kw2):
        super(DoseResponseResult, self).__init__(kw1, **kw2)

    def plot(self):
        """instead of calling super().plot, we overwrite it, b/c
        super().plot cannot handle toxicity point, so the curve line may be too short if
        the last data point is a toxic point"""
        sns.set()
        plt.figure(figsize=(4,3))
        plt.clf()

        #https://github.com/mwaskom/seaborn/issues/3462
        import warnings
        warnings.filterwarnings("ignore", "is_categorical_dtype")
        warnings.filterwarnings("ignore", "use_inf_as_na")

        def show_curve(X0, p, perr, color):
            f=self._d['f']
            Y0=f(X0, p)
            sns.lineplot(x='X', y='Y', data=pd.DataFrame({'X':X0, 'Y':Y0}), color=color)
            x0=np.array([X0[0]+0.05, p[2], X0[-1]-0.05])
            y0=f(x0, p)

            if perr is None: return

            dx0=perr[2]
            # plot error_bar at ic50
            plt.plot([x0[1]-perr[2],x0[1]+perr[2]], [y0[1],y0[1]], color=color)
            if self._d['model']=='A':
                plt.plot([x0[0],x0[0]], [y0[0]-perr[1], y0[0]+perr[1]], color=color)
            else:
                plt.plot([x0[0],x0[0]], [y0[0]-perr[0], y0[0]+perr[0]], color=color)
            if self._d['model']=='A':
                plt.plot([x0[2],x0[2]], [y0[2]-perr[0], y0[2]+perr[0]], color=color)
            else:
                plt.plot([x0[2],x0[2]], [y0[2]-perr[1], y0[2]+perr[1]], color=color)

        if len(self._dd)==0: return
        X0=np.linspace(self._dd.X.min(), self._dd.X.max(), 50)
        p=self._d['p']
        if sum(np.isnan(X0))==0 and self._d['f'] is not None:
            if self.has_fit():
                show_curve(X0, self._d['p'], self._d['perr'], '#e6550d')
            else:
                show_curve(X0, self._d['auc_p'], None, '#636363')

        data=self._dd
        if 'TOXICITY' in data.header() and data.TOXICITY.sum()>0:
            t=data[data.TOXICITY]
            # markers: https://matplotlib.org/2.0.2/api/markers_api.html
            # to show markers, one has to provide a style column, here True become a numerical column
            plt.plot(t.X, t.Y, color='#de2d26', marker='+', linestyle='')
            #x='X', y='Y', data=t, color='#de2d26', style=True, markers='+')
            data=data[~ data.TOXICITY].copy()
        if 'OUTLIER' in data.header() and data.OUTLIER.sum()>0:
            t=data[data.OUTLIER]
            plt.plot(t.X, t.Y, color='#636363', marker='x', linestyle='')
            data=data[~ data.OUTLIER].copy()
        if len(data):
            plt.plot(data.X, data.Y, color='#3182bd', marker='o', linestyle='')

        ax=plt.gca()
        b,t=ax.get_ylim()
        if self._d['model']=='A':
            b=min(0, b)
            t=max(1, t)
        else:
            b=min(1, b)
            t=max(1.5, t)
        plt.ylim(b, t)
        ax.set(xlabel='Log10(uM)', ylabel='Response')
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        # show parameters
        s="bottom={:.4g}\ntop={:.4g}\nlog(ic50)={}{:.4g}\nslope={:.4g}\nfc={:.4g}\nR2={:.3g}".format(p[0],p[1],self._d['fuzzy'],p[2],p[3],self._d['fc'],self._d['r2'])
        if self._d['model']=='A':
            ax.text(self._dd.X.min()+0.1, 0.05, s, fontsize=9)
        else:
            ax.text(self._dd.X.min()+0.1, 1.05, s, fontsize=9)
        plt.tight_layout()

    def clone(self):
        res=DoseResponseResult(copy.deepcopy(self.data))
        return res

    def set_bounds(self, opt_lb, opt_ub):
        # remember the original bounds, which could be different from lb,ub
        # the latter can be changed during the fitting process
        self._d['opt_lb']=opt_lb.copy()
        self._d['opt_ub']=opt_ub.copy()

    def trivial_solution(self, inplace=False):
        """Since the res maybe due to top/bottom locking, the self._d['lb'] and self._d['ub']
        is no longer good for trival determination, so we need to use fitting opt['LB'], opt['UB']
        Original bounds are stored in opt_lb and opt_ub"""
        res=self.clone()
        # undo outliers in fuzzy solution
        if 'OUTLIER' in res._dd.header():
            res._dd['OUTLIER']=False
        min_x, max_x=self.min_X(), self.max_X()
        min_y, max_y=self.min_Y(), self.max_Y()
        mu=res._dd.Y.mean()
        stdv=res._dd.Y.std()
        #Y0=self._dd[ ~self._dd.OUTLIER].Y.values
        Y0=self._dd.Y.values # no more outliers
        # we add new logic requiring at least two points to above threshold, in case
        # there was only one extrement high/low data point that dominate the mu/stdv calculation
        # second highest/lowest reads
        N_MIN=2
        if len(Y0)<N_MIN:
            second_max=second_min=np.nan
            if len(Y0)==1:
                second_max=second_min=Y0[0]
                stdv=0
        else:
            Y0=np.sort(Y0)
            second_max, second_min=Y0[-N_MIN], Y0[N_MIN-1]
        if self._d['model']=='A':
            threshold=0.5
            if mu>=0.7 and mu/2<threshold:
                threshold=mu/2
            elif mu<=0.3 and (mu+1)/2>threshold:
                # if its potent, mu is the bottom instead of top
                threshold=(mu+1)/2
            if (min(mu-stdv, second_max)>threshold or min_y>threshold): # or mu>=0.7):
                res._d['fuzzy'], res._d['p'][2] = ('>', max_x)
            elif (max(mu+stdv, second_min)<threshold or max_y<threshold): # or mu<=0.3):
                res._d['fuzzy'], res._d['p'][2] = ('<', min_x)
        else:
            threshold=min(self._d['opt_ub'][0], self._d['opt_lb'][1])
            threshold=max((threshold+1)/2, self._d['opt_lb'][1]-0.2)
            if (max(mu+stdv, second_max)<threshold or max_y<self._d['opt_lb'][1]): # or mu<=threshold):
                res._d['fuzzy'], res._d['p'][2] = ('>', max_x)
            elif (min(mu, second_min)>(max(self._d['opt_ub'][0], self._d['opt_lb'][1])+0.3)):
                # set to a more stringent condition, because sometimes FC does not meet minEff
                res._d['fuzzy'], res._d['p'][2] = ('<', min_x)
        # No fit case
        if res._d['fuzzy']=='=': res._d['p'][2]=np.nan
        res._d['success']=False
        res.post_calc()
        if not inplace:
            return res
        else:
            res.set_previous(self.data, 'Set to trivial solution')
            self.data=res.data
            return self

    def validate(self, inplace=False):
        """Invaliate solution, if bottom is higher than top
        When the constrains of bottom and top overlaps, for shallow curves,
        one might get a solution where bottom is higher than top.
        This is b/c the sign of slope alone does not force the direction of the curve
        """
        if self._d['p'][0] <= self._d['p'][1]: return self
        res=self.clone()
        res._d['success']=False
        res._d['r2']=0
        val=(res._d['p'][0]+res._d['p'][1])/2
        res._d['p'][0]=res._d['p'][1]=val
        res._d['auc_p'][0]=res._d['auc_p'][1]=val
        res.post_calc()
        if not inplace:
            return res
        else:
            res.set_previous(self.data, 'Curve if flipped, not a valid fit.')
            self.data=res.data
            return self

    def trim_ic50(self, min_x=None, max_x=None, inplace=False):
        """If IC50 is outslide the boundary, we would rather return a fuzzy value
        If the max conc is 10uM, instead of reporting 12uM, we would rather use >10uM.
        Keeping the max at 10uM make the data comparison easier in practice."""

        if min_x is None: min_x=self.min_X()
        if max_x is None: max_x=self.max_X()
        fuzzy, ic50=self._d['fuzzy'], self._d['p'][2]
        change=False
        if ic50>max_x:
            fuzzy, ic50='>', max_x
            change=True
        elif ic50<min_x:
            fuzzy, ic50='<', min_x
            change=True
        if change:
            debug(f"Trim IC50: before fuzzy={self._d['fuzzy']}, IC50={self._d['p'][2]:.4g}, after: {fuzzy}{ic50:.4g}")
            res=self.clone()
            res._d['fuzzy']=fuzzy
            res._d['p'][2]=ic50
            res._d['OUTLIER']=False
            res._d['success']=False
        else:
            return self
        if not inplace:
            return res
        else:
            res.set_previous(self.data, 'Trim IC50 to be within the range')
            self.data=res.data
            return self

    def trim_fuzzy(self, min_x, max_x, inplace=False):
        """Due to toxicity and outliers, we may be fuzzy values not ends with boundaries values,
        although it is conceptually more accurate, as we cannot say about outliers,
        it is easier to reset the value to bounds for practical convenience."""
        if self.has_fit(): return self
        l_fix=False
        if self._d['fuzzy']=='>' and self._d['p'][2]<max_x:
            res=self.clone()
            res._d['p'][2]=max_x
            l_fix=True
        elif self._d['fuzzy']=='<' and self._d['p'][2]>min_x:
            res=self.clone()
            res._d['p'][2]=min_x
            l_fix=True
        if not l_fix: return self
        if not inplace:
            return res
        else:
            res.set_previous(self.data, 'Trim fuzzy IC50 to use Min and Max')
            self.data=res.data
            return self

    def pct_ending(self, p):
        comment=''
        if self.has_fit():
            #self._dd.display()
            min_x=self.min_X()
            max_x=self.max_X()
            Y0=self._d['f'](np.array([min_x, max_x]), p)
            Y0=(Y0-p[0])/(p[1]-p[0]+1e-3)
            min_y=(self.min_Y()-p[0])/(p[1]-p[0]+1e-3)
            max_y=(self.max_Y()-p[0])/(p[1]-p[0]+1e-3)
            if self._d['model']=='A':
                firstPct=min(Y0[0], max_y)
                lastPct=max(Y0[1], min_y)
                if firstPct<0.8:
                    comment='Missing Top'
                if lastPct>0.2:
                    comment='Missing Bottom'
                if firstPct<0.8 and lastPct>0.2:
                    comment='Missing Ends'
            else:
                firstPct=max(Y0[0], min_y)
                lastPct=min(Y0[1], max_y)
                if firstPct>0.2:
                    comment='Missing Bottom'
                if lastPct<0.8:
                    comment='Missing Top'
                if firstPct>0.2 and lastPct<0.8:
                    comment='Missing Ends'
        else:
            firstPct=lastPct=np.nan
        return (firstPct, lastPct, comment)

    def fc(self):
        fc=bottom=top=None
        min_x, max_x=self.min_X(), self.max_X()
        if self.has_fit():
            bottom, top = self._d['p'][0], self._d['p'][1]
        else:
            residue=self._d.get('auc_residue', np.inf)
            if not np.isinf(residue):
                if self._d['model']=='A':
                    residue/=self._d['auc_p'][1]
                else:
                    residue/=self._d['auc_p'][0]
            # acceptable R2 to use for FC, even if no fit
            LEAST_FC_R2=0.7
            MAX_FC_RESIDUE=0.07
            min_x=self.min_X()
            max_x=self.max_X()

            # check if auc_p is a good approximate for FC calculation
            firstPct, lastPct, comment=self.pct_ending(self._d['auc_p'])

            if comment!='' and ((self._d.get('auc_r2', 0)>=LEAST_FC_R2 or residue<=MAX_FC_RESIDUE)):
                bottom, top = self._d['auc_p'][0], self._d['auc_p'][1]
                f=self._d['f']
                if (self._d['model']=='B'):
                    top=f(self.max_X(), self._d['auc_p'])
                else:
                    bottom=f(self.max_X(), self._d['auc_p'])
            elif (self._d['model']=='B'):
                bottom=1
                # use Y from max X, average if multiple points
                top=self._dd[np.abs(self._dd.X.values-max_x)<1e-20]['Y'].median()
            else:
                top=1
                bottom=self._dd[np.abs(self._dd.X.values-max_x)<1e-20]['Y'].median()
        if (bottom>top): # invalid curve
            top=bottom=(top+bottom)/2
        fc=self.fc2(bottom, top)
        return (fc, bottom, top)

    def fc2(self, bottom, top):
        if bottom is None or top is None: return None
        if self._d['model']=='B':
            return (top-bottom)/bottom
        else:
            return (top-bottom)/top

    def monte_carlo(self, n_MC=8):
        l_has_outlier=False
        if 'OUTLIER' not in self._dd.header():
            self._dd['OUTLIER']=False
        opt_lb=self._d['opt_lb']
        opt_ub=self._d['opt_ub']
        drfit=DoseResponseFit(self._d['model'], opt_lb=opt_lb, opt_ub=opt_ub)
        drfit.monte_carlo(self, n_MC=n_MC)

    def undo_monte_carlo(self):
        DoseResponseFit(self._d['model']).undo_monte_carlo(self)

    def has_fit(self):
        """Has curve, not a fuzzy value or has any missing value in curve parameters"""
        #return (self._d['success'] and (self._d['fuzzy']=='=' and not np.any(np.isnan(self._d['p'])) ))
        return (self._d['fuzzy']=='=' and not np.any(np.isnan(self._d['p'])))

    def post_calc(self):
        """Add fc, firstPct, lastPct, comment"""
        if 'outlier_sensitivity' not in self._d:
            self._d['outlier_sensitivity']=np.nan
        if 'auc' not in self._d:
            self._d['auc']=[]
        if 'fuzzy' not in self._d:
            self._d['fuzzy']='='
        comment=''
        X=self._dd.X.values
        Y=self._dd.Y.values
        if 'OUTLIER' not in self._dd.header(): self._dd['OUTLIER']=False
        outliers=self._dd.OUTLIER.values

        firstPct, lastPct, comment=self.pct_ending(self._d['p'])
        self._d['firstPct']=firstPct
        self._d['lastPct']=lastPct
        self._d['comment']=comment
        #% 2/2/2010 calculate FC
        Y2=Y[~outliers]
        n=len(Y2)
        if self.has_fit():
            bottom, top=self._d['p'][0], self._d['p'][1]
            if self._d['model']=='A':
                if comment in ['Missing Bottom', 'Missing Ends']:
                    bottom=np.clip(Y2[0], bottom, top)
                if comment in ['Missing Top','Missing Ends']:
                    top=np.clip(top, self._d['opt_lb'][1], self._d['opt_ub'][1])
            else:
                if comment in ['Missing Top','Missing Ends']:
                    top=np.clip(Y2[0], bottom, top)
                if comment in ['Missing Bottom','Missing Ends']:
                    bottom=np.clip(bottom, self._d['opt_lb'][0], self._d['opt_ub'][0])
            fc=self.fc2(bottom, top)
        else:
            fc, bottom, top=self.fc()
            self._d['p'][0], self._d['p'][1] = bottom, top
            self._d['success']=False
            self._dd['OUTLIER']=False
            self._d['outlier_sensitivity']=np.nan
        self._d['fc']=min(fc, 1e3) #user use wrong constrain, model B with botton close to 0, fc can be huge

    def print(self, ident="", brief=True):
        queue=[self]
        while queue[0].previous is not None:
            queue.insert(0, queue[0].previous)
        for i, res in enumerate(queue):
            ident="        "*i
            print()
            if 'cpd' in self._d: print(f"{ident}***{self._d['cpd']}***")
            print(f'{ident}Comment={self.previous_comment}')
            if not brief:
                for k,v in self._d.items():
                    pprint([ident, k, v])
            else:
                res._dd.display()
                print(f"{ident}success={res._d['success']}")
                fc=res._d.get('fc', np.nan)
                comment=res._d.get('comment', '')
                print(f"{ident}fuzzy={res._d['fuzzy']}, bottom={res._d['p'][0]:.4g}, top={res._d['p'][1]:.4g}, log(ic50)={res._d['p'][2]:.4g}, slope={res._d['p'][3]:.4g}, r2={res._d['r2']:.2g}")
                print(f"{ident}fc={fc:.4g}")
                print(f"{ident}bounds=[{res._d['lb'][0]:.4g}, {res._d['ub'][0]:.4g}, {res._d['lb'][1]:.4g}, {res._d['ub'][1]:.4g}]")
                if ('stable' in res._d): # has Monte Carlo
                    nMC_success=res._d.get('nMC_success', 0)
                    print(f"{ident}MC stable={res._d['stable']}, nMC_success={nMC_success}")
                    print(f"{ident}perr={res._d['perr'][0]:.4g},{res._d['perr'][1]:.4g},{res._d['perr'][2]:.4g},{res._d['perr'][3]:.4g}")
                    for perr in res._d['perr_mc']:
                        print(f"{ident}perr_mc={perr[0]:.4g},{perr[1]:.4g},{perr[2]:.4g},{perr[3]:.4g}")
                    print(f"{ident}perr_single={res._d['perr_single'][0]:.4g},{res._d['perr_single'][1]:.4g},{res._d['perr_single'][2]:.4g},{res._d['perr_single'][3]:.4g}")
                else:
                    print(f"{ident}perr={res._d['perr'][0]:.4g},{res._d['perr'][1]:.4g},{res._d['perr'][2]:.4g},{res._d['perr'][3]:.4g}")
                if 'auc_p' in self._d:
                    print(f"{ident}AUC_p={res._d['auc_p'][0]:.4g},{res._d['auc_p'][1]:.4g},{res._d['auc_p'][2]:.4g},{res._d['auc_p'][3]:.4g}, AUC_r2={res._d['auc_r2']:.4g}, AUC_residue={res._d['auc_residue']:.4g}")
            print(f"{ident}comment={res._d.get('comment','')}")
            print()
        print()

    @staticmethod
    def empty(model, data, kw):
        def nan(shape, value=np.nan):
            x=np.empty(shape)
            x[:]=value
            return x

        if len(data)==0:
            auc_p=nan(4)
            auc_perr=np.zeros(4)
        else:
            v=np.mean(data.Y)
            auc_p=np.array([v, v, data.X.mean(), -1 if model=='A' else 1])
            auc_perr=np.zeros(4)
        auc_r2=0
        auc_residue=np.inf
        data={'data':data, 'model': model, 'success':False, 'fuzzy':'=', 'p':nan(4), 'perr':nan(4), 'SSreg':np.nan, 'r2':0, 'df':0, 'F':np.nan, 'SStot':np.nan, 'n':len(data), 'n_param':np.nan, 'locks':np.zeros(4, dtype=bool), 'lb':nan(4), 'ub':nan(4), 'p0':nan(4), 'noise':np.nan, 'residue': np.nan, 'f':None, 'outlier_sensitivity':np.nan, 'AUC':[], 'auc_p':auc_p, 'auc_r2':auc_r2, 'auc_residue':auc_residue, 'auc_perr':auc_perr}
        if kw is not None:
            data.update(kw)
        res=DoseResponseResult(data)
        res.post_calc()
        return res

    def AUC(self, auc):
        """Calculate AUC, auc is a list of (x_min, x_max) ranges"""
        area=[]
        LEAST_R2=0.6
        for (c_min, c_max) in auc:
            if 'auc_p' not in self._d: # probably no fit case
                # not enough data points
                area.append(0)
                continue
            elif self._d['auc_r2']<LEAST_R2: # good enough for estimation
                # fit is not a good approximation
                if self._d['fuzzy']=='<':
                    area.append(self._d['fc'])
                else:
                    area.append(0) # for weak cpd or no fit
            else: # use fit curve
                p=self._d['auc_p']
                if self._d['model']=='A':
                    # area between curve and y=top, normalized by 1/top, so get values [0,1]
                    f=lambda x: (p[1]-p[0])/p[1]*(1-1/(1+10.**((-x+p[2])*p[3])))
                else:
                    # area between curve and y=bottom, normalized by 1/bottom, so get values [0,inf]
                    f=lambda x: (p[1]-p[0])/p[0]*(1/(1+10.**((-x+p[2])*p[3])))
                v = quad(f, np.log10(c_min), np.log10(c_max))[0]
                v /= (np.log10(c_max)-np.log10(c_min))
                area.append(v)
        self._d['AUC']=area

    def auto_qc(self):

        def update(mask, note):
            self._d['mask']=mask
            self._d['note']=note

        if (self._d['fuzzy'])!='=': return update('Auto Approved','')
        if pd.isnull(self._d['p'][2]): return update('Auto Approved','')
        if self._d['outlier_sensitivity']>2: return update('To Be Reviewed', 'Outlier')
        if self._d['r2']<=0.7: return update('To Be Reviewed', 'Low Confidence')
        if abs(self._d['p'][3])<0.5: return update('To Be Reviewed', 'Flat Curve')
        if self._d['model']=='A':
            if self._d['firstPct']<0.8: return update('To Be Reviewed', 'Curve Exprapolation (Top)')
            if self._d['lastPct']>0.2: return update('To Be Reviewed', 'Curve Exprapolation (Bottom)')
            if (self._d['p'][1]-self._d['p'][0])/self._d['p'][1] < 0.5: return update('To Be Reviewed', 'Flat Curve')
        else:
            if self._d['lastPct']<0.8: return update('To Be Reviewed', 'Curve Exprapolation (Top)')
            if self._d['firstPct']>0.2: return update('To Be Reviewed', 'Curve Exprapolation (Bottom)')
            if (self._d['p'][1]-self._d['p'][0])/self._d['p'][0] < 0.5: return update('To Be Reviewed', 'Flat Curve')
        return update('Auto Approved', '')


class SmartDoseResponseFit:

    def __init__(self, data, opt=None):
        """data has been pre-sorted, and there are at least X, Y, ID columns,
        X is in uM, not log-transformed yet"""
        #            res=dr.fit() #data.X.values, data.Y.values, None, None, opt['LB'], opt['UB'])
        self.opt={
            'MODEL': 'A',
            'LB': np.array([0, 0.8, -np.inf, -3.0]),
            'UB': np.array([0.2, 1.2, np.inf, -1/3]),
            'LOCK': np.zeros(4, dtype=bool),
            'OUTLIER': True,
            'MULTIPLE_OUTLIER': False,
            'MAX_OUTLIER': 2,
            'MIN_FC': 0.05,
            'MONTE_CARLO': 8,
            'AVERAGE_POINTS': True,
            'INTELLI_QC': True,
            'TOXICITY': False,
            'AUC': [],
            'TRIM_FUZZY': True, # make sure fuzzy values are set to either min/max, remove all outliers and toxicity points
            'DEBUG': False
        }
        if opt is not None:
            self.opt.update(opt)
        DEBUG=self.opt['DEBUG']
        self.data_keep=data.copy()
        self.data_keep.rename2({'X':'X0', 'Y':'Y0'})
        self.data_keep['X']=np.log10(data.X.clip(1e-12,1e6))
        self.data_keep['Y']=data.Y.clip(0,1e10)
        #self.data_keep.display()

        if self.opt['AVERAGE_POINTS']:
            grp=np.insert(np.abs(np.diff(self.data_keep.X.values))<1e-6, 0, False)
            if np.sum(grp)>0:
                self.data_keep['REPEAT']=grp # flag replicates, leave the first entry
                self.data_keep.loc[ ~ self.data_keep.REPEAT, 'GRP']=range(np.sum(~grp))
                self.data_keep['GRP']=self.data_keep['GRP'].fillna(method='ffill')
                self.data_keep['GRP']=self.data_keep['GRP'].astype(np.int8)
                self.data=self.data_keep[['GRP','X','Y']].groupby('GRP').mean().reset_index()
            else:
                self.data_keep['GRP']=self.data_keep.ID
                self.data=self.data_keep[['GRP','X','Y']].copy()
        else:
            self.data_keep['GRP']=self.data_keep.ID
            self.data=data_keep[['GRP','X','Y']].copy()
        self.data_keep['TOXICITY']=False
        if self.opt['TOXICITY']:
            self.toxicity()
        if 'OUTLIER' not in self.data.header():
            self.data['OUTLIER']=False
        self.data.sort_values('X', ascending=False, inplace=True)
        #self.data_keep.display()
        #self.data.display()
        # self.data only contain non-redundant, non-toxic data points, sorted from high conc to low conc
        # the ID column is now GRP instead of ID

    def set_opt(self, opt=None):
        if opt is not None: self.opt.update(opt)

    def fix_data_for_report(self, res):
        data=res._dd[['GRP','OUTLIER']]
        data=self.data_keep.merge(data, left_on='GRP', right_on='GRP', how='left')
        # since there are toxic points, the merged table contains missing value
        data['OUTLIER']=data['OUTLIER'].fillna(False)
        # just to be double sure
        data['OUTLIER']=data['OUTLIER'].astype(bool)
        # If no fit, let's also remove any outliers or toxic point flags
        if not res.has_fit():
            data['OUTLIER']=False
            data['TOXICITY']=False
        else:
            data.loc[data.TOXICITY, 'OUTLIER']=False
        data.sort_values('ID', inplace=True)
        res._d['data']=data

    def toxicity(self):
        if self.opt['MODEL']=='A': return
        #bad EC50 can show toxicity at high concentrations
        c_tox={}
        n_param=4 #a shortcut for now, 1/15/2010
        X, Y=self.data.X.values, self.data.Y.values
        LB,UB=self.opt['LB'],self.opt['UB']
        n=len(X)
        tox=np.zeros(n, dtype=bool)
        n_tox=0 #number of obvious toxic outliers
        max_y=np.max(Y)
        if max_y>max(LB[1], UB[0]):
            threshold=max(max_y-1, 0)*0.8+1
            for i in range(n-2):
                second=np.sort(Y[i+1:])[-2]
                if Y[i]<second and Y[i]<threshold and (n-n_tox)>=(n_param+1):
                    tox[i]=True
                    n_tox+=1
                else:
                    break
        # make sure self.data has no toxic point, to simply the fitting
        # fitting only needs to worry about OUTLIERS and TOXICITY won't be visible
        if np.sum(tox)>0:
            # flag data_in
            ids=self.data.GRP[tox]
            self.data_keep['TOXICITY']=self.data_keep.GRP.isin(ids)
            self.data=self.data[ ~ tox ].copy()

    def fit(self):
        X, Y=self.data.X.values, self.data.Y.values
        bounds=[self.opt['LB'].copy(), self.opt['UB'].copy()]
        opt_lb, opt_ub = self.opt['LB'].copy(), self.opt['UB'].copy()
        mylb=bounds[0].copy()
        myub=bounds[1].copy()

        DEBUG=self.opt['DEBUG']
        n=len(X)
        if n==0:
            res=DoseResponseResult.empty(self.opt['MODEL'], self.data, {'opt_lb':opt_lb, 'opt_ub':opt_ub })
            return res
        SKIP_INTELLI_QC=False
        INTELLI_QC=self.opt['INTELLI_QC']
        OUTLIER=self.opt['OUTLIER']
        MAX_INTELLI_QC=3
        n_MC=self.opt['MONTE_CARLO']
        min_FC=0.05 if pd.isnull(self.opt['MIN_FC']) else self.opt['MIN_FC']

        locks=self.opt['LOCK'].copy()
        myoutlier=OUTLIER

        #if n<=4:
        if n<=MIN_POINTS_OUTLIER: # too few points, disable outlier detection
            res = self.fit_entry(self.data, mylb, myub, outlier=False, n_MC=n_MC, min_FC=min_FC)
        else:
            res = self.fit_entry(self.data, mylb, myub, outlier=myoutlier, n_MC=n_MC, min_FC=min_FC)
        #res.print(brief=False)
        if res.has_fit() and res._d['r2']<LEAST_R2:
            debug('Bad fit, set to trivial')
            res.trivial_solution(inplace=True)
            SKIP_INTELLI_QC=True
            debug('SmartDoseResponseFit.fit(): Skip IntelliQC, we do not rescue fuzzy values')

        n_run=0
        while (INTELLI_QC and not SKIP_INTELLI_QC and n_run<=MAX_INTELLI_QC):
            n_run+=1
            if self.opt['DEBUG']:
                debug(f'vvvvv SmartDoseResponseFit.fit(): IntelliQC Run {n_run}')
                res.print()
                debug(f'^^^^^ SmartDoseResponseFit.fit(): IntelliQC Run {n_run}')
            #try to fix No Fit case
            # the following does not seem to be necessary, if there is a unstable solution
            # it is arleady in res
            #if not res.has_fit():
            #    if n_run==1:
            #        #variables start with "best_" memorize the best solution so far
            #        #refit without Monte Carlo, maybe the fit was not considered stable? better than nothing
            #        best_res=self.fit_entry(self.data, mylb, myub, myoutlier, n_MC=0, min_FC=min_FC)
            #        if best_res.has_fit() and best_res._d['r2']>=ACCEPTABLE_R2 and best_res._d['comment'] in ('Missing Top', 'Missing Bottom', 'Missing End'):
            #            res = best_res
            #    else:
            #        break
#
            if res.has_fit():
                X, Y, outliers=res._dd.X.values, res._dd.Y.values, res._dd.OUTLIER.values
                if self.opt['MODEL']=='A':
                    if np.any(outliers):
                        if res.min_Y()>=0.5 or res.max_Y()<=0.5:
                            debug("After outlier removal, remaining points are trivial")
                            res.trivial_solution(inplace=True)
                            continue
                        #X has been sorted descendingly
                        elif (X[0]==np.max(X) and sum(outliers)==1 and outliers[0] and Y[1]==np.min(Y) and np.sum(Y<=min(0.7, res._d['p'][1]*0.8))==1):
                            #outlier is the last data point and 2nd to last is the minimum and only data point <= 0.7
                            #sometime the curve goes down nicely and the last point is an obvious outlier, we want to keep that
                            res=self.fit_entry(self.data, mylb, myub, outlier=False, n_MC=n_MC, min_FC=min_FC)
                            if self.opt['DEBUG']:
                                util.info_msg('SmartDoseResponseFit.fit(): IntelliQC, reprocessing disabling outlier detection')
                            continue
                    if res.has_fit(): #check curve ends
                        refit=False
                        mylocks=res._d['locks']
                        if (not mylocks[1] and res._d['comment'] in ['Missing Top', 'Missing Ends']):
                            #mylb[1]=max(min(myub[1],1.0), mylb[1])
                            mylb[1]=np.clip(1.0, opt_lb[1], opt_ub[1])
                            myub[1]=mylb[1]
                            refit=True
                        if (not mylocks[0] and res._d['comment'] in ['Missing Bottom', 'Missing Ends']):
                            #mylb[0]=max(min(myub[0],0.0), mylb[0])
                            mylb[0]=np.clip(0.0, opt_lb[0], opt_ub[0])
                            myub[0]=mylb[0]
                            refit=True
                        if refit:
                            if self.opt['DEBUG']:
                                util.info_msg('SmartDoseResponseFit.fit(): IntelliQC, reprocessing fixing top/bottom')
                            self.data['OUTLIER']=False
                            res=self.fit_entry(self.data, mylb, myub, outlier=myoutlier, n_MC=n_MC, min_FC=min_FC)
                            continue
                else: #model B
                    if np.any(outliers):
                        #still need to change this part to <=max(ub(1),lb(2))
                        if res.max_Y()<=max(opt_ub[0], opt_lb[1]):
                            debug("SmartDoseResponseFit.fit(): After outlier removal, remaining points are trivial")
                            res.trivial_solution(inplace=True)

                    if res.has_fit():
                        mylocks=res._d['locks']
                        if (not mylocks[0] and res._d['comment'] in ['Missing Bottom', 'Missing Ends']):
                            #mylb[0]=max(min(myub[0],1.0), mylb[0])
                            mylb[0]=np.clip(1.0, opt_lb[0], opt_ub[0])
                            myub[0]=mylb[0]
                            if self.opt['DEBUG']:
                                util.info_msg('SmartDoseResponseFit.fit(): IntelliQC, reprocessing fixing bottom')
                            self.data['OUTLIER']=False
                            res=self.fit_entry(self.data, mylb, myub, outlier=myoutlier, n_MC=n_MC, min_FC=min_FC)
                            continue

                        #need to fix as well
                        if (res._d['comment'] in ['Missing Top', 'Missing Ends'] and (min(res.max_Y(), res._d['p'][1])<(opt_lb[1]-1)/2+1)):
                            #if res.max_Y()<=myub[0]:
                            res.trivial_solution(inplace=True)

            # no strategy any more
            break

        # we don't trim res within fit_entry(), b/c it could be missing top/bottom, which led to IC50 values outside the range
        # after top/bottom anchoring, the value might be correct after IntelliQC step, so we do it here
        #% %Eff < MinEff. We make sure fc is at least minFC.
        #% apply to model 1 as well 2/17/2012
        #%if (model==2&&fc<minFC)
        if res.has_fit() and res._d['fc']<min_FC:
            debug("SmartDoseResponseFit.fit(): FC<minFC, set to trivial")
            res.trivial_solution(inplace=True)
        if res.has_fit() and len(self.data_keep):
            # even if the last point is an outlier, we still keep extrapolation case, if it's within original x_max
            res.trim_ic50(self.data_keep.X.min(), self.data_keep.X.max(), inplace=True)
        if self.opt['TRIM_FUZZY'] and len(self.data_keep):
            res.trim_fuzzy(self.data_keep.X.min(), self.data_keep.X.max(), inplace=True)
        # calculate AUC
        res.AUC(self.opt['AUC'])
        return res

    def fit_entry(self, data, lb, ub, outlier=False, n_MC=0, min_FC=0.04):
        opt_lb, opt_ub=self.opt['LB'][:], self.opt['UB'][:] # original bounds before locking adjustment
        drfit=DoseResponseFit(self.opt['MODEL'], opt_lb=opt_lb, opt_ub=opt_ub)
        n=len(data)
        data['OUTLIER']=False
        #data.display()
        locks=np.zeros(4, dtype=bool)
        # trivial X or too few input data points
        # at least 3 points for IC50, at least 4 points for EC50
        if (data.X.max()==data.X.min() or (n<2 and self.opt['MODEL']=='A') or (n<3 and self.opt['MODEL']=='B')):
            if self.opt['DEBUG']:
                util.info_msg('SmartDoseResponseFit.fit_entry(): not enough data points')
            res=DoseResponseResult.empty(self.opt['MODEL'], data, {'opt_lb': opt_lb, 'opt_ub':opt_ub})
            if n>0: return res.trivial_solution()
            return res
        p=np.array([np.nan, np.nan, np.nan, np.nan])
        p, locks, lb, ub = self.fix_xlock(data, p, lb, ub, n)
        n_param=4-sum(locks)
        # I add 0.1 as a buffer. Sometimes, the top in A, bottom in B is shifted,
        # Say in A, even if all points are just above 0.5, the top maybe >1, so a 0.5 point is below 50%
        if (self.opt['MODEL']=='A' and data.Y.max()<min(opt_lb[1], 0.5-0.1, (opt_ub[0]+opt_lb[1])/2)) or \
           (self.opt['MODEL']=='B' and data.Y.max()<opt_lb[1]) or \
           (self.opt['MODEL']=='A' and data.Y.min()>max(opt_ub[0], 0.5+0.1, (opt_ub[0]+opt_lb[1])/2)):
           #(self.opt['MODEL']=='B' and (not locks[1]) and data.Y.max()<=lb[1]) or \
            #% || ((model==2) && min(ydata)>=max(ub(1),lb(2)))
            #%, may still want to set bottom at 1
            if self.opt['DEBUG']:
                util.info_msg('SmartDoseResponseFit.fit_entry(): obvious no fit case, signal too low.')
                print(["Ymin", data.Y.min(), "Ymax", data.Y.max(), "opt_lb", opt_lb, "opt_ub", opt_ub])
            # we nevertheless do a fit to generate auc_p and auc_r2
            res=drfit.fit(data, p, locks, lb, ub)
            return res.trivial_solution(inplace=True)
        if outlier and n>=MIN_POINTS_OUTLIER:
            res=drfit.fit_outlier(data, p, locks, lb, ub)
        else:
            res=drfit.fit(data, p, locks, lb, ub)
        # use Monte Carlo to estimate STDV
        #print(res._d)
        if res.has_fit():
            if res._d['r2']>=LEAST_R2: #R2 must be at least 0.5
                #% calculate theoretical Y values as the center for Monte Carlo bootstrapping
                res.monte_carlo(n_MC=n_MC)
                if not res._d['stable']:
                    if res._d['r2']>=ACCEPTABLE_UNSTABLE_R2:
                        debug('SmartDoseResponseFit.fit_entry(): Unstable solution, but R2 is good enough')
                    else:
                        debug('SmartDoseResponseFit.fit_entry(): Unstable solution and R2 is poor, set to trivial')
                        res.trivial_solution(inplace=True)
            else:
                debug('SmartDoseResponseFit.fit_entry(): R2 is too poor, set to trivial')
                res.trivial_solution(inplace=True)
        return res

    def fix_xlock(self, data, p, lb, ub, n):
        # guess initial p0
        locks=np.zeros(4, dtype=bool)
        for i in range(4):
            if not np.isinf(lb[i]) and not np.isinf(ub[i]) and lb[i]==ub[i]:
                locks[i]=True
                p[i]=lb[i]
        #try to lock bottom and/or top if there is not enough data points
        if n<=sum(~locks):
            if self.opt['DEBUG']:
                util.info_msg('SmartDoseResponseFit.fix_xlock(): not enough data points, extra locks imposed')
            if self.opt['MODEL']=='A':
                locks[0]=locks[1]=True
                p[0]=np.clip(0, lb[0], ub[0])
                p[1]=np.clip(1, lb[1], ub[1])
                if n<=sum(~locks):
                    locks[2]=True
                    p[3]=np.clip(-1, lb[3], ub[3])
            else:
                locks[0]=True
                p[0]=np.clip(1, lb[0], ub[0])
                #lock slope if there is too few data points
                #set constrain to the top
                y_max=data[~data.OUTLIER].Y.max()
                if np.isinf(ub[1]) or ub[1]>y_max:
                    ub[1]=max(lb[1], y_max)
                    locks[1]=True
                if n<=sum(~locks):
                    locks[3]=True
                    p[3]=np.clip(1, lb[3], ub[3])
        LOCK_FACTOR=1
        if self.opt['MODEL']=='B':
            max_y=data[~data.OUTLIER].Y.max()
            min_y=data[~data.OUTLIER].Y.min()
            if (ub[1]>max_y):
                ub[1]=p[1]=np.clip(max_y*LOCK_FACTOR, lb[1], ub[1])
                #ub[1]=p[1]=min(ub[1], max_y*LOCK_FACTOR)
                # constrain top to be within a small range
                if ub[1]<lb[1]:
                    lb[1]=ub[1]
                    locks[1]=True
                lb[1]=np.clip(max(ub[1]-1,0)*0.9+1, lb[1], ub[1])
                #locks[1]=True
            if min_y>ub[0]:
                p[0]=min(max(1, lb[0]), lb[1])
                locks[0]=True

        return (p, locks, lb, ub)

    @staticmethod
    def fit_a_cpd(X, Y, opt):
        n=len(X)
        data=pd.DataFrame(data={'X':X, 'Y':Y, 'ID':range(1,n+1)})
        data.dropna(inplace=True)
        data.sort_values('X', ascending=False, inplace=True)
        min_x, max_x, min_y, max_y=data.X.min(), data.X.max(), data.Y.min(), data.Y.max()
        opt['LB'][2]=np.log10(min_x)-2
        opt['UB'][2]=np.log10(max_x)+2
        if opt['MODEL']=='B':
            opt['UB'][1]=np.clip(min_y+(max_y-min_y)*1.5, opt['LB'][1], opt['UB'][1])
        #print(cpd, data, opt)
        #dr=DoseResponseFit(model=model)
        #if cpd=='29760660': break
        #if cpd!='31771567': continue
        debug("\n\nBEGIN***********************>> "+opt['CPD']+" <<**************")
        CurveFit.reset_count()
        dr=SmartDoseResponseFit(data, opt)
        res=dr.fit() #data.X.values, data.Y.values, None, None, opt['LB'], opt['UB'])
        res._d['cpd']=opt['CPD']
        res._d['count_fit']=CurveFit.get_count()
        dr.fix_data_for_report(res)
        debug("\n\nEND***********************>> "+opt['CPD']+" <<**************")
        debug(f"@@@ {opt['CPD']} @@@ {res._d['count_fit']} @@@")
        return res

class Bounds:

    @staticmethod
    def default(model_id):
        if model_id==4: #CYP Inhibition
            LB=[0, 1, -np.inf, -3]
            UB=[0, 1, np.inf, -1/3]
        elif model_id in (1,3): #A
            LB=[0, 0.5, -np.inf, -3]
            UB=[0.5, 1.5, np.inf, -1/3]
        elif model_id==2: #B
            LB=[0.5, 1.25, -np.inf, 1/3]
            UB=[1.25, np.inf, np.inf, 3]
        else:
            util.error_msg("Unsupported model id", model_id, "for bounds!")
        return [LB, UB]

    # users do not specify all bounds, fill the rest with default
    @staticmethod
    def fix_bounds(bounds, model_id):
        LB,UB=Bounds.default(model_id)
        lb,ub=bounds
        for i in range(len(lb)):
            j=i+1 if i==2 else i # user never specify IC50 constrains, so 3rd is for slope
            LB[j]=lb[i]
            UB[j]=ub[i]
        return [LB, UB]

def test_dr():
    X=np.array([10,3.3333333333333335,1.1111111111111112,0.3703703703703704,0.1234567901234568,0.0411522633744856,0.013717421124828532,0.004572473708276177,0.0015241579027587256,5.080526342529085E-4,1.6935087808430283E-4])
    X=np.log10(X)
    Y=np.array([0.2198,0.632,0.761,0.912,1.014,1.061,1.058,0.998,0.973,0.98,1.027])
    data=pd.DataFrame({'X':X, 'Y':Y})
    data['OUTLIER']=False
    dr=DoseResponseFit(model='A')

    locks=np.zeros(4, dtype=bool)
    lb=np.array([0,0.8,-np.inf,-3])
    ub=np.array([0.2,1.2,np.inf,-0.3])
    p0=np.array([0, 1, np.nan, np.nan])
    locks[0]=True
    locks[1]=True
    res=dr.fit(data, p0, locks, lb, ub)
    dr.monte_carlo(res)
    res.savefig('test.png')
    res.print()

if __name__=="__main__":

    test_dr()
