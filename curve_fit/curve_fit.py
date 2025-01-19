#!/usr/bin/env python
import pandas as pd
from scipy.optimize import least_squares
from scipy.linalg import svd
import scipy.stats as ss
import numpy as np
from pprint import pprint
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import copy

DEBUG=False
count=0

def debug(header, *w):
    if DEBUG:
        print("DEBUG> "+header)
        for s in w:
            pprint(s)
        print()

class CurveFit:
    """This class is to solve the bound minimization problem
    the curve_fit in scipy does not return the needed results
    the least_squares in scipy does not support locked parameters,
    and it is tricky to calculate standard deviation

    """

    ALPHA=0.1   # determine if model is trivial (compared to mean), if trival, we set success=False
    GOOD_R2=0.9 # good fit, no need to perform furthr outlier detection
    LEAST_OUTLIER_R2=0.6 #minimal acceptable R2 for an outlier solution, otherwise, ignore the outlier
    # outlier solution has to have p value less than this to consider better
    # is_better()
    OUTLIER_IS_BETTER_P=0.2

    def __init__(self, res_cls=None):
        """res_cls, return results as an instance of res_cls class
        """
        self.res_cls = Result if res_cls is None else res_cls

    def p_init(self, data, lb, ub):
        """Return intial guess p0, given data and constrains lb, ub"""
        pass

    @staticmethod
    def reset_count():
        global count
        count=0

    @staticmethod
    def inc_count():
        global count
        count+=1

    @staticmethod
    def get_count():
        global count
        return count

    def fit_outlier(self, data, p0=None, locks=None, lb=None, ub=None):
        """outlier identification routine if r2<GOOD_R2"""
        data=data.copy()
        res0=self.fit(data, p0, locks, lb, ub)
        if res0._d['success'] and res0._d['r2']>=self.GOOD_R2:
            # sufficiently good already
            return res0
        # leave-one-out outlier identification
        n=len(data)
        data.index=range(n)
        best=res0
        for i in range(n):
            data['OUTLIER']=False
            data.loc[i, 'OUTLIER']=True
            res=self.fit(data, p0, locks, lb, ub)
            if res._d['success'] and res._d['r2']>=self.LEAST_OUTLIER_R2 and res._d['r2']>best._d['r2']:
                best = res
        if best.is_better(res0, self.OUTLIER_IS_BETTER_P):
            debug('CurveFit.fit_outlier(): accept outlier solution.')
            return best
        else:
            return res0

    def fit(self, data, p0=None, locks=None, lb=None, ub=None):
        """all parameters must be numpy arrays
            data must have column X and Y, OUTLIER (defaults to False), others are ignored
        """
        # we make a copy to avoid any side effect
        # e.g., when fit is called by fit_outliers multiple times, our change to lb, ub won't affect others
        data=data.copy()
        n_param=0
        if p0 is not None: p0=p0.copy()
        if locks is not None: locks=locks.copy()
        if lb is not None: lb=lb.copy()
        if ub is not None: ub=ub.copy()
        kw={'method': 'trf'}
        if 'OUTLIER' in data.header() and data.OUTLIER.sum()>0:
            X, Y=data[~data.OUTLIER].X.values, data[~data.OUTLIER].Y.values
        else:
            X, Y=data.X.values, data.Y.values

        if p0 is None:
            p0=self.p_init(data, lb, ub)
        elif np.any(np.isnan(p0)):
            # any missing p0 will be filled with p_init results
            p=self.p_init(data, lb, ub)
            mask=np.isnan(p0)
            p0[mask]=p[mask]
        else:
            p0=np.array(p0)

        n=len(X)
        if locks is None:
            n_param=len(p0)
            locks=np.zeros(n_param, dtype=bool)
        else:
            locks=np.array(locks)
            n_param=np.sum(~locks)
        if len(X)<n_param:
            def nan(shape, value=np.nan):
                x=np.empty(shape)
                x[:]=value
                return x
            n2=len(p0)
            kw={'p':nan(n2), 'perr':nan(n2), 'SSreg':np.nan, 'r2':np.nan, 'df':np.nan, 'F':np.nan, 'SStot':np.nan, 'n':n, 'n_param':n_param, 'locks':locks, 'lb':lb, 'ub':ub, 'p0':p0, 'noise':np.nan, 'residue': np.nan, 'f':self.f}
            return self.res_cls.empty(data, kw)
        bounds=[-np.inf, np.inf]
        if lb is not None and ub is not None:
            for i in range(len(locks)):
                if not np.isinf(lb[i]) and lb[i]==ub[i]: locks[i]=True
        if lb is not None:
            lb=np.array(lb)
            bounds[0]=lb[~locks]

        if ub is not None:
            ub=np.array(ub)
            bounds[1]=ub[~locks]

        for i in range(len(p0)):
            if lb[i]==ub[i]: p0[i]=lb[i]

        def f_wrapper(X, Y, p0, locks):
            # p is only for unlocked part, p0 has all values
            def f(p): # turns self.f into a residue function
                p_=p0.copy()
                p_[~ locks]=p
                return self.f(X, p_)-Y
            return f

        if getattr(self, "jacobi_wrapper", None) is not None:
            kw['jac']=self.jacobi_wrapper(X, Y, p0, locks)
        self.inc_count() # track how many times curve fitting is performed
        debug(f'Counts> {count}')
        res=least_squares(f_wrapper(X, Y, p0, locks), p0[~locks], bounds=bounds, **kw)
        p_=np.array(res.x)
        success=res.success
        # copied from https://github.com/scipy/scipy/blob/v0.19.1/scipy/optimize/minpack.py#L502-L784`
        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        if pcov is None:
            # indeterminate covariance
            pcov = np.zeros((4,4), dtype=float)
        else:
            if n > 4:
                s_sq = 2*res.cost / (n - 4)
                pcov = pcov * s_sq
            else:
                pcov.fill(np.inf)
        perr_ = np.sqrt(np.diag(pcov))
        # p_, perr_ so far are only for unlocked parts
        p=p0.copy()
        perr=np.zeros(len(p))
        p[~locks]=p_
        perr[~locks]=perr_
        df=n-n_param

        data['Y_pred']=self.f(data.X.values, p)
        Y_pred=data[~data.OUTLIER].Y_pred.values
        residue=Y_pred-Y
        noise=np.std(residue)
        SStot=np.clip(np.var(Y, ddof=0)*n, 1e-10, np.inf)
        SSreg=np.sum(np.square(residue))
        r2=np.clip(1-SSreg/SStot, 0, 1)
        if r2>=self.GOOD_R2:
            success=True
        #F=SStot/SSreg
        #print(n, df, SStot, SSreg)
        if df>0:
            F=SSreg/SStot/df*(n-1)
            # check if trivial fit is better
            #if F/(n-1)*df<=1 or ss.f.cdf(F, n-1, df)<1-self.ALPHA:
            pval=ss.f.cdf(F, df, n-1)
            if not (F<1 and pval<self.ALPHA):
                # overfit
                debug(f'Model overfit: F={F:.4g}, pval={pval:.4g}')
                success=False
        else:
            F=np.nan
        # p_auc is a copy that will be used as an approximate, even if p is overwritten later,
        # e.g., p_auc curve can be used to provide a visual guide
        # p_auc can also be used to calculate AUC for trival solutions
        auc_residue=np.mean(np.abs(residue))
        res=self.res_cls({'success':success, 'data':data, 'p':p, 'perr':perr, 'SSreg':SSreg, 'r2':r2, 'df':df, 'F':F, 'SStot':SStot, 'n':n, 'n_param':n_param, 'locks':locks, 'lb':lb, 'ub':ub, 'p0':p0, 'noise':noise, 'residue': residue, 'f':self.f, 'auc_p':p.copy(), 'auc_perr':perr.copy(), 'auc_r2':r2, 'auc_residue':auc_residue})
        return res
        #self.res_cls({'success':success, 'data':data, 'p':p, 'perr':perr, 'SSreg':SSreg, 'r2':r2, 'df':df, 'F':F, 'SStot':SStot, 'n':n, 'n_param':n_param, 'locks':locks, 'lb':lb, 'ub':ub, 'p0':p0, 'noise':noise, 'f':self.f})

    def f(self, X, p):
        pass

    def monte_carlo(self, res, n_MC=8):
        """side effect:
            add res.stable, copy res.perr to res.perr_single, overwrite res.perr
        """
        p=res._d['p']
        if np.any(np.isnan(p)):
            res._d['nMC_success']=0
            return
        X=res._dd.X.values
        Y_pred=res._dd.Y_pred.values
        n=len(X)
        noise=res._d['noise']
        res._d['nMC_success']=0
        if n_MC==0: #still stable check
            res._d['perr_single']=res._d['perr'].copy()
            res._d['stable']=res._d['success']
            res._d['perr_mc']=[]
        else:
            perr=[]
            res._d['perr_single']=res._d['perr'].copy()
            for i in range(n_MC):
                debug('Monte Carlo loop: {}'.format(i))
                tmp_Y=Y_pred+np.random.randn(n)*noise
                data=res._dd.copy()
                data['Y']=tmp_Y
                tmp=self.fit(data, p, res._d['locks'], res._d['lb'], res._d['ub'])
                if tmp._d['success']:
                    perr.append(tmp._d['p'])
            res._d['perr_mc']=perr
            res._d['stable']=len(perr)>max(n_MC/4+1, 2)
            res._d['nMC_success']=len(perr)
            if res._d['stable']:
                res._d['perr']=np.std(np.vstack(perr), axis=0)

    def undo_monte_carlo(self, res):
        # undo and use the original fit
        if 'perr_mc' not in res._id: return # nothing to undo
        res._d['perr']=res._d['perr_single']
        del self.res._d['stable']
        del self.res._d['perr_mc']
        del res._d['perr_single']

class Result:

    def __init__(self, kw1, **kw2):
        """Expecting at members:
            data: dataframe of original data points
            data must have at least two columns: X, Y
            f: residue function
        """
        # to store result history
        self.previous=None
        self.previous_comment=""
        self.data={}

        for k,v in kw1.items():
            self.data[k]=v

        for k,v in kw2.items():
            self.data[k]=v

    @property
    def _d(self):
        """shortcut"""
        return self.data

    @property
    def _dd(self):
        """shortcut"""
        return self.data['data']

    def max_X(self):
        """The difference bwt max_X and np.max(X) is here we exclude outliers"""
        if 'OUTLIER' in self._dd.header():
            return self._dd.X[~ self._dd.OUTLIER].max()
        else:
            return self._dd.X.max()

    def min_X(self):
        if 'OUTLIER' in self._dd.header():
            return self._dd.X[~ self._dd.OUTLIER].min()
        else:
            return self._dd.X.min()

    def max_Y(self):
        if 'OUTLIER' in self._dd.header():
            return self._dd.Y[~ self._dd.OUTLIER].max()
        else:
            return self._dd.Y.max()

    def min_Y(self):
        if 'OUTLIER' in self._dd.header():
            return self._dd.Y[~ self._dd.OUTLIER].min()
        else:
            return self._dd.Y.min()

    def n(self):
        if 'OUTLIER' in self._dd.header():
            return np.sum(~ self._dd.OUTLIER)
        else:
            return len(self._dd)

    def clone(self):
        res=self.__class__(copy.deepcopy(self.data))
        return res

    def print(self):
        print("=== Result Object ===")
        if self.previous is not None:
            print()
            self.previous.print()
            print(self.previous_comment)
            print()
        for k,v in self._d.items():
            pprint([k, v])
        print()

    def has_fit(self):
        return self._d['success'] and not np.any(np.isnan(self._d['p']))

    def is_better(self, res2, GOOD_P=0.2):
        # change GOOD_P to 0.2 to be more aggresive in outlier detection

        # add special case handle when df is zero
        if self._d['df']==res2._d['df'] or (self._d['df']<=0 or res2._d['df']<=0):
            return self._d['r2']>res2._d['r2']
        F=self._d['SSreg']/res2._d['SSreg']*res2._d['df']/self._d['df']
        # F is mean_res(this model)/mean_res(res2 model)
        # F should be smaller
        debug(f"Check if better: F={F}")
        if F>1: return False
        #else:
        #    if (self._d['df']-res2._d['df'])>0:
        #        # simpler model
        #        if (self._d['SSreg']-res2._d['SSreg'])<=0: return True
        #    else:
        #        if (self._d['SSreg']-res2._d['SSreg'])>=0: return False
        #GOOD_P=0.1

        #F=self._d['SSreg']/res2._d['SSreg']*res2._d['df']/self._d['df']
        p=ss.f.cdf(F, self._d['df'], res2._d['df'])
        #print(p, F, res2._d['r2'], self._d['r2'],res2._d['df'],self._d['df'])
        debug(f"Check if better: p={p:.5g}")
        if (p > GOOD_P): return False
        return True

    def plot(self):
        sns.set()
        plt.figure(figsize=(4,3))
        plt.clf()
        X0=np.linspace(self._dd.X.min(), self._dd.X.max(), 50)
        if sum(np.isnan(X0))==0 and self._d['f'] is not None:
            if not np.array_equal(self._d['auc_p'], self._d['p']):
                Y0=self._d['f'](X0, self._d['auc_p'])
                sns.lineplot(x='X', y='Y', data=pd.DataFrame({'X':X0, 'Y':Y0}), color='#636363')
            if self.has_fit():
                Y0=self._d['f'](X0, self._d['p'])
                sns.lineplot(x='X', y='Y', data=pd.DataFrame({'X':X0, 'Y':Y0}), color='#e6550d')
        if len(self._dd):
            if 'OUTLIER' in self._dd.header() and self._dd.OUTLIER.sum()>0:
                data=self._dd[~self._dd.OUTLIER]
                sns.scatterplot(x='X', y='Y', data=data, color='#3182bd')
                data=self._dd[self._dd.OUTLIER]
                # markers: https://matplotlib.org/2.0.2/api/markers_api.html
                # to show markers, one has to provide a style column, here True become a numerical column
                g=sns.scatterplot(x='X', y='Y', data=data, color='#636363', style=True, markers='x')
                g.legend_.remove()
            else:
                sns.scatterplot(x='X', y='Y', data=self._dd, color='#3182bd')
        plt.tight_layout()

    def savefig(self, file_name):
        self.plot()
        plt.savefig(file_name, transparent=True)
        plt.close()

    def set_previous(self, result_obj, comment=''):
        self.previous_comment=comment
        self.previous=result_obj
        return self

    @classmethod
    def empty(cls, data, kw):
        """use classmethod as Result can be inherited,
        see https://stackoverflow.com/questions/4691925/in-python-how-to-get-name-of-a-class-inside-its-static-method
        """
        data={'data':data, 'success':False }
        if kw is not None:
            data.update(kw)
        res=cls(data)
        return res

class ExampleFit(CurveFit):

    def __init__(self):
        super(ExampleFit, self).__init__(res_cls=Result)

    def f(self, X, p):
        return p[0]*np.exp(-p[1]*X)

    def jacobi_wrapper(self, X, Y, p0, locks):
        # p is only for unlocked part, p0 has all values
        def jac(p):
            # J_{i,k} = \frac{\partial{f(x_i)}{\partial{p_k}}
            p_=np.array(p0)
            p_[~locks]=p
            J0=np.exp(-p_[1]*X)
            J1=-p[0]*X*J0
            J=np.vstack((J0, J1))
            return J[~locks].T
        return jac

    def p_init(self, data, lb, ub):
        """Initial guess"""
        X, Y=data.X.values, data.Y.values
        p0=np.zeros(2)
        if len(X)>=2:
            p0[0]=max(np.max(Y), 1e-5)
            p0[1]=np.log(max(np.min(Y), 1e-5)/p0[0])/np.max(X)
        p0=np.clip(p0, lb, ub)
        return p0

if __name__=="__main__":
    X=np.array([0, 10, 20, 50, 100])
    Y=np.array([400, 200, 50, 10, 2])
    X=X[:1]
    Y=Y[:1]
    data=pd.DataFrame({'X':X, 'Y':Y})
    data['OUTLIER']=False
    dr=ExampleFit()
    lb=np.array([0,1e-5])
    ub=np.array([np.inf,100])
    res=dr.fit(data, None, None, lb, ub)
    dr.monte_carlo(res)
    #res.savefig('test.png')
    res.print()


