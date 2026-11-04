from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_plots(result,out):
    rows=result['rows']; methods=sorted(set(r['method'] for r in rows)); colors=plt.cm.tab10(np.linspace(0,1,len(methods)))
    fig,ax=plt.subplots(figsize=(7,5))
    for name,color in zip(methods,colors):
        xs=[];ys=[]
        for d in (.05,.1,.2):
            rr=[r for r in rows if r['method']==name and r['delta']==d]
            xs.append(d);ys.append(np.mean([r['risk'] for r in rr]))
        ax.plot(xs,ys,'o-',label=name,color=color)
    ax.plot([0,.25],[0,.25],'k--',alpha=.5,label='requested = realized')
    ax.set(xlabel='Requested risk budget Δ',ylabel='Independent execution risk',title='Risk control')
    ax.legend(ncol=2);fig.tight_layout();fig.savefig(out/'requested_vs_actual.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    c=result['calibration'];ax.scatter([r['predicted'] for r in c],[r['actual'] for r in c],s=16,alpha=.7)
    ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Gaussian union bound',ylabel='Independent Monte Carlo risk',title='Surrogate calibration')
    fig.tight_layout();fig.savefig(out/'calibration.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for name,color in zip(methods,colors):
        rr=[r for r in rows if r['method']==name and r['delta']==.1]
        ax.scatter(np.mean([r['cost'] for r in rr]),np.mean([r['risk'] for r in rr]),label=name,color=color,s=65)
    ax.set(xlabel='Mean executed path length',ylabel='Mean execution risk',title='Safety and cost at Δ = 0.10');ax.legend(ncol=2)
    fig.tight_layout();fig.savefig(out/'safety_cost.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(methods,[np.mean([r['seconds'] for r in rows if r['method']==m]) for m in methods],color=colors)
    ax.set(ylabel='Planning seconds per task batch',title='Planning time')
    fig.tight_layout();fig.savefig(out/'runtime.png',dpi=160);plt.close(fig)
