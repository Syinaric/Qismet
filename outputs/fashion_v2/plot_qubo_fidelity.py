"""Presentation plots of held-out QUBO prediction fidelity; no model refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE/'approximation_check'
OUT.mkdir(exist_ok=True)
seeds=[11,22,33,44,55]
tables=[]; metrics=[]
for seed in seeds:
    folder=BASE/'results'/f'seed_{seed}'
    labels=pd.read_csv(folder/'labels.csv',dtype={'mask':str})
    audit=labels.query("split == 'audit'").copy()
    coef=np.load(folder/'surrogates.npz')
    masks=np.array([[int(v) for v in m] for m in audit['mask']])
    predictions=(float(coef['quadratic_constant'])+masks@coef['quadratic_linear']+
                 np.einsum('bi,ij,bj->b',masks,coef['quadratic_pairs'],masks))
    np.testing.assert_allclose(predictions,audit.Quadratic_prediction,atol=1e-12)
    audit['predicted_loss']=predictions
    audit['pruning']=100*(1-audit.keep_groups/8)
    for keep in [4,2]:
        part=audit[audit.keep_groups==keep].copy()
        assert len(part)==(35 if keep==4 else 14)
        metrics.append(dict(seed=seed,pruning=100*(1-keep/8),n=len(part),
            mae=mean_absolute_error(part.val_loss,part.predicted_loss),
            r2=r2_score(part.val_loss,part.predicted_loss),
            spearman=spearmanr(part.val_loss,part.predicted_loss).statistic,
            bias=float((part.predicted_loss-part.val_loss).mean())))
        for col,name in [('val_loss','actual_rank_percentile'),('predicted_loss','predicted_rank_percentile')]:
            part[name]=(rankdata(part[col])-1)/(len(part)-1)*100
        tables.append(part)
data=pd.concat(tables,ignore_index=True)
metric=pd.DataFrame(metrics)
summary=metric.groupby('pruning')[['mae','r2','spearman','bias']].mean()
data.to_csv(OUT/'heldout_predictions.csv',index=False)
metric.to_csv(OUT/'per_seed_metrics.csv',index=False)
summary.to_csv(OUT/'mean_metrics.csv')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
colors=['#2563eb','#f97316','#059669','#9333ea','#db2777']

for mode in ['loss','ranking']:
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,pruning in zip(axes,[50,75]):
        part=data[data.pruning==pruning]
        x='val_loss' if mode=='loss' else 'actual_rank_percentile'
        y='predicted_loss' if mode=='loss' else 'predicted_rank_percentile'
        if mode=='loss':
            lo=min(part[x].min(),part[y].min()); hi=max(part[x].max(),part[y].max())
            pad=(hi-lo)*.06; lo-=pad; hi+=pad
        else: lo,hi=-4,104
        ax.plot([lo,hi],[lo,hi],color='#475569',ls='--',lw=1.5,zorder=0)
        for seed,color in zip(seeds,colors):
            p=part[part.seed==seed]
            ax.scatter(p[x],p[y],s=29,c=color,label=f'Seed {seed}',alpha=.75,edgecolors='white',linewidths=.3)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),title=f'{pruning}% neuron pruning · {len(part)} held-out cases')
        ax.set_aspect('equal',adjustable='box')
        ax.grid(alpha=.15)
        m=summary.loc[pruning]
        if mode=='loss':
            ax.set_xlabel('Measured validation loss after retraining')
            ax.set_ylabel('QUBO-predicted validation loss')
            annotation=f"Mean MAE: {m.mae:.3f}   |   Mean R²: {m.r2:.2f}"
        else:
            ax.set_xlabel('Actual loss rank within each seed (0 = best)')
            ax.set_ylabel('Predicted loss rank within each seed (0 = best)')
            annotation=f'Mean Spearman rank correlation: {m.spearman:.3f}'
        ax.text(.5,-.23,annotation,transform=ax.transAxes,ha='center',va='top',fontsize=11)
    title='Does the QUBO predict retraining loss?' if mode=='loss' else 'Does the QUBO rank subnetworks correctly?'
    fig.suptitle(title,fontsize=19,y=.98)
    fig.text(.5,.92,'Fashion-MNIST · masks excluded from fitting and regularization selection · dashed line = perfect agreement',
             ha='center',fontsize=10,color='#475569')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=5,loc='lower center',bbox_to_anchor=(.5,.055),frameon=False)
    fig.text(.5,.02,'Five original initialization seeds on one fixed data split. Metrics are averaged across seeds; test data is unused.',
             ha='center',fontsize=9,color='#475569')
    fig.subplots_adjust(left=.07,right=.98,top=.85,bottom=.26,wspace=.25)
    name='qubo_loss_fidelity' if mode=='loss' else 'qubo_ranking_fidelity'
    fig.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

note='''# Does the quadratic approximation match actual retraining loss?

These plots reuse the frozen original Fashion-MNIST surrogates. No fitting, coefficient tuning,
or neural-network retraining was performed to create them. All plotted masks were excluded
from surrogate fitting and CV. There are 35 audit masks at 50% pruning and 14 at 75%, for each
of five initializations. Each label is validation cross-entropy after 20 epochs from surviving
original weights, using the original fixed training protocol. The final test set is not used.

In qubo_loss_fidelity.png, each dot is one mask/initialization pair. The x coordinate is actual
retraining loss and the y coordinate is predicted loss. Closeness to the dashed identity line
indicates accurate loss prediction. Metrics are calculated per initialization, then averaged;
they are not pooled estimates or uncertainty intervals.

| Neuron pruning | Mean absolute error (CE units) | Mean R² | Mean rank correlation |
|---|---:|---:|---:|
'''
for pruning,row in summary.iterrows():
    note+=f'| {pruning:.0f}% | {row.mae:.3f} | {row.r2:.2f} | {row.spearman:.3f} |\n'
note+='''
The current QUBO is not a reliably close approximation of absolute retraining loss.
Negative per-seed R² means larger squared error than a constant equal to that audit set's
mean measured loss; this is a diagnostic baseline, not a proposed deployable predictor.
MAE is in cross-entropy units, not accuracy percentage points.

In qubo_ranking_fidelity.png, ranks are calculated within each initialization before plotting.
The model retains moderate ranking information at 50% pruning and stronger ranking information
at 75%. A score can rank candidates usefully while systematically misestimating their loss.
That motivates surrogate-guided search, but does not establish that QAOA is a superior solver.
The original classical best-of-random selection matched or exceeded QAOA's mean test accuracy.
Linear-only ranking was also slightly better on average in the existing ablation.

Suggested presentation statement: "Our quadratic surrogate captures some retrainability ranking,
especially at 75% pruning (mean held-out Spearman 0.736), but absolute loss prediction remains
poor. This motivates improving the surrogate and comparing quantum and classical searches;
it does not demonstrate quantum advantage."
'''
(OUT/'README.md').write_text(note,encoding='utf-8')
print(summary.to_string())
print('Verified frozen coefficients against all 245 held-out predictions; wrote PNG/SVG plots and CSVs.')
