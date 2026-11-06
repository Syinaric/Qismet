"""Presentation-ready figures and executed notebook, with hardware status explicit."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'/'constrained_v4'
fresh=pd.read_csv(OUT/'fresh_seed_summary.csv')
pilot=pd.read_csv(OUT/'pilot_p3_validation.csv')
means=fresh.mean(numeric_only=True)
gains={}
for prefix in ['', 'best32_']:
    delta=100*(fresh['qaoa_'+prefix+'expected_test_accuracy']-fresh['random_'+prefix+'expected_test_accuracy'])
    half=t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5)
    gains[prefix]={'mean_pp':float(delta.mean()),'interval_95_pp':[float(delta.mean()-half),float(delta.mean()+half)],
                   'positive_seeds':int((delta>0).sum())}
(OUT/'fresh_seed_effects.json').write_text(json.dumps(gains,indent=2))

fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,prefix,title in zip(axes,['','best32_'],['One accepted proposal','Best of 32 feasible proposals']):
    for _,r in fresh.iterrows():
        ax.plot([0,1],[100*r['random_'+prefix+'expected_test_accuracy'],100*r['qaoa_'+prefix+'expected_test_accuracy']],
                marker='o',alpha=.65,label=str(int(r.seed)))
    ax.set(xticks=[0,1],xticklabels=['Uniform random','Revised ideal QAOA'],ylabel='Expected retrained test accuracy (%)',title=title)
    ax.text(.03,.95,f"Mean gain: +{gains[prefix]['mean_pp']:.3f} percentage points",transform=ax.transAxes,va='top',fontsize=9)
axes[0].legend(title='Fresh initialization',fontsize=8)
fig.suptitle('Five fresh initializations — ideal simulation, not hardware',fontsize=13)
fig.tight_layout(); fig.savefig(OUT/'fresh_seed_performance.png',dpi=170); plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4))
for seed in fresh.seed:
    table=pd.read_csv(OUT/'fresh_seeds'/f'seed_{int(seed)}'/'all_mask_metrics.csv')
    audit=table.loc[~table['fit']]
    axes[0].scatter(audit.val_loss,audit.predicted_loss,s=15,alpha=.6,label=str(seed))
limits=axes[0].get_xlim(); axes[0].plot(limits,limits,'k--',lw=1)
axes[0].set(xlabel='Measured retrained validation loss',ylabel='Predicted QUBO loss',title=f'Held-out masks; mean R² = {means.audit_r2:.2f}')
axes[0].legend(fontsize=7)
axes[1].bar(['Uniform','Revised ideal QAOA'],[10,means.top10_probability*100],color=['#888888','#2563eb'])
axes[1].set(ylabel='Chance of sampling top 10% by QUBO score (%)',ylim=(0,100),title='Mean over five fresh initializations')
fig.tight_layout(); fig.savefig(OUT/'surrogate_and_sampling.png',dpi=170); plt.close(fig)

hardware_summary=OUT/'hardware'/'summary.json'
hardware_headline=''
hardware_text='Hardware follow-up is still running. The results above are ideal simulations. No new hardware improvement is being claimed yet.'
if hardware_summary.exists():
    h=json.loads(hardware_summary.read_text()); depths=pd.read_csv(OUT/'hardware'/'depth_comparison.csv')
    chosen=depths.loc[depths.layers==h['chosen_layers']].iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    axes[0].bar(depths.layers-.18,depths.top10_probability_conditional*100,width=.36,label='Among valid outputs')
    axes[0].bar(depths.layers+.18,depths.top10_probability_raw*100,width=.36,label='Among all raw shots')
    axes[0].axhline(10,color='black',ls='--',label='Uniform feasible: 10%')
    axes[0].set(xlabel='QAOA layers (0 = initial-state control)',ylabel='Probability of a top-10% QUBO mask (%)',ylim=(0,50),xticks=[0,1,2,3])
    axes[0].legend(fontsize=8)
    hardware_gains=100*(depths.best32_expected_test_accuracy-h['uniform_best32_expected_test_accuracy'])
    axes[1].bar(depths.layers,hardware_gains)
    axes[1].axhline(0,color='black',ls='--',label='Uniform random baseline')
    axes[1].set(xlabel='QAOA layers',ylabel='Best-of-32 test accuracy gain (pp)',xticks=[0,1,2,3])
    axes[1].legend(fontsize=8)
    fig.suptitle('Actual IBM hardware — all predeclared depths shown')
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(OUT/'hardware_performance.png',dpi=170); plt.close(fig)
    hardware_headline=f'''**Real hardware result:** the revised sampler achieved **{chosen.best32_expected_test_accuracy*100:.3f}% expected best-of-32 test accuracy**, versus **{h['uniform_best32_expected_test_accuracy']*100:.3f}%** for uniform random search, a **+{100*(chosen.best32_expected_test_accuracy-h['uniform_best32_expected_test_accuracy']):.3f} percentage-point** improvement at the same 32-raw-proposal budget. This is one model initialization and one finite measurement batch per depth; it is not a speedup claim.'''
    hardware_text=f'''Four real follow-up jobs ran on **{h['backend']}**, consuming **{h['additional_qpu_seconds']:g} QPU seconds**. All completed jobs across the project report **{h['total_project_qpu_seconds']:g} seconds** of QPU usage. The earlier queued Kingston attempt was cancelled before any QAOA result because of queue latency; its metrics currently report zero charge with pending usage status. Its 60-second reservation remains conservatively accounted. Both the partial attempt and all four Fez comparisons are preserved.

The chosen depth is **{h['chosen_layers']}**, selected using mean QUBO score before test evaluation. Its feasible fraction is **{chosen.feasible_fraction:.1%}**. Among feasible outputs it samples top-10% masks **{chosen.top10_probability_conditional:.1%}** of the time; counting every raw shot this is **{chosen.top10_probability_raw:.1%}**. Uniform feasible sampling is 10%.

Expected test accuracy per accepted hardware proposal is **{chosen.expected_test_accuracy*100:.2f}%**, versus **{h['uniform_expected_test_accuracy']*100:.2f}%** for uniform feasible masks. Expected best-of-32 accuracy, including the raw-shot acceptance process, is **{chosen.best32_expected_test_accuracy*100:.2f}%**, versus **{h['uniform_best32_expected_test_accuracy']*100:.2f}%** for random search. Probability of no feasible mask in 32 shots: **{chosen.no_feasible_in32_probability:.2g}**.

The single frozen 32-shot selection achieved **{h['selected_test_accuracy']*100:.2f}%** test accuracy. The corresponding seeded random selection achieved **{h['random_selected_test_accuracy']*100:.2f}%**. Classical exact QUBO search found the same selected mask (state 60) and therefore matches its 81.81% test accuracy. Single draws are not the primary evidence of distribution quality.

The one-layer circuit also showed a small positive best-of-32 gain. The three-layer circuit did not improve that metric, illustrating the tradeoff between ideal expressiveness and device noise. The two-layer circuit accepts only **{chosen.feasible_fraction:.1%}** of raw shots. Conditional sampling gains must be reported alongside this rejection overhead.

All four depths, job IDs, shot counts, validation loss and test accuracy are in `hardware/depth_comparison.csv`. Hardware uncertainty and drift are not captured by the initialization intervals below.'''
    uncertainty_path=OUT/'hardware'/'finite_shot_uncertainty.json'
    if uncertainty_path.exists():
        u=json.loads(uncertainty_path.read_text()); bounds=u['selected_best32_accuracy_gain_pp_interval_95']
        hardware_text+=f'''\n\nA multinomial bootstrap of these finite-shot batches, repeating QUBO-based depth selection, gives a sampling-only 95% interval of **[{bounds[0]:.3f}, {bounds[1]:.3f}] percentage points** for the expected best-of-32 gain. It excludes device drift, dataset uncertainty and earlier experiment-design choices. It is not evidence of universal advantage.'''

single=gains['']; best=gains['best32_']
report=f'''# Qismet: revised performance experiment

{hardware_headline}

The revised ideal QAOA sampler improved expected retrained test accuracy over uniform random pruning on **all five fresh initializations**. Mean gain per accepted proposal: **+{single['mean_pp']:.3f} percentage points**. Mean expected best-of-32 gain: **+{best['mean_pp']:.3f} percentage points**. These are ideal-simulation results; hardware results are stated separately below.

## What changed and why

The dataset remains Fashion-MNIST and the model remains 784→16→10. We fit the QUBO on **35 masks at the desired size**, rather than mixing multiple network sizes. Regularization is selected by cross-validation on those masks. Another 35 masks audit prediction fidelity. Only half the hidden neurons survive, reducing parameters from **12,730 to 6,370**.

The new circuit starts with four Bell pairs and uses paired RXX/RYY rotations on a ring. These XY mixers preserve the number of retained groups in an ideal circuit. Removing the large size penalty lets the cost phases represent quality differences directly. This is the constrained alternating-operator approach described by [Hadfield et al.](https://arxiv.org/abs/1709.03489).

Three QAOA layers use six angles, optimized on a CPU simulator with three fixed starts. Hardware executes frozen angles; it does not perform the earlier online SPSA loop. Both the earlier and revised methods are hybrids with classical preparation costs.

## Independent ideal-simulation follow-up

Fresh initializations: 101, 202, 303, 404, 505. Each network, surrogate, circuit and selected mask is frozen before test evaluation. The same Fashion-MNIST split is reused. This is an exploratory follow-up; fresh seeds check initialization dependence, not a new dataset.

| Metric | Uniform feasible masks | Revised ideal QAOA |
|---|---:|---:|
| Expected test accuracy per proposal | {means.random_expected_test_accuracy*100:.3f}% | {means.qaoa_expected_test_accuracy*100:.3f}% |
| Expected test accuracy, best of 32 | {means.random_best32_expected_test_accuracy*100:.3f}% | {means.qaoa_best32_expected_test_accuracy*100:.3f}% |
| Chance of top-10% QUBO mask | 10.00% | {means.top10_probability*100:.2f}% |

Dense mean accuracy: **{means.dense_test_accuracy*100:.2f}%**. The revised ideal best-of-32 subnetworks retain half the parameters with an average accuracy drop of **{100*(means.dense_test_accuracy-means.qaoa_best32_expected_test_accuracy):.2f} percentage points**.

The paired 95% t interval across five initializations is **[{single['interval_95_pp'][0]:.3f}, {single['interval_95_pp'][1]:.3f}] pp** for the one-proposal gain and **[{best['interval_95_pp'][0]:.3f}, {best['interval_95_pp'][1]:.3f}] pp** for best-of-32. These small-sample intervals condition on this dataset and ideal probabilities. They do not include dataset uncertainty, hardware noise or all researcher choices.

Expected metrics integrate the frozen sampling distribution against the actual retrained performance of all 70 valid masks. Best-of-32 selects by the QUBO, not by true test performance. This avoids presenting one lucky quantum draw against one unlucky random draw. It does not remove the simulator's optimization cost from the experiment.

![Fresh initializations](fresh_seed_performance.png)

## Is the QUBO more accurate?

The revised surrogate's mean held-out R² is **{means.audit_r2:.3f}** and mean absolute validation-loss error is **{means.audit_mae:.4f}** on the five fresh initializations. On the original five development seeds, fixed-size fitting improved mean R² from **−3.465 to {pilot.audit_r2.mean():.3f}**, and mean absolute error from **0.0357 to {pilot.audit_mae.mean():.4f}**, using the same original audit masks. Prediction is useful but still imperfect.

![Surrogate and sampling](surrogate_and_sampling.png)

## Actual hardware follow-up

{hardware_text}

## What the success does and does not establish

The ideal follow-up demonstrates better proposal quality and a small positive best-of-32 accuracy gain against the specified uniform-random baseline. It also improves held-out surrogate fidelity. All five fresh initialization results are reported.

Classical exact QUBO search remains cheap for 70 masks and is a stronger optimization reference than random sampling. This experiment does **not** demonstrate quantum speedup, superiority over all classical methods, or quantum advantage. A cost-aware comparison must include label generation, regression, simulator angle optimization, circuit preparation, QPU usage, queue latency and final retraining. Repeatedly evaluating candidates through a learned surrogate is a benefit available to classical optimizers too.

## Files

- `Revised_Walkthrough.ipynb` and `Results.html`: executed explanatory notebook and readable export.
- `fresh_seed_summary.csv`: all five independent ideal results.
- `fresh_seeds/seed_*/selection_frozen.json`: choices frozen before test evaluation.
- `hardware/depth_comparison.csv`: all real circuits and their quality metrics, when complete.
- `constrained_experiment.py`: actual revised circuit and IBM execution code.
'''
(OUT/'Results.md').write_text(report,encoding='utf-8')
md=nbformat.v4.new_markdown_cell; code=nbformat.v4.new_code_cell
cells=[md(report),code('''from pathlib import Path
import inspect,json
import pandas as pd
from IPython.display import display,Image
import constrained_experiment as experiment
BASE=Path('.').resolve()
experiment.verify()
print(inspect.getsource(experiment.build))
display(pd.read_csv(BASE/'fresh_seed_summary.csv')[['seed','audit_r2','qaoa_expected_test_accuracy','random_expected_test_accuracy','qaoa_best32_expected_test_accuracy','random_best32_expected_test_accuracy']])
'''),md('## Actual circuit\nThe initial Bell pairs and every XY mixer preserve four retained groups in the ideal circuit. Real-device errors can violate this constraint, so hardware acceptance must be measured.'),code('''import matplotlib.pyplot as plt
qc=experiment.build(seed=11,layers=1)
fig=qc.draw(output='mpl',fold=28)
fig.savefig(BASE/'constrained_circuit.png',dpi=130,bbox_inches='tight')
display(fig); plt.close(fig)
'''),md('## Saved hardware evidence'),code('''path=BASE/'hardware'/'depth_comparison.csv'
if path.exists():
    display(pd.read_csv(path))
    display(Image(filename=str(BASE/'hardware_performance.png')))
else:
    print('Hardware follow-up pending; the simulation results above are not hardware results.')
''')]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(OUT)}}).execute()
nbformat.write(nb,OUT/'Revised_Walkthrough.ipynb')
body,_=HTMLExporter().from_notebook_node(nb)
(OUT/'Results.html').write_text(body,encoding='utf-8')
print('Saved report, standalone figures, executed notebook and HTML. Hardware complete:',hardware_summary.exists())
