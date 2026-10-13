"""Render the exact published refinement fixture, without selecting a fit window."""
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import run_refinement

EXPECTED='5ff75ce6517b236e4f6b3f03d764f9a136521001d8af2fb31e08d44781dc43a3'

def main():
    directory=Path(sys.argv[1]);raw=(directory/'observations.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=EXPECTED:raise ValueError('Changed observations require a new interpretation')
    d=json.loads(raw);run_refinement.verify(d)
    plt.rcParams.update({'font.size':10,'svg.hashsalt':'periodic-h0-refinement8','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    colors=['#9da7b1','#728899','#397998','#123e57']
    for g,color in zip(d['config']['grids'],colors):
        s=next(x for x in d['summary'] if x['grid']==g and x['method']=='cubical')
        axes[0].plot([np.sqrt(b['a']*b['b']) for b in s['bins']],[b['ratio_to_leading'] for b in s['bins']],marker='o',label=str(g)+'²',color=color)
    axes[0].axhline(1,color='#ac6333',ls='--',lw=1,label='leading-term reference')
    axes[0].set(xscale='log',xlabel='Lifetime (bin geometric midpoint)',ylabel='Measured bin mass / leading bin mass',title='Shortest bins keep changing')
    axes[0].legend(fontsize=8,ncol=2)
    for m,color in zip(run_refinement.METHODS,['#123e57','#b66a3c','#66916d']):
        rows=[r for r in d['comparisons'] if r['kind']=='grid' and r['from'][1]==m]
        axes[1].plot([r['to'][0] for r in rows],[max(p['finite_bottleneck'] for p in r['per_field']) for r in rows],marker='o',color=color,label=m)
    axes[1].set(xscale='log',yscale='log',xlabel='Finer grid side length',ylabel='Largest finite bottleneck distance\nacross the eight coupled fields',title='Diagrams approach each other under refinement')
    axes[1].set_xticks([256,512,1024],labels=['256','512','1024']);axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());axes[1].legend(fontsize=8)
    fig.suptitle('Resolution diagnostic · 8 existing fields · finite Fourier cutoff 24',fontsize=13)
    fig.savefig(directory/'refinement.svg',metadata={'Date':None});plt.close(fig)
    p=directory/'refinement.svg';p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    text=['# What finer grids change','',
          '**The shortest bins remain unresolved.** The added grids and diagonal controls show substantial reduction of the discretization effect, while several larger bins stabilize. This is not a confirmation or refutation of the continuum coefficient.','',
          '![Coupled grid refinement and diagram distances](refinement.svg)','',
          'Eight seeds (34000–34007) were selected as the first eight original pilot fields before this follow-up was run. Four nested grids and three filtrations give 96 coupled evaluations, not 96 independent fields. Every original bin and every positive finite interval is retained in [observations.json](observations.json); [RUN.json](RUN.json) records the actual execution.','',
          '## Cubical counts across the four grids','',
          '| Lifetime bin | 128² | 256² | 512² | 1024² |','|---|---:|---:|---:|---:|']
    cube=[s for s in d['summary'] if s['method']=='cubical']
    for j,b in enumerate(cube[0]['bins']):text.append('| ['+str(b['a'])+', '+str(b['b'])+') | '+' | '.join(str(s['bins'][j]['total']) for s in cube)+' |')
    text+=['','The first bin rises and then falls under refinement. Counts need not vary monotonically with resolution. A leading-term ratio close to one on one grid is therefore not reliable evidence of the asymptotic law.','',
           '## Finest-grid comparison','',
           '| Lifetime bin | Cubical | PL + diagonal | PL − diagonal | Cubical mean mass ± one field-level SE |','|---|---:|---:|---:|---:|']
    fine=[s for s in d['summary'] if s['grid']==1024]
    for j,b in enumerate(fine[0]['bins']):text.append(f"| [{b['a']}, {b['b']}) | "+' | '.join(str(s['bins'][j]['total']) for s in fine)+f" | {b['mean_mass']:.6g} ± {b['se_mass']:.3g} |")
    text+=['','All three methods agree in aggregate for bins starting at 0.004 at 1024². Several such counts already agree at 512², but this is eight-field empirical agreement, not a certified bin sandwich or an ensemble confidence result. The paired per-field changes are retained; aggregation can conceal compensating changes.','',
           '## Distance and error are different quantities','',
           '| Comparison | Largest finite-diagram bottleneck distance |','|---|---:|']
    for r in d['comparisons']:
        if r['kind']=='grid' and r['from'][1]=='cubical':text.append(f"| Cubical {r['from'][0]}² → {r['to'][0]}² | {max(p['finite_bottleneck'] for p in r['per_field']):.9g} |")
    text+=['','These are actual GUDHI distances between computed finite diagrams. They do not bound distance to an unknown continuum diagram. Essential classes are retained separately and excluded from these finite-diagram distances.','',
           'The [deterministic approximation note](../../APPROXIMATION.md) proves an O(h²) bound for the cubical and triangulated filtrations of a specified smooth finite realization, conditional on certified derivative and nodal-error inputs. The simple floating Fourier-Hessian diagnostic gives finest-grid endpoint-error estimates from '+f"{min(r['interpolation_epsilon_float'][-1] for r in d['diagnostics']):.6g} to {max(r['interpolation_epsilon_float'][-1] for r in d['diagnostics']):.6g}."+' These are conservative, not outward-rounded certificates. They cannot certify the shortest bins, and they do not enclose the infinite-field tail.','',
           'The next step is to evaluate sharper certified numerical bounds and a valid asymptotic remainder budget. [Confirmation readiness](../../CONFIRMATION_READINESS.md) states what remains missing. No held-out confirmation window or fitted exponent is selected here.','',
           '[Reproduce](../../README.md) · [Frozen design](../../REFINEMENT_PROTOCOL.md) · [Earlier pilot](../pilot32/RESULTS.md)','']
    (directory/'RESULTS.md').write_text('\n'.join(text))

if __name__=='__main__':main()
