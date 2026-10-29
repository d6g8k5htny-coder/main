"""Render the fixed pilot observations; no fitted window or parameter estimation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_pilot import verify_observations

# The interpretation below belongs to this exact pilot, not any future run.
PILOT_SHA256='fa310ea8a259fbf58413bd49713260c7b7c73f9c3c9387426c10947e487000b4'

def render(folder,png=None):
    raw=(folder/'observations.json').read_bytes();d=json.loads(raw);verify_observations(d)
    if d['config']!=json.loads(Path(__file__).with_name('pilot_config.json').read_text()):
        raise ValueError('This report is specific to the frozen pilot32 design')
    if hashlib.sha256(raw).hexdigest()!=PILOT_SHA256:
        raise ValueError('Interpretation is bound to the published pilot observations; review changed outcomes first')
    lookup={(r['cutoff'],r['grid']):r for r in d['summary']}
    grids=d['config']['grids'];cutoffs=d['config']['cutoffs'];cutoff=max(cutoffs)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.hashsalt':'periodic-h0-pilot32'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for grid,color in zip(grids,('#8795a1','#3286a3','#203d57')):
        bins=lookup[cutoff,grid]['bins'];x=[math.sqrt(b['a']*b['b']) for b in bins]
        axes[0].errorbar(x,[b['ratio_to_leading'] for b in bins],
            yerr=[b['se_mass']/b['leading_prediction'] for b in bins],label=f'{grid} × {grid}',color=color,marker='o',capsize=3)
    for c,color in zip(cutoffs,('#d18039','#203d57')):
        bins=lookup[c,max(grids)]['bins'];x=[math.sqrt(b['a']*b['b']) for b in bins]
        axes[1].errorbar(x,[b['ratio_to_leading'] for b in bins],
            yerr=[b['se_mass']/b['leading_prediction'] for b in bins],label=f'Mode cutoff {c}',color=color,marker='o',capsize=3)
    for ax in axes:
        ax.set_xscale('log');ax.axhline(1,color='#a63b32',ls='--',lw=1,label='Leading-law reference')
        ax.set_xlabel('Lifetime bin (geometric center for display)');ax.set_ylabel('Measured bin mass / leading prediction')
        ax.grid(alpha=.15);ax.legend(fontsize=9);ax.set_ylim(bottom=0)
    axes[0].set_title('Grid sensitivity • cutoff 24')
    axes[1].set_title('Spectral sensitivity • 256 × 256 grid')
    fig.suptitle('Short-bar comparison is unresolved at these resolutions',fontsize=15)
    svg=folder/'comparison.svg'
    fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    if png:fig.savefig(png,dpi=150)
    plt.close(fig)
    rows=[]
    for j,b in enumerate(lookup[cutoff,max(grids)]['bins']):
        counts=[lookup[cutoff,g]['bins'][j]['total'] for g in grids]
        se=b['se_mass']/b['leading_prediction']
        rows.append(f"| [{b['a']:g}, {b['b']:g}) | {' | '.join(map(str,counts))} | {b['ratio_to_leading']:.3f} ± {se:.3f} |")
    cov=d['covariance_empirical'];z=[(a-b)/s for a,b,s in zip(cov['mean_products'],cov['finite_target'],cov['se_products'])]
    report='''# What the first periodic H₀ pilot shows

**The comparison is inconclusive at the tested resolutions.** The short-bar counts are strongly grid-sensitive. This pilot supplies reproducible observations and exposes a numerical bottleneck; it neither confirms the asymptotic coefficient nor refutes the continuum theorem.

The fixed design uses 32 independent planar fields on a side-24 torus, each evaluated on grids 64², 128² and 256² with square Fourier cutoffs 12 and 24. These are 192 coupled evaluations, **not 192 independent fields**. All predeclared bins and configurations are retained. There was no fitted slope, chosen confirmation window or discarded zero-count realization.

![Grid and spectral comparisons, with strong short-bin drift](comparison.svg)

Error bars are one standard error of the mean across independent fields, divided by the integrated leading prediction. They are descriptive marginal sampling errors, not simultaneous confidence bands, grid-error bounds or theorem error bars. Plot centers are display positions only; predictions integrate over each whole bin.

## Counts and the scale of the grid effect

The table uses spectral cutoff 24. Counts are totals across the same 32 fields. The final column is the finest-grid mass divided by `(3c/2)(b^(2/3)−a^(2/3))`, with `c≈0.07340691930603427` from the separate source coefficient enclosure. This floating-point comparison does not preserve that enclosure's twenty-digit precision.

| Lifetime bin | 64² count | 128² count | 256² count | 256² ratio ± one SE |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

In the first bin, refinement changes the count from 11 to 56 to 139. For the same fields, the 128²→256² difference in per-area mass is about 0.004503 with a paired standard error of 0.000731. Sampling more fields at the same grids would not remove this observed resolution sensitivity. The larger bins appear less grid-sensitive in this pilot, but their finite lifetimes are not known to lie inside the theorem's unevaluated asymptotic window.

The fixed cutoff-12 series retains about 0.9979384 of the target variance; its omitted squared second-derivative-frequency sum through mode64 is about 0.3976. Cutoff24 leaves only about 2.52×10⁻¹⁰ omitted variance through mode64, but this is a floating-point spectral diagnostic, not a uniform field approximation certificate. Low variance error alone can conceal appreciable derivative error. Both cutoffs show short-bin grid drift.

## What was checked

GUDHI's actual periodic vertex filtration agrees with an independent descending union-find oracle on small known and random/tied landscapes. The tests distinguish periodic gluing, elder selection, positive finite versus essential bars, translations, amplitude scaling, half-open bins and volume normalization. A deterministic basis-response covariance check of the FFT agrees with a separately derived finite cosine sum; wrong spectrum and missing conjugate variance mutants fail. The reporting audit recomputes every saved count and summary from retained intervals, every calibration aggregate from retained samples, and the covariance targets, spectral diagnostics and deterministic controls.

A separate 1,024-field calibration sample gives standardized product-mean discrepancies '''+', '.join(f'{v:.3f}' for v in z)+''' at the four declared lags. These are diagnostics with marginal estimated standard errors, not a multiple-testing guarantee. Every calibration point sample is retained.

## Next research decision

Do not fit the smallest bins yet. First study spatial interpolation/filtration approximation and identify a lifetime range stable under further coupled refinement; then predeclare a separate held-out confirmation design. A certified comparison additionally needs a quantitative continuum error bound and usable remainder constants/range. Near-diagonal unmatched features and bin crossings must be accounted for. This pilot contains no typed-contact estimator, so it cannot measure a candidate-minus-elder defect.

## Exact output and reproduction

All finite positive intervals, essential births, zero-length counts, per-realization bins, paired differences and calibration samples are in [observations.json](observations.json). [RUN.json](RUN.json) records the executed code/config identities, runtime and versions. Follow the [reproduction instructions](../../README.md). Original mathematical sources remain in the [source manifest](../../../../docs/research-translation/20260930/SOURCES.json); this run changes no scientific status.

Observation SHA256: `'''+hashlib.sha256(raw).hexdigest()+'`.\n'
    (folder/'RESULTS.md').write_text(report)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--png',type=Path)
    args=p.parse_args();render(args.folder,args.png)
