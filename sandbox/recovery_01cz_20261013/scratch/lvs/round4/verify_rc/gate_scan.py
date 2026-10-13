import sys, json, re
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3])
sys.path.insert(0, str(PKT))
import check
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
if isinstance(registered, dict) and 'registered' in registered:
    registered = registered['registered']
P = 'UniversalLaw.Spec.RN.RareClusterPoissonObstructionO'
print('HAS_PROP', P in registered)
try:
    print('REG_ENTRY', json.dumps(registered[P], ensure_ascii=False)[:800])
except Exception as e:
    print('REG_ENTRY_ERR', e)
text = FILE.read_text(encoding='utf-8')
info = check.scan_module(text, 'RNCluster', registered)
print('IMPORTS', info['imports'])
print('NAMESPACES', info['namespaces'])
print('AUX_THEOREMS', info.get('aux_theorems'), 'AUXDEFS', info.get('auxiliary_definitions'))
print('SET_OPTIONS', info.get('set_options'))
for t in info['theorems']:
    print('THEOREM', json.dumps(t, ensure_ascii=False))
code, docs = check.lex(text)
print('REFUSED', check.refused_tokens(code))
print('HASH', check.HASH_COMMAND.search(code))
# independent extra lexical checks on comment-stripped code
for pat in [r'fun\s*⟨', r'fun\s*\|', r'\bmatch\b', r'set_option', r'\bprivate\b', r'\bdef\b', r'\babbrev\b', r'\binstance\b', r'\bexample\b', r'\bvariable\b', r'\bsection\b']:
    ms = [code.count('\n',0,m.start())+1 for m in re.finditer(pat, code)]
    print('PATTERN', pat, ms)
print('SCAN_OK')
# negative controls
hdr = 'theorem rareClusterPoissonObstructionO (μ : ℝ → ℕ → ℝ) (a A : ℝ) :\n    UniversalLaw.Spec.RN.RareClusterPoissonObstructionO μ a A := by'
assert hdr in text, 'header text not found verbatim'
bad = text.replace(hdr, 'theorem rareClusterPoissonObstructionO (μ : ℝ → ℕ → ℝ) (a A : ℝ) (h : 0 < a) :\n    UniversalLaw.Spec.RN.RareClusterPoissonObstructionO μ a A := by')
try:
    check.scan_module(bad, 'RNCluster', registered); print('CONTROL_EXTRA_HYP: NOT REFUSED (checker weak)')
except Exception as e:
    print('CONTROL_EXTRA_HYP refused:', str(e)[:160])
bad2 = text.replace('  linarith\n', '  sorry\n', 1)
try:
    check.scan_module(bad2, 'RNCluster', registered); print('CONTROL_SORRY: NOT REFUSED')
except Exception as e:
    print('CONTROL_SORRY refused:', str(e)[:160])
bad3 = text.replace(hdr, 'theorem rareClusterPoissonObstructionO (μ : ℝ → ℕ → ℝ) (a A : ℝ) :\n    UniversalLaw.Spec.RN.RareClusterPoissonObstructionO μ a (2*A) := by')
try:
    check.scan_module(bad3, 'RNCluster', registered); print('CONTROL_NARROW: NOT REFUSED')
except Exception as e:
    print('CONTROL_NARROW refused:', str(e)[:160])
bad4 = text + '\n#print axioms foo\n'
try:
    check.scan_module(bad4, 'RNCluster', registered); print('CONTROL_HASH: NOT REFUSED')
except Exception as e:
    print('CONTROL_HASH refused:', str(e)[:160])
bad5 = text.replace('namespace UniversalLaw.Proofs.RNCluster\n', 'namespace UniversalLaw.Proofs.RNClusterX\n').replace('end UniversalLaw.Proofs.RNCluster\n','end UniversalLaw.Proofs.RNClusterX\n')
try:
    check.scan_module(bad5, 'RNCluster', registered); print('CONTROL_NS: NOT REFUSED')
except Exception as e:
    print('CONTROL_NS refused:', str(e)[:160])
