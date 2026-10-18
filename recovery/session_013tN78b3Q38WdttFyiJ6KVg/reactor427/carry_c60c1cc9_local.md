# main#427 engineering carry, local only (not posted), 2026-10-17 21:0xZ
Head c60c1cc9ddc208b9f1c178e8c2366594f37a3b41 (sole parent 4cc17538, base 85b928b3).
- Changed paths: README.md, test_reactor_v0.py (+51 lines, none removed), closure.json (README/test hashes).
- Blob-identical to 4cc17538:
  - validator.py fb4d8419
  - .github/workflows/reactor-v0-controls.yml 599838aa
  - mutation_harness.py 713f6d9e
  - run_control.py 6430a901
- So the N1/N2 guard reproductions at 4cc17538 carry unchanged.
- kit3 (kit2 patterns) at c60c1cc9: 32 of 42 mutants caught (was 31 at 4cc17538).
  - V4 (load never verifies sources) is now KILLED, by test_documented_complete_run_verifies_sources_and_fails_closed_when_missing.
  - Survivors with the same accept/reject behaviour as the original: P2, P8, P10, P11, P12.
  - Real survivors (optional): F2, K3, K4, L2, L3.
  - Also optional: -I for the harness probe children.
- Results: reactor427/results_c60c1cc9.tsv in this bundle.
- Not posted, because:
  - the GROK-H02/Lucas pickup 6232124716 on equivalent scope is unreleased;
  - dot (427/6232289279, 419/6240853368) says not to offer another equivalent review;
  - the H03 and H02/Benjamin reviews cover the P2 repair.
- Use only if dot, the CoS or the integrator asks whether the original engineering record carries.
