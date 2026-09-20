# C093 RELEASE AND AMENDMENT POLICY

`Q0_C092_FINAL_MASTER.md` remains the mathematical source of truth. C093 is a
release closeout, not a competing theorem master.

Future amendments have exactly four types:

- `ERRATUM`: wording, citation, or metadata; graph unchanged.
- `CORRECTION`: a live dependency fails; downstream invalidation required.
- `SHARPENING`: stronger constants or domain; original theorem remains valid.
- `GENERALIZATION`: a new model, volume, height, or limit; separate root.

Every amendment must record affected claim IDs, mathematical reason, source
hashes, artifact hashes, invalidated descendants, replacement claims, and
review status.

The successor projects Q0-REFEREE, Q0-SHARP, Q0-IV, Q0-B, and Q0-LLM have no
upstream edge into the C092 core. Their failure cannot invalidate C092. Their
success obtains a new root.

The C093 ZIP is immutable. Any byte change creates a new release ID and
manifest. Detached attestations may refer to the ZIP hash without creating a
manifest self-reference.
