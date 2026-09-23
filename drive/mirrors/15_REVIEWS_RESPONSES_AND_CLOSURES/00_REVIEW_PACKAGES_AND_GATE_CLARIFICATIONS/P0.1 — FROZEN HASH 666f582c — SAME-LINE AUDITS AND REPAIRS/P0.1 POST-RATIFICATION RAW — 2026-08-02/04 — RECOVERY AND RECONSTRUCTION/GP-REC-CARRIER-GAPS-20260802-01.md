# GP-REC-CARRIER-GAPS-20260802-01 — unrecovered post-R2 carrier ledger

The current workspace and every attached ZIP were searched by filename, archive member, content, size, and SHA-256 inventory.

Unrecovered exact carriers:

- verifier runs 15–17;
- R2.5 addendum;
- post-R2.5 regenerated manifest;
- original corrected G.9.2 carrier and `verify_g9_periodized_transfer_v1.py`.

Available state:

- verifier runs 1–7 only in the uploaded packages;
- valid V3 37-row manifest: 4,533 B / SHA `f1c686e5b5a7e13c329b01d7f7e1fd6e251c6fb6c24fe066e24c1dc18d7dfcb6`, but it predates R2.5;
- corrected G.9.1 canonical Drive set: note `19Vwfm7SBk-4V_RIwMBxXepjcw6NUoyMy`, verifier `18nGSc8G3DhKCtuk86VVILGQyYNTLlw7-`, transcript `1vZBvlQxSWOTrWI8t4hdIbIxLxJQdAUY5`;
- recovered V3.4 archive: `13QS9QQHxSiuLPIPSh9o5plS5HkClwSmz`;
- AO48-DER-037 identifies the missing G.9.2 verifier by name and records its receipts, but did not re-execute it.

No replacement is labelled as a recovered original. Any recreated G.9.2 note/script must carry a new ID, fresh hashes, and an explicit reconstruction disclaimer.

