# Exact supplemental W8 input recovered from Drive

File: `c027 sweep core40.json`
Drive ID: `1cVez6js8qbeigoS8f1BKqlmNKZiKtgfj`
Raw bytes: 7,533
SHA-256: `0003756d4075bbfa881edfeac68d6cca7242c54cee2c551f2814fd57759b4c18`

This matches the exact input hash embedded in verify_lambda_grid_v3.py. The data were absent from the supplied SepOK outer archive at that hash; they were retrieved in this session, not fabricated or reconstructed from a report. Discovery used the known Drive file index, followed by current metadata and raw materialization.

Use the existing environment override `SWEEP_JSON_PATH` to point to this file. Do not edit the frozen source merely to change an absolute path. Run from a scratch copy because the source may create caches beside itself. The mutation driver still has its own hardcoded input reference and requires a separately disclosed portability change or faithfully reconstructed sandbox; do not alter it without a new identity.

The shipped v3 output is conditional on H-B3 and terminates at a failed kill gate. The recovered data repair input completeness, not those mathematical limitations.
