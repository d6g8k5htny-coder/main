# GP-LB-REC-002 — Drive upload readback ledger v1.0

Date: 2026-08-04

Landing root: Drive folder `1S0hKEC9AX1_ILzlpVcDAUEd1hWbZhwWT`.

## Container readback

- Normalized export folder `161j01EjqLf1GxOvB4JRSVSk802hZjRw5`: 18 direct files; every connector-reported byte count equals the local source byte count.
- Original-identity folder `1N7gXXF8MZ-Qcc2z-zhHd8hAzgKu5p2EL`: 11 direct files; every connector-reported byte count equals the local source byte count.
- GP adjudication folder `1UZCWXmt1X-vvia7OzUWQajzQkvKLeXmA`: 6 direct files; every connector-reported byte count equals the local source byte count.
- Native audit mirror `13qsobJTrWf0b_m_56bfEO78so5gEl3kO7Lv7uMH6a2M`: readback shows one date chip, seven rich-link chips, eighteen list paragraphs, one title, one subtitle, and six H1 sections.

## Exact raw-byte readback

The following Drive files were downloaded after upload, decoded, and freshly SHA-256 hashed. Every result matches the source bytes:

| Object | Drive ID | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Outer delivery archive | `1Ykd9FO5oRUA5KoqOcDw64hwsQwXbs5ku` | 6,957,146 | `01b3f61ccf7fb4d1b8b33907cd0267f8d698d0b2c5f0ea9515b14d6e91bc1165` |
| Normalized LB export ZIP | `1wepNiM_rciN_45lLeDHfQfEZOhe2PIKr` | 142,929 | `a34d5ca5705934b2d5042e591b5cb1d59a5b7ad90a6a3b876dde56cf9764cb10` |
| KIMI-THM-023 normalized | `1gD9hXfmCVzEPVYBdt0Cx5KeziRqQin9D` | 14,936 | `61e14810c55f688561bacd5a62bfa3e0872e0672fa387161c60a0455be65f92a` |
| KIMI-AUD-023 normalized | `1rIXNLWoa7_vZo3b84ZyfgeNUOdH546ve` | 5,448 | `df6565acf9231b3ae41e0f5f27f55384d786119013bebc06b79bd87a8e70e874` |
| KIMI-AUD-024 normalized | `1UGKmNYHkD3UDd9UCowJRmGjYEjs2byfH` | 3,369 | `309aefb8bc5c6dab656cf2d6ddae0a570e4ba593a3e327a2008f785c97bdc47c` |
| GP reconciled handoff ZIP | `15gX850PNLflrffke9sOz0XqXaKpSF5S-` | 152,831 | `ffa615fbb9378366ff170693b00421f6b619884153cc07d8d4f92620c2b9cac1` |
| GP-LB-REC-001 raw adjudication | `1FtRtsCKPaJcmWDjKakUJjN3u8QEq8RZO` | 10,318 | `e875ea17313ea0d72ab623b3a45f6ef07ccafe8e8a7ca747a765db3d311bca07` |

## Register readback

- Carrier manifest rows 67-102: 36 appended rows; first and last ranges read back correctly.
- GP-REG Recent Activity rows 565-567: three LB-RATE entries read back with their source links.
- GP-REG Artifact Index rows 703-710: eight artifacts read back with their source links.
- GP-REG Open Questions row 15: `OQ-014`, WP re-derivation, read back as OPEN/P0.

Status effect: none. This ledger verifies landing and registration only; it does not admit the lower-bound theorem.
