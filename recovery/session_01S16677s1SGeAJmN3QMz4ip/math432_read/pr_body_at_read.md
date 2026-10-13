Replace the three mutable action tags in `.github/workflows/axial-density-review.yml` with the exact commits behind their current releases. The workflow keeps its existing Python 3.11 setup, timeout, replay commands, upload settings and trigger paths.

| Action release | Verified commit |
| --- | --- |
| actions/checkout v7.0.1 | 3d3c42e5aac5ba805825da76410c181273ba90b1 |
| actions/setup-python v7.0.0 | 5fda3b95a4ea91299a34e894583c3862153e4b97 |
| actions/upload-artifact v7.0.2 | cf430e030ddbb5b0abf93d22962f4752f3646cd9 |

Current source head: `e27c3801fb160f415dfa5423f6a3d0b7f12928c8`, tree `8b1f65d66bb2e25345fdac7ce501ee3ff6c2fcaf`. Current Main base: `b3b6c0daa2237a4dc27910781053ece4b6543365`. The carry commit has ordered parents original own head `74cd387ec3481333c3f1a279431bd6db5d169408` and current Main `b3b6c0d…`; the branch update used expected-old head and a nonforce fast-forward.

Source validation: complete official tag and commit responses join all three exact references. Raw source, forward and inverse diffs, inert parsed YAML, and complete native Git trees were checked. The 1,699-byte candidate has SHA-256 `3f514545fe78c77e92e8eabbb9229131046c66cfe1b31b8437f02514eab0e3fe`. The current 2,501-entry tree differs from Main at exactly this one leaf. All four C183 ledger additions from PR429 and every other leaf are preserved. The native PR diff is exactly three replacements in one file. This source check does not review C183 mathematical or scientific claims.

Hosted validation: original head `74cd387…` and tested merge `6502e28ef4938d470e855dea65c31d436a7098cf` passed both attempt-1 runs and all four jobs: axial replay 38174466852/1; downstream replay, formal evidence and aggregate gate 38174467114/1. All four complete original logs were retained. The receipt binding was observed in logs; original artifact receipt/archive bytes were not acquired. These successes are retained for their original source composition.

Main advanced through PR429 at 21:06:31 UTC. The current carry composition requires fresh source-triggered checks before integration; the original successes do not transfer to the new tested merge `9c8ecc22749c146e45daf3ae0e21b0766636591c`. No rerun of an old attempt is requested.

Scope: routine workflow supply-chain pinning. Existing mathematical scope, pending independent review, scientific effect NONE, and 4D navigation HOLD remain as recorded. Source preparation ownership: Work Events WE1878, valid through 22:48:07 UTC; root is the sole publisher. Distinct source author/reviewer coordination is within OpenAI and carries organizational-independence credit 0.
