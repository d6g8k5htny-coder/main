# Portable RN Gaussian moment certificates — candidate v1

This is an author-side, unreviewed implementation for exact generic Gaussian
algebra. It changes no claim, premise, grade, gate, scientific status or source
register. Organizational independence credit is zero. A successful replay does
not identify an RN field law, prove a spatial cover, perform interval Cholesky
conditioning, establish weighted Palm composition or close RN-UNIF.

`research/rn/certificate.py` produces and verifies data-only JSON certificates.
`tools/rn_certificate.py` is the command-line entry point. Neither interface
evaluates certificate code, imports a payload-specified module, follows a
payload-specified path, or fetches a payload URL. It uses Python's standard
library and the repository's existing exact `research.interval.Interval`.

## Statement and proof carried

The supported statement is: for every fixed choice of interval-enclosed
intercept `a`, slope `b` and symmetric **PSD** covariance `S`, with covariance
constant in the mark, let `X(t) ~ N(a+b*t, S)` and
`D = X_xx X_yy - X_xy^2`. The supplied nonnegative rational `C` bounds
`E[D^k]` throughout the complete closed rational mark interval, for `k=2` or
`k=4`. Singular Gaussian laws and zero-width mark intervals are included.
Singleton parameter intervals represent fixed rational laws. No independence
between the three coordinates or between uncertain parameters is assumed.

The JSON has exactly four top-level fields:

| Field | Contents and checks |
|---|---|
| `schema` | Exact version `RN_GAUSSIAN_MOMENT_CERTIFICATE_V1`; unknown versions are refused. |
| `claim` | Context, all intercept/slope/covariance interval endpoints, common mean/covariance law identifiers and determinant degree. |
| `proof` | Enclosed moment-polynomial coefficients, covariance feasibility state/witness, a complete ordered mark partition with every leaf's Bernstein coefficients, and claimed upper bound `C`. |
| `scope` | Exact fixed fields: authority `NONE`, zero independence credit and false field/spatial/prize/status-change flags. |

The format intentionally has no executable witness field. Rationals use reduced canonical strings:
`"0"`, `"-1"`, `"3/4"`; floats, booleans, decimal strings, `"2/2"`, `"-0"`,
nonpositive denominators and leading zeros are refused as rational encodings.
Intervals are two-element arrays of ordered endpoints. Unknown fields are
refused rather than silently ignored.

The context preserves the pinned RN5 source ID/hash, dimension two,
coordinate order `(xx, yy, xy)`, explicit normalization, conditioning-law
identifier, complete mark domain and candidate evidence tier. It uses the
existing `Context` checks. This binding records declared provenance; a correct
RN5 source hash does not prove that caller-supplied numbers describe a field.

Replay derives every moment coefficient again. For multi-index `alpha`, choose
the first nonzero coordinate `i`, and write `beta=alpha-e_i`. The Gaussian
integration-by-parts identity gives

`M_alpha(t) = (a_i+b_i*t) M_beta(t) + sum_j beta_j S_ij M_(beta-e_j)(t)`.

The checker fills all moments in increasing total degree through `2k`, then
forms `sum_j (-1)^j binom(k,j) M_(k-j,k-j,2j)`. It checks equality with every
supplied coefficient enclosure; trusting a supplied polynomial alone would
not establish that it bounds a Gaussian moment. The coefficient inclusion
argument, including singular covariance and repeated parameters, is in
[RN_INTERVAL_FAMILIES.md](RN_INTERVAL_FAMILIES.md).

For each mark leaf `[l,h]`, substitute `t=l+(h-l)u` and recompute all Bernstein
coefficient intervals. A Bernstein basis is nonnegative and sums to one on
`[0,1]`, so the largest coefficient upper endpoint bounds every point in the
leaf. The checker verifies a sorted chain from the original left endpoint to
the original right endpoint, with no gaps, overlaps, missing children or
interior degenerate leaves. The maximum of the leaf caps is `U`; it verifies
`C >= U >= 0`. This is a complete interval argument, not point sampling.

Covariance feasibility is a separate predicate. Replay recomputes the exact
singleton test or sufficient symmetric diagonal-dominance test, its margins,
and the midpoint witness. `PSD_MEMBERS_ONLY` remains a conditional quantifier;
a failed sufficient test is not proof of a non-PSD member. A failed midpoint
test leaves nonemptiness unestablished. The certificate must preserve these
distinctions even when the moment enclosure itself replays successfully.

## Results, diagnostics and identity

`verify_bytes(data, source_bytes=None, expected_context=None,
max_operations=200000)` returns a data-only result. `expected_context` uses
the JSON representation of the context: rational domain strings and an order
array. These checks have distinct meanings:

| Result | Meaning |
|---|---|
| `certificate_valid` | Exact replay succeeded for the declared generic law family and scope. It is not a formal proof-assistant verdict or field identification. |
| `source_bytes_match` | `true`/`false` if external source bytes were supplied; otherwise `null`/unchecked. It checks byte identity, not provenance authenticity. |
| `expected_context_match` | `true`/`false` if the caller supplied a valid expected context; otherwise `null`/unchecked. It checks declared applicability, not field truth. |
| `requested_checks_passed` | Every optional check actually requested succeeded. With neither requested, this concerns mathematical replay only; nulls remain visible. |
| `outcome` | `VALID`, `REJECTED`, or `INCONCLUSIVE` after a resource limit. No non-valid outcome authorizes reuse. |

The bound diagnostic reports `replay_upper=U` and
`certified_lower_bound_on_slack=C-U`. If replay succeeds, `C-U` is a certified
lower bound on the mathematical slack `C-sup E[D^k]`; it need not equal that
slack. If it is negative, this certificate fails to prove the requested bound;
that does not establish that an actual law/mark violates `C`. The diagnostic
kind explicitly says `SUFFICIENT_BOUND_SLACK_NOT_ACTUAL_OPTIMUM`.

`claim_sha256` hashes the canonical serialized claim, including all numerical
inputs and declared context. `certificate_sha256` additionally binds the
specific proof data, scope and format version. They identify representations,
not semantic mathematical equivalence, organizational independence or belief
at a historical time. No retraction or canonical-status log is introduced.

## Producer, checker and shared trust

The producer calls the existing memoized `FamilyMomentEngine`. Replay uses a
separate iterative recurrence and never calls `MomentEngine.moment` or
`MomentEngine.determinant`. A test replaces both producer methods with failures
and requires replay still to succeed. This is meaningful implementation
separation, but both use the same Gaussian identity.

The trusted code still includes Python/Fraction, the existing exact Interval
implementation, family input/PSD validation, context validation, JSON/parser
code, and the certificate's Bernstein conversion shared by producer and
checker. There is no claim of independently authored checkers or formal
verification. Tests therefore challenge semantics through the separately
written direct expansion in independent standard normals, closed-form
variance-family moments, and the interior maximum of `(1-t^2)^4`. Such tests
can expose mistakes; agreement over their finite cases is not a universal
proof of implementation correctness.

## Resource admission and reproducible pilots

Version 1 admits at most 524,288 input bytes, JSON nesting depth 16, 64
subdivision leaves, and degree at most eight in the moment polynomial.
Each input rational numerator/denominator has at most 96 decimal digits;
proof rationals have at most 2,048 digits. Instrumented rational operations
stop beyond 8,192-bit intermediates or a caller-selected budget no larger than
200,000 operations. Duplicate keys, non-finite numbers and JSON float literals
are rejected before mathematical work. File reads for this CLI are bounded.

The instrumented operation count is not a complete time/bit-cost model:
parsing, hashing, covariance feasibility, integer binomial/power preparation
and some bookkeeping are not charged. Their sizes and loop counts are bounded
by the admission rules. There is no universal one-second verifier promise or
performance claim. A resource refusal reports `INCONCLUSIVE` and never sets
`certificate_valid`. Producer resource failure raises `ResourceLimit` without
writing a partial artifact.

Three explicit synthetic candidates are committed:

| Pilot | Meaning |
|---|---|
| `variance2_certificate_v1.json` | Centered independent `X,Y~N(0,1)`, `Z~N(0,s)`, `s in [0,1]`: `E[D^2]=1+3s^2 <= 4`. |
| `variance4_certificate_v1.json` | The same family: `E[D^4]=9+18s^2+105s^4 <= 132`. |
| `affine4_certificate_v1.json` | Uncertain affine means/slopes and nonzero correlated covariance intervals on `[-1,1]`; a generic uniform fourth-moment bound. |

For the variance examples only, a separate closed-form argument proves that
the endpoint `s=1` attains 4 and 132. The replay interface does not generalize
this optimality fact to its other certificates. The affine example makes no
claim of a tight or useful RN field constant.

```bash
python tools/rn_certificate.py check-candidates
python tools/rn_certificate.py verify research/rn/candidates/variance4_certificate_v1.json
python tools/rn_certificate.py produce variance4 --output /tmp/variance4_certificate.json
python -m pytest tests/test_rn_certificate.py -q
```

`check-candidates` verifies the complete explicitly enumerated candidate set,
the pinned local RN5 source bytes, expected contexts, and complete expected
mathematical inputs and outputs. It prints one summary line for CI.
`verify --source PATH --expected-context PATH` adds optional external binding
checks. CLI exit codes are 0 for successful replay plus all requested checks,
1 for rejection or requested binding mismatch, and 2 for resource exhaustion.
`produce` refuses to overwrite an existing artifact.

Adversarial tests alter coefficients, covariance entries, context/source
identity, feasibility quantifiers, partition coverage, Bernstein coefficients,
caps and scope flags; malformed JSON, noncanonical rationals and exhausted
budgets also fail closed. Passing them records an engineering result only.
