"""Exact-rational tail majorants; no fields, fits or coefficient digits computed."""
from fractions import Fraction as F
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def tail(K):
    q = K+1
    return 2*(1+F(16, q))*F(1000, 2718)**(q*q//32)


def main():
    # e > sum_{j=0}^6 1/j! = 1957/720 > 2718/1000.
    assert F(1957, 720) > F(2718, 1000)
    lower_Z64 = F(289, 9)
    tail24, tail64 = tail(24), tail(64)
    square24 = 22*tail24+tail24**2
    square64 = 22*tail64+tail64**2
    covariance = (square24+square64)/lower_Z64
    normalization = square64/lower_Z64
    assert covariance < F(3, 200000000)  # 1.5e-8, a safe rational upper bound.
    assert normalization < F(1, 10**56)
    out = {
        "computed_at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "actual_author":"/root/i2_incidence_review",
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "covariance_decay":"c=2*pi^2/24^2=pi^2/288",
        "basis_reference":"square cutoff24 covariance divided by square cutoff64 normalization",
        "actual_source_reference":"full integer lattice divided by full lattice normalization",
        "elementary_constants":"3<pi<4; e>1957/720>2718/1000",
        "theta_sum_upper":11,
        "normalization_Z64_lower":"289/9",
        "two_sided_tail24_exact":str(tail24),
        "two_sided_tail64_exact":str(tail64),
        "full_covariance_difference_upper_exact":"3/200000000",
        "full_covariance_difference_upper_decimal_label":"1.5e-8",
        "normalization_covariance_difference_upper_exact":"1/10^56",
        "normalization_amplitude_difference_upper_exact":"1/10^56",
        "full_covariance_derived_bound_float_display":float(covariance),
        "normalization_derived_bound_float_display":float(normalization),
        "all_exact_rational_inequalities":"PASS",
        "scope":"Ideal coefficient covariance series only. No PCG64 ideal-law coupling, floating roundoff, C4/pathwise error, barcode matching, count sandwich, statistical interval or coefficient enclosure.",
        "fresh_fields_generated":0,
        "fits_rerun":0,
        "protocol_tolerances_changed":False,
        "scientific_effect":"NONE"
    }
    (ROOT/"TRUNCATION_NORMALIZATION_BOUND.json").open("x").write(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in out.items() if not k.startswith("two_sided_tail")},indent=2))


if __name__ == "__main__":
    main()

