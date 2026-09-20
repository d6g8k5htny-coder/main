"""Source-bound interval-family candidate. Assertions are not authenticated laws."""
from dataclasses import asdict, dataclass
import hashlib
import json

from engine.operations.rn_applicability import Context, applicable
from research.interval import Interval
from research.rn.gaussian_families import FamilyMomentEngine, family_moment_cap, interval


@dataclass(frozen=True)
class FamilyLaw:
    context: Context
    intercept: tuple
    slope: tuple
    covariance: tuple
    mean_law: str
    covariance_law: str

    def fingerprint(self):
        def endpoints(x):
            i = interval(x)
            return [str(i.lo),str(i.hi)]
        payload = {'context':asdict(self.context), 'intercept':list(map(endpoints,self.intercept)),
                   'slope':list(map(endpoints,self.slope)),
                   'covariance':[[endpoints(x) for x in row] for row in self.covariance],
                   'mean_law':self.mean_law,'covariance_law':self.covariance_law}
        return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()


def apply_family_bound(law, expected, *, degree, source_bytes, depth=2):
    applicable(law,expected)
    if hashlib.sha256(source_bytes).hexdigest() != expected.source_sha256:
        raise ValueError('source bytes do not match pinned RN5 identity')
    engine = FamilyMomentEngine(law.intercept,law.slope,law.covariance,order=law.context.order)
    p = engine.determinant(degree)
    return {'operation':'RN5_INTERVAL_FAMILY_BOUND_V1','family_sha256':law.fingerprint(),
            'source_sha256':expected.source_sha256,'context':asdict(expected),
            'degree':degree,'polynomial':p,
            'upper':family_moment_cap(p,Interval(*expected.domain),depth=depth),
            'covariance_feasibility':engine.feasibility,
            'authority':'NONE','independence_credit':0,'field_certified':False,
            'original_prize_closed':False,'spatial_cover_certified':False,
            'does_not_establish':'No identified RN field law, interval Cholesky conditioning, spatial cover, weighted Palm bound, novelty, research-wide utility or scientific status change. PSD-members-only results do not certify the entire covariance box.'}
