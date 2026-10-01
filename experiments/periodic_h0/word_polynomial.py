"""Exact unit transport from real dyadic Fourier modes to a finite H0 diagram."""
from fractions import Fraction as Q
import finite_certificate as fc
import nodal_core as nc
import hessian_grid_core as hc
import exact_h0 as h0


def _dyadic(value):
    fc.require(type(value) is Q, 'Exact Fraction coefficient required')
    d=value.denominator
    fc.require(d & (d-1)==0 and d<=1<<192,
               'Real coefficient must be dyadic with denominator at most2^192')
    return value


def lift_record(polynomial, cutoff):
    """Return (complex record for L*P, minimal integer power-of-two L).

    The complete real mode order is the sampler's half-square order. The
    legacy `seed=0` field is only a compatibility label, never an RNG seed.
    No coefficient is rounded. See WORD_POLYNOMIAL.md for the unit proof.
    """
    fc.require(type(cutoff) is int and 1<=cutoff<=64, 'Cutoff in [1,64] required')
    fc.require(type(polynomial) is dict and set(polynomial)=={'dc','modes'},
               'Unexpected real polynomial schema')
    dc=_dyadic(polynomial['dc']);modes=polynomial['modes']
    expected=[(x,y) for x in range(cutoff+1) for y in range(-cutoff,cutoff+1)
              if x>0 or (x==0 and y>0)]
    fc.require(type(modes) is list and len(modes)==len(expected), 'Complete real mode set required')
    converted=[];bits=dc.denominator.bit_length()-1
    for item,frequency in zip(modes,expected):
        fc.require(type(item) in (tuple,list) and len(item)==4,'Malformed real mode')
        x,y,a,b=item
        fc.require(type(x) is int and type(y) is int and (x,y)==frequency,
                   'Incomplete, unordered or invalid frequency')
        real,imag=_dyadic(a)/2,-_dyadic(b)/2
        bits=max(bits,real.denominator.bit_length()-1,imag.denominator.bit_length()-1)
        converted.append((x,y,real,imag))
    lift=1<<max(0,bits-96)
    record={'seed':0,'dc':[str(lift*dc),'0'],
            'modes':[[x,y,str(lift*a),str(lift*b)] for x,y,a,b in converted]}
    return record,lift


def analyze(polynomial, cutoff, grid, hessian_grid, edges):
    """Certify one supplied finite polynomial; no sampling law is inferred."""
    fc.require(type(grid) is int and grid>=4, 'Sample grid at least4 required')
    record,lift=lift_record(polynomial,cutoff)
    centers,stages,error=nc.evaluate(record,grid)
    hessian=hc.bound_record(record,24,hessian_grid)
    scale=lift*nc.SCALE;eta=error/lift
    fc.require(all(Q(abs(b),scale)<=eta for _,b in centers),
               'Imaginary centers escape the real polynomial error')
    spatial=Q(24**2,grid**2)*Q(hessian['spatial_coefficient'])/lift
    epsilon=eta+spatial
    values=[a for a,_ in centers]
    barcode=h0.compute(values,grid)
    fc.require(h0.verify_by_connectivity(values,grid,barcode) is True,
               'Independent exact connectivity check failed')
    bins=h0.bin_transfer(barcode,scale,epsilon,edges)
    encoded=dict(barcode)
    encoded['intervals']=[[str(b),str(d)] for b,d in barcode['intervals']]
    encoded['essential']=[str(b) for b in barcode['essential']]
    return {'cutoff':cutoff,'side':24,'grid':grid,'hessian_grid':hessian_grid,
            'lift':str(lift),'lifted_record':record,
            'compatibility_seed_label_has_no_rng_meaning':True,
            'sample_scale':str(scale),'sample_count':len(values),
            'sample_sha256':fc.digest([str(v) for v in values]),
            'sample_digest_encoding':'canonical JSON array of row-major decimal integer strings',
            'lifted_stages':stages,'lifted_nodal_error':str(error),
            'lifted_hessian':hessian,'nodal_error':str(eta),'spatial_error':str(spatial),
            'finite_polynomial_diagram_bound':str(epsilon),
            'finite_polynomial_diagram_bound_decimal_up':fc.decimal_up(epsilon,24),
            'barcode':encoded,'computed_barcode_error':'0',
            'bins':[{k:str(v) if type(v) is Q else v for k,v in row.items()} for row in bins],
            'independent_connectivity_check':True,'gaussian_law_certified':False,
            'infinite_field_diagram_bound':None,'historical_coupling_error':None,
            'lifetime_law_certified':False}
