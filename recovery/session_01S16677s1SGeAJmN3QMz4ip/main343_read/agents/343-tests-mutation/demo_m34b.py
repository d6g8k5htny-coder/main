import sys, time
sys.argv = ['x', 'mutants/M34b_both_routes_missing_factor_two_THREE_POINT']
exec(open('proposed_tests.py').read().split("if __name__ == '__main__':")[0])
cert = campaign()   # stub direct = n*cdf_mass(tau) of the MUTANT module (both routes halved)
print('negative_controls status:', cert['negative_controls']['status'], '| gates all true:', all(cert['negative_controls']['gates'].values()))
print('whole_rectangle_mass reported:', cert['negative_controls']['whole_rectangle_mass'], '(literal branch of cdf_mass; computed route never consulted)')
print('overall status:', cert['status'])
for r in cert['rows']:
    print(f"  j={r['j']} status={r['status']:22} coefficient_status={r['coefficient_status']:22} intersection={[float(x) for x in r['intersection_normalized']]}")
print('coefficient interval B:', [float(x) for x in cert['coefficient_interval']])
