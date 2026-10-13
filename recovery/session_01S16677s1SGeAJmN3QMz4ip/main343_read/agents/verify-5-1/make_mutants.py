import shutil, sys, pathlib, subprocess
V = pathlib.Path(sys.argv[1])
PC = 'experiments/universality/two_parameter_cusp_population/population_certificate.py'
MUT = {
 'M02': ("    return sqrt_bounds(a[0], BITS)[0], sqrt_bounds(a[1], BITS)[1]",
         "    return sqrt_bounds(a[0], BITS)[1], sqrt_bounds(a[1], BITS)[1]"),
 'M04': ("    normalized = core_lower + tail_low, core_upper + tail_upper_ceiling + sliver",
         "    normalized = core_lower + tail_low, core_upper + tail_upper_ceiling"),
 'M06': ("            tail_upper_floor += step * Q(endpoint_floor, 2 * SCALE)\n            tail_upper_ceiling += step * Q(endpoint_ceiling, 2 * SCALE)",
         "            tail_upper_floor += step * Q(midpoint_lower, SCALE)\n            tail_upper_ceiling += step * Q(midpoint_lower + midpoint_width, SCALE)"),
 'M16': ("    result.update({'status': 'CERTIFIED' if all(gates.values()) else 'INCONCLUSIVE_PRECISION',",
         "    result.update({'status': 'CERTIFIED',"),
}
for name, (old, new) in MUT.items():
    dst = V / 'mutants' / name
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(V / 'root', dst, symlinks=True)
    p = dst / PC
    src = p.read_text()
    assert src.count(old) == 1, (name, src.count(old))
    p.write_text(src.replace(old, new))
    print('built', name, 'lines changed:', [i+1 for i,(a,b) in enumerate(zip((V/'root'/PC).read_text().splitlines(), p.read_text().splitlines())) if a!=b])
