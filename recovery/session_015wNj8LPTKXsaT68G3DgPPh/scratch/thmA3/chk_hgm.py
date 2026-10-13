exec(open('t3.py').read())
for (m, sg) in ((0, 1), (Fr(1,3), Fr(141,100)), (1, Fr(3,2))):
    A = half_gauss_moments(m, sg, 8)
    B = [gauss_half_moment(j, m, sg) for j in range(9)]
    print([float(a.mid()) for a in A]); print([float(b.mid()) for b in B]); print(max(float(a.width()) for a in A))
