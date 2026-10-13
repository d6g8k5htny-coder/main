exec(open('t3.py').read())
for (r0, r1) in ((Fr(31,2**14), Fr(1,512)), (Fr(15,4096), Fr(1,256)), (Fr(15,2048), Fr(1,128))):
    law=band_law(r0, r1)
    for bb, kb in (((Fr(0),Fr(1,512)),(Fr(63,32),Fr(2))), ((Fr(0),Fr(1,4)),(Fr(1,2),Fr(1))), ((Fr(1,4),Fr(1)),(Fr(1,2),Fr(1)))):
        res=box_bound(law, bb, kb, r0, r1, x0=Fr(1,64), theta=Fr(1,32))
        print(float(r1), [float(x) for x in bb+kb], 'total %.6g main %.6g tail %.3g qA2 %.3g qfar %.3g qx %.3g zt %.4f minarg %.3g' % tuple(float(res[k]) for k in ('total','main','tail','qA2','qfar','qx','zt','min_arg')))
