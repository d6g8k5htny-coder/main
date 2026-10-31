# Interior smooth bumps: all pin jets and the old boundary do not determine the elder pair

Dylan Roy — delegated AI review. Actual performer: OpenAI/Codex agent
`/root/c39_elder_proof_review`. Principal: Dylan Roy. His personal reading is
PENDING. Organizational-independence credit: zero. This is a fresh, narrow
assumption-ablation derivation, not a second acceptance vote for the D1 theorem.
The executor saw the OpenAI source chain and Claude review 5369064629 before
constructing this example. The valid cap theorem is not refuted.

Coordination: root owns claim `261de489-5594-464a-9542-719b9dfd8493`, Work Event
370, and PR196 comment 5921236957. No remote write, branch edit, merge, scientific
register change, Gaussian estimate, Kac–Rice audit or formal verification occurs
in this packet.

## 1. Exact source and question

PR196 at `9706a2cb3ca4cc2df792b33ab440c92ee7c76dcc`, source/base
`3e0a91b69892b9717564c592e7a966492a81566a`, gives the following sufficient
conditions in `reviews/d1_external_audit_20260930/AUDIT.md`, lines 34–72:

- M is interior to a compact cap C, b=f(M), s<b;
- (C1) f<=b throughout C;
- (C2) f<=s on its entire boundary;
- (C3) a path from M reaches a point of height >b with all path heights >=s.

Then the global connection value

    d_f(M)=sup_{gamma(0)=M, f(gamma(1))>b} min_t f(gamma(t))

equals s. The exact first-exit proof is sound: C1 puts every older endpoint
outside C; C2 bounds each path from above by s; C3 bounds the supremum from below.
On the compact Morse/distinct-critical-value locus, the unique critical point
of height s identifies the elder saddle. Neither branch adjacency nor a ridge
being a gradient trajectory is used in this implication.

The present question is different from repeating its already-reviewed proof:
could C1 be replaced by agreement of all pin jets, agreement near the complete
old cap boundary, and arbitrarily small uniform error? The answer is NO, even
for smooth Morse functions on a fixed compact torus with distinct critical
values. C2 and C3 will remain literally true at the old saddle height. C1 fails
through an older maximum inside the cap.

Sources: AUDIT blob `fdfccaf5356f4cc0cf8764bf323f94b6ff071768`, 19468 bytes,
SHA256 `98356c5625aa07bcf62b9e6f1a3869ec1fea7dd789e8ea495a97b5c7abb473cf`;
CAP blob `0633aca3c2a2882b0de4399da0a75d64c2e6b2e1`, 15160 bytes,
SHA256 `0bf922b9203c29088b12388807aa0e2ecd020485eb0f6e919679841b5b2636fc`.
Inspected slices: AUDIT §§2–6 and CAP §§1–5. The other D1 analytic interfaces
are outside this result. PR194 and PR175 were checked only to distinguish
the existing polynomial and thin-hard-chart examples from this fixed-cap,
all-jets, arbitrarily-small-C0 construction.

## 2. The explicit perturbation

Define the smooth bump

    eta(t)=exp(1-1/(1-t^2)) for |t|<1, and eta(t)=0 otherwise.

It has values in [0,1], equals 1 only at zero, and is C-infinity, including
at ±1: every derivative from the interior is an exponential times a rational
function of 1-t^2, and therefore tends to zero at the endpoints. It is even,
increasing on (-1,0), and decreasing on (0,1).

Take D=[-2,2] x [-2,2], C=[-2,1/2] x [-2,2],

    q(x)=2x^3-(3/2)x-1/2,
    f0(x,y)=q(x)-128y^2,
    M=(-1/2,0), S=(1/2,0), b=0, s=-1.

For 0<delta<1/16 let

    xi(x)=eta(4(x+1/2)),
    B_delta(x,y)=256 delta^2 xi(x) eta(2(y/delta-1)),
    f_delta=f0+B_delta.                                      (B1)

The bump is supported in

    [-3/4,-1/4] x [delta/2,3delta/2],

a compact set strictly inside C. It vanishes on neighborhoods of M, S, every
point of the complete boundary of C, and the whole line y=0. Consequently:

1. Every derivative of every finite order at both pins agrees exactly with f0.
2. Every old cap-boundary value and the original axial older-reaching path agree.
3. The uniform difference, on D and on the compact completion below, is exactly
   ||f_delta-f0||_infinity=256 delta^2. The maximum is attained at (-1/2,delta).

The path gamma(y)=(-1/2,y), 0<=y<=delta, has heights

    f_delta(gamma(y))=-128y^2+256delta^2 eta(2(y/delta-1))
                    >=-128delta^2>-1/2>-1,                  (B2)

and its endpoint has height 128delta^2>0. Hence

    d_f_delta(M)>=-128delta^2>-1.                            (B3)

This is a continuous-path inequality, not sampling. It applies to every
continuous extension preserving D. Therefore S is not the elder saddle for any
compact Morse/distinct-value completion. Meanwhile C2 and C3 at s=-1 still
hold: the old cap boundary is unchanged and the y=0 path from x=-1/2 to x=2
has minimum -1 and endpoint q(2)=25/2>0. Precisely C1 is lost, since the point
(-1/2,delta) inside C is older than M.

## 3. Exact new saddle and exact death, independent of the exterior

For completeness, the construction has a definite local Morse saddle; it is
not relying on a possibly degenerate new peak. On x=-1/2 put t=y/delta and

    v(t)=-128t^2+256 eta(2t-2).

For t in (1/2,1), write u=2-2t in (0,1). The stationary equation v'(t)=0 is
equivalent to

    A(u)=256,
    A(u)=32(2-u)(1-u^2)^2 exp(u^2/(1-u^2))/u.              (B4)

Its logarithmic derivative has positive denominator and numerator

    A'(u)/A(u)
       =2[-1+u^3+3u^4-2u^5]/[u(2-u)(1-u^2)^2].           (B5)

The polynomial p(u)=-1+u^3+3u^4-2u^5 has p(0)=-1, p(1)=1 and

    p'(u)=u^2(3+12u-10u^2)>0 for 0<u<1.

Thus A decreases to one strict minimum and then increases, tending to infinity
at both endpoints. Also A(1/2)=54 exp(1/3)<81<256. Equation (B4) therefore has
exactly two simple roots. Equivalently v has one simple minimum t_- in (1/2,3/4)
and one simple maximum t_+ in (3/4,1). To locate them in these intervals directly,
v'(1/2)=-128, v'(1)=-256 and v'(3/4)>0, since exp(-1/3)>2/3.
For t in [1,3/2], the bump derivative is nonpositive, so v'<0. Elsewhere on
the positive half-line it is just -256t. There are no further positive roots.

Set m=v(t_-) and h=v(t_+). Then

    -128<m<0<128<h<=256.                                   (B6)

Indeed the minimum occurs before t=1, the nonnegative bump bounds it below
by -128, and the maximum exceeds v(1)=128. The upper bound follows from
eta<=1 and -128t^2<=0.

The full critical points of f_delta in D are exactly M, S and

    T_delta=(-1/2,delta t_-), O_delta=(-1/2,delta t_+).

To see this without an unproved genericity claim, observe that on the bump's
x support, q'(x) and xi'(x) have the same sign: positive for x<-1/2 and
negative for x>-1/2. Outside that support the x derivative is q'(x). Therefore
its only zeros, for every y, are x=-1/2 and x=1/2. At x=1/2 the bump vanishes
and the only y zero is 0; at x=-1/2 the preceding one-variable analysis applies.
At the latter line the mixed derivative is zero and

    f_xx=-6-8192delta^2 eta(2t-2)<0,

because eta''(0)=-2. The simple y roots give f_yy=v''(t_-) >0 at T_delta
and f_yy=v''(t_+) <0 at O_delta. Thus T_delta is a nondegenerate saddle,
O_delta is a nondegenerate maximum, and M,S retain their original Hessians.
Their four heights are delta^2 m, delta^2 h, 0,-1, all distinct for delta<1/16.

There is an exact smaller cap

    C_delta=[-2,1/2] x [-2,delta t_-].

On the x interval, q<=0 and xi<=1, so f_delta(x,y)<=f_delta(-1/2,y).
The latter is <=0 throughout this smaller cap. On the top boundary it is
at most delta^2 m. On the left and right boundaries the bump vanishes and
the values are at most -27/2 and -1, respectively; on the bottom they are
at most -512. All these are below delta^2 m>-1/2. The vertical path to
(-1/2,delta) has minimum exactly delta^2 m and older endpoint 128delta^2.
Applying the two first-exit inequalities directly gives

    d_f_delta(M)=delta^2 m,  and  -128delta^2<d_f_delta(M)<0. (B7)

It follows that the actual death levels tend to birth level 0, although the
entire original cap boundary and all pin jets remain fixed, and the uniform
perturbation tends to zero. This is not a contradiction to persistence-diagram
stability: the older new maximum replaces which birth carries the long-lived
component, while M now has a short lifetime. This last interpretation is only
explanatory; the proof is the explicit cap/path argument above.

## 4. A compact Morse torus completion with distinct critical values

The statements about compact Morse functions can be realized, rather than
postulated. Work on the circle R/(16Z), keeping [-2,2] fixed. Choose a smooth
periodic Q agreeing with q on that interval, with exactly two additional
critical points on the complementary arc: a nondegenerate maximum of height
10000 and a nondegenerate minimum of height -10000. Choose a smooth periodic
V agreeing with -128y^2 on [-2,2], with exactly one additional critical point
on its complementary arc, a nondegenerate minimum of height -1000.

Here is an elementary existence construction for these choices. On the
complementary Q arc start at x=2 with value 25/2 and positive derivative,
increase to the chosen maximum, decrease to the chosen minimum, and increase
to x=14 (the coordinate -2), of value -27/2 and positive derivative. Put
ordinary quadratic germs with the required Hessian signs at the two new
extrema. Preserve the given q germs in short endpoint collars. Between them
join the positive or negative derivatives with smooth strictly sign-preserving
functions. The required integrals are the prescribed successive height
differences. Shrink all fixed collars until their integrals are smaller than
these differences, choose a positive interior baseline of sufficiently small
integral, and add a nonnegative compactly supported smooth bump to match each
integral exactly. Integration supplies the desired smooth monotone arcs with
no additional critical points and with all endpoint derivatives matched.
The V arc decreases from -512 to -1000 and increases back to -512, with the
same construction and a quadratic germ at its single minimum. Its fixed
endpoint derivatives have the required signs (-512 and +512). Arc lengths
are positive and place no restriction on these finite height differences.

On the fixed torus X=(R/(16Z))^2 define

    F0(x,y)=Q(x)+V(y),
    F_delta(x,y)=F0(x,y)+B_delta(x,y),

where the compactly supported bump is extended by zero. For F0 the eight
critical values are

    0, -1, 10000, -10000, -1000, -1001, 9000, -11000.

All are distinct; the Hessians are nonsingular diagonal matrices. F_delta
has these same eight points and exactly the two new points T_delta,O_delta.
The sign argument for the x derivative remains valid globally; outside the
bump's support it is Q'. At the three x critical lines other than -1/2 the
bump is identically zero. At x=-1/2 it only adds the two simple y roots
already analyzed. The two new critical values are in (-1/2,0) and (0,1), so
all ten remain distinct. All ten Hessians are nonsingular. Thus both F0 and
F_delta are globally Morse and have distinct critical values.

Equation (B7) holds on this torus, and the only critical point with that height
is T_delta. It is the actual ordinary superlevel elder saddle for M. For F0,
the original valid cap and axial path give d_F0(M)=-1 and partner S.

## 5. Exactly which good-event assumption fails

F0 on D satisfies the source good event at r=k=1: M3=12, M4=0, lambda=256,
and 256>(4/3)12^2=192. F_delta preserves lambda and every endpoint jet, but
the bound is over the entire cylinder. From the expansion

    eta(t)=1-t^2-(1/2)t^4+O(t^6)

one gets eta''''(0)=-12, hence at (-1/2,delta)

    |partial_y^4 B_delta|=49152/delta^2.                   (B8)

Therefore the whole-cylinder M4 is at least 49152/delta^2, and r M4<=3/10
fails decisively. No violation of the source cap theorem has been found.

The narrow technical conclusion is that its interior-height hypothesis C1
cannot be dropped even if C2, C3, all pin jets and arbitrarily good uniform
closeness are retained. The explicit G_r-to-C1 derivation in PR196 remains
load-bearing and is not replaceable by those weaker data. This result does
not re-audit or accept the other deterministic estimates or any stochastic,
coefficient, unrestricted-remainder or formal claim.

`probe.py` checks the finite algebra, a rigorous rational exponential bound,
the two root isolations and resulting death enclosure. Those controls do not
formalize this smooth continuum construction or constitute external review.
