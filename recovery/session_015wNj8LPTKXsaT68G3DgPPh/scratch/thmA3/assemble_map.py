import json
D='/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/thmA3'
P='/home/user/wt_d3/frontiers/c8_theorem_a_constants_d3_20261001'
s=open(D+'/SOURCE_MAP.json').read()
note=("Theorem E3: for the d = 3 eight-pin law of [LP] and the good event G_r of [LP] section 7 / [CAP] (1) (partial-block operator "
      "norms on D = [-2r, 2r] x ball(0, 2r)), on the band b in [0, 1], k in [1/2, 2], uniformly over frames and torus sides L >= 10 "
      "(and for the reference kernel), Q^W(G_r^c) <= C3(r_*) r^3 and Z_r / r^2 >= z3(r_*) for 0 < r <= r_*, with "
      "C3(1/4096) = 22754801.7964 (1.0148 c_G^(3)(0, 2)), C3(1/1024) = 22974134.2920, C3(1/512) = 23332922.4752 (C3 r_*^3 <= 0.174), "
      "z3(1/512) = 5.859005 and three further certified (uninformative) pairs. The proof is self-contained apart from the "
      "definitions and the pins of [LP] (3.1)-(3.3): it does not consume [LP] (4.1), (4.2), (5.4), (5.5) or (7.5)-(7.7). With [LP] "
      "section 8 (1 - p_r <= Q^W(G_r^c)) this gives explicit pairs (C, r_*) for [LP] Theorem A (1.1) in d = 3. Every C3 here is "
      "a cap-route constant (a constant for [LP] (7.8)), hence >= c_G^(3)(0, 2) = 2.2424e7 (Math-#208). C8 stays OPEN: no catalog "
      "edit; c_{B,K}, c_{d,L} and d >= 4 are not addressed.")
s=s.replace('"@@NOTE@@"', json.dumps(note))
json.loads(s)
open(P+'/SOURCE_MAP.json','w').write(s)
print(len(s.encode()))
