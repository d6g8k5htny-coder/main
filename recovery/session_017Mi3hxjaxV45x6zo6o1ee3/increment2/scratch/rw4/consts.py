MUTANTS = ("image-shells", "d3-floor", "sandwich-swap", "window-scale", "trace-slope", "moment-recurrence", "remainder-dropped",
           "schur-sign", "tau-cross", "sphere", "tilt-gamma")
MUT = None

PREC = 60
CF = Context(prec=PREC, rounding=ROUND_FLOOR, Emin=-9999999, Emax=9999999)
CC = Context(prec=PREC, rounding=ROUND_CEILING, Emin=-9999999, Emax=9999999)
CE = Context(prec=PREC, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999)
# Every arithmetic step below names one of the three contexts; operator syntax on Decimals would round to the thread's
# current context, which is widened as a safety net (rule LIBRARY_EXACT checks exp, sqrt and the negation).
setcontext(Context(prec=2 * PREC + 20, rounding=ROUND_HALF_EVEN, Emin=-9999999, Emax=9999999))
ZERO, ONE, HALF = Decimal(0), Decimal(1), Decimal("0.5")
INF = math.inf
ROOT_DIGITS = 70
K_ORDER, RHO, CELL, T_CUT = 40, Fr(6), Fr(1, 2), Fr(60)          # Taylor order, Cauchy radius, cell width, cut (Math-#199 section 3)
C4_REF_199 = "0.023321666002952835094521194952856885153101372363"     # Math-#199 RESULTS.json (c_{4,ref}, pinned there)
PINNED = PINS_PLACEHOLDER
