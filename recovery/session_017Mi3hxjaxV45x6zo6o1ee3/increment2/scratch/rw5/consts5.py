MUTANTS = ("image-shells", "d4-floor", "sandwich-swap", "window-scale", "trace-slope", "moment-recurrence", "remainder-dropped",
           "parts-sign", "schur-sign", "tau-cross", "sphere", "tilt-gamma", "layer-dropped")
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
K_ORDER, RHO, CELL, T_CUT = 48, Fr(6), Fr(1, 2), Fr(80)          # Taylor order, Cauchy radius, cell width, cut (Math-#199 section 3)
C5_REF_200 = "0.0132193193800849680760334875146413965210528"        # Math-#200 RESULTS.json (c_{5,ref}, certified there)
F4B_209 = "0.1539033000676815392013091707812771979851002980"        # Math-#209 PINNED (F^(4)_[0,1], the d = 4 band birth factor)
PINNED = PINS_PLACEHOLDER
