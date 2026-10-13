MUTANTS = ("image-shells", "d5-floor", "sandwich-swap", "window-scale", "trace-slope", "moment-recurrence", "remainder-dropped",
           "parts-sign", "schur-sign", "tau-cross", "sphere", "tilt-gamma", "layer-dropped", "cumulative-shift")
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
K_ORDER, RHO, CELL, T_CUT = 48, Fr(6), Fr(1, 2), Fr(100)         # Taylor order, Cauchy radius, cell width, cut (Math-#199 section 3)
D5_204 = ("44.1306518750741327103620248954026331485852349159356435082422",   # Math-#204 RESULTS.json cone_moments/D_5 (certified)
          "44.1306518750741327103620248954026331489792412102184405166029")
C6_REF_204 = "0.0076928786292368482786666312812273900811"          # Math-#204 RESULTS.json (c_{6,ref}, certified there)
F4B_209 = "0.1539033000676815392013091707812771979851002980"        # Math-#209 PINNED (F^(4)_[0,1], the d = 4 band birth factor)
F5B_213 = "0.04371743008362033053724245662643528133240074085"       # Math-#213 PINNED (F^(5)_[0,1], the d = 5 band birth factor)
PINNED = PINS_PLACEHOLDER
