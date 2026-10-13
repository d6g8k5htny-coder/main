from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING, getcontext
getcontext().prec = 200
def out(lo, hi, places):
    q = Decimal(1).scaleb(-places)
    return str(Decimal(lo).quantize(q, rounding=ROUND_FLOOR)), str(Decimal(hi).quantize(q, rounding=ROUND_CEILING))
