def backticked(ss):           # render_docs11's lambda, extended to a single item (it would print ' and `X`')
    ss = ['`' + s + '`' for s in ss]
    return ss[0] if len(ss) == 1 else ', '.join(ss[:-1]) + ' and ' + ss[-1]


def plain(ss):
    return ss[0] if len(ss) == 1 else ', '.join(ss[:-1]) + ' and ' + ss[-1]
