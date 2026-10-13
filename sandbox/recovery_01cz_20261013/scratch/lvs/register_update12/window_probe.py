import sys, re, subprocess
sys.argv = [sys.argv[0]]  # unused
B = {"38116277597": ("00d2ac6e", "c4c66bb8"), "38121713840": ("910809d6", "5b198699"), "38122386459": ("d9b8dd33", "866136cc"),
     "38124262242": ("79731b96", "318ff194"), "38149888699": ("0dcb0016", "4363b692"), "38149524759": ("88c21058", "3306233b"),
     "38150363063": ("12e072d6", "86a6e8c8"), "38180705952": ("48bfcbae", "9833d6d9"), "38195810463": ("c6ce0256", "1320c5b2"),
     "38198716748": ("32fc9845", "37e0f1d8"), "38205232428": ("de732f1e", "8fe0759e"), "38205232441": ("de732f1e", "8fe0759e"),
     "38205232621": ("de732f1e", "8fe0759e")}
def need(flat, pos, h, m):
    for w in range(0, 3000, 10):
        win = flat[max(0, pos - w):pos + w]
        if h in win and m in win:
            return w
    return 9999
for path in sys.stdin.read().split():
    flat = open(path).read().replace("\n", " ")
    res = []
    for run, (h, m) in B.items():
        i = flat.find(run)
        while i >= 0:
            res.append((need(flat, i, h, m), run, flat[max(0, i - 60):i + 40]))
            i = flat.find(run, i + 1)
    res.sort(reverse=True)
    print(path)
    for r in res[:14]:
        print("  ", r[0], r[1], "|", r[2])
