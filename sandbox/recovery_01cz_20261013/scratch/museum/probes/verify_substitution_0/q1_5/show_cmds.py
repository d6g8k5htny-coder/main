import json, sys
m = json.load(open(sys.argv[1]))
want = ("d2-lifetime-remainder", "d4-fixed-remote-rn", "d6-p15-full-price")
print(" | ".join(c["id"].split("-")[0] + "=" + str(c["replay"]["command"]) for c in m["claims"] if c["id"] in want))
