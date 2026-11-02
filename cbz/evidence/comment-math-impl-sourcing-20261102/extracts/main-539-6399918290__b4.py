#!/usr/bin/env python3
"""C317 final byte joins and exact finite regression algebra; not a theorem test."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument("root",type=Path)
p.add_argument("--review-sha256",required=True)
args=p.parse_args(); root=args.root
checks=[]
def check(name, ok, detail):
    checks.append({"check":name,"passed":bool(ok),"detail":detail})
    if not ok: raise AssertionError(name)
def digest(b): return hashlib.sha256(b).hexdigest()
specs={
"QUALIFICATION.md":(14524,"b98c6700e78cd9544ddab29d353d58414574182a59160e4e54b9ea3dc4b73adb"),
"sources/W2D_ORIGINAL.md":(11229,"13cf4bd04cd567222c5d31b8de68c671d8dda18f014553af47954007d0710756"),
"sources/W2D_FINDING.md":(8871,"1b03b2784e987768488edbff4887290df4504f7f44e5119a7d7b490f19c4bd24"),
"sources/P.md":(40261,"9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7"),
"sources/E2.md":(9062,"845abf9f9c99d672c2a10a887b5a2e7206a3d2de3d876f35f75ff6e2dc13e62f"),
"sources/AAL_v3.pdf":(326691,"636aa9274e6acc3bec5f2eec87918cdf72e36386a757615f9e37a5a5c743a03f"),
}
for name,(size,sha) in specs.items():
    b=(root/name).read_bytes()
    check("identity:"+name,len(b)==size and digest(b)==sha,{"bytes":len(b),"sha256":digest(b)})
b=(root/"REVIEW.md").read_bytes()
check("review_identity",digest(b)==args.review_sha256,{"bytes":len(b),"sha256":digest(b)})
for name,blob in [("P.md","dfed3b8d318a3ab1950957f393307733a4bef3f2"),("E2.md","fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a")]:
    b=(root/"sources"/name).read_bytes()
    observed=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    check("git_blob:"+name,observed==blob,observed)
for filename,identity in [("ORIGINAL_FINAL_NATIVE.json","W2D_ORIGINAL.md"),("FINDING_FINAL_NATIVE.json","W2D_FINDING.md")]:
    envelope=json.loads((root/"evidence"/filename).read_text())
    native=json.loads(envelope["structuredContent"]["content"])
    check("current_native_comment:"+identity,native["body"].encode("utf-8")==(root/"sources"/identity).read_bytes(),{"id":native["id"],"updated_at":native["updated_at"]})
# Independent coordinate construction: Y=A*(xi,eta), U=xi, Cov(xi,eta)=I.
A=[[F(1),F(1)],[F(2),F(-1)]]
u=[F(1),F(0)]
sigma=sum(x*x for x in u)
B=[sum(row[j]*u[j] for j in range(2))/sigma for row in A]
R=[[A[i][j]-B[i]*u[j] for j in range(2)] for i in range(2)]
CovR=[[sum(R[i][k]*R[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
cross=[sum(row[j]*u[j] for j in range(2)) for row in R]
check("control_C_regression",B==[1,2] and R==[[0,1],[0,-1]],{"B":list(map(str,B)),"R":[list(map(str,r)) for r in R]})
check("control_C_cross_covariance",cross==[0,0],list(map(str,cross)))
check("control_C_singular_mark",CovR==[[1,-1],[-1,1]] and CovR[0][0]*CovR[1][1]-CovR[0][1]*CovR[1][0]==0 and CovR[0][0]!=0,[list(map(str,r)) for r in CovR])
check("control_C_pin_sum",sum(B)==3 and all(sum(R[i][j] for i in range(2))==0 for j in range(2)),"Y1+Y2=3v for every residual sample")
check("control_C_translation_norm",sum(x*x for x in B)==5,"Squared Euclidean displacement is exactly 5*(v-w)^2")
report={
"checked_at":datetime.now(timezone.utc).isoformat(),
"claim":"806dba8a-771a-4873-b0d0-6eeb0805e4f1",
"qualification":"b98c6700e78cd9544ddab29d353d58414574182a59160e4e54b9ea3dc4b73adb",
"checks":checks,"passed":len(checks),"failed":0,
"semantic_controls":{"A":"Deterministic topological counterexample; assessed analytically in qualification and nonauthor review, not an executed Gaussian model.","B":"Alternating joint Gaussian laws; exact characteristic-function derivation assessed analytically, not a numerical convergence test.","C":"Exact rational matrix arithmetic above; supports only the finite algebra, not the general theorem."},
"applicable_master_boundary":"One final source-identity/native-comment join and exact finite algebra run on stable candidate/review. Analytic adequacy is the separate full nonauthor source review.",
"not_run":["project programs","field simulations","Lean/kernel","hosted CI","integration/deployment","whole-project master execution"],
"hosted_ci_applicability":"No repository source/ref/PR mutation in this comment-native source delivery; no candidate hosted checks exist or are claimed.",
"scientific_effect":"NONE"
}
(root/"FINAL_VERIFICATION.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"passed":len(checks),"failed":0,"qualification":report["qualification"],"review":args.review_sha256}))
