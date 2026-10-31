#!/bin/bash
# Batch-4 delivery, one step per invocation so each result can be inspected:
#   commit   - commit the six files in wt4 (author Dylan Roy, committer Claude, message C14/commit_b4.txt)
#   stage    - cherry-pick that commit onto the stage branch in wt3m (at origin/main 19a55750), compare trees, push, verify
#   pr       - open the draft PR from C14/pr_b4_body.filled.md (title below); prints number and URL
#   post PR  - post the delivery comment (C14/m322_batch4_delivery.filled.md, @@PR@@/@@HEAD@@ filled here) on Math-#322 and its mirror on main#619
set -uo pipefail
C=/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14
WT=$C/wt4; ST=$C/wt3m; BR=claude/determined-bardeen-mrqy4y-c8-wrappers
TITLE='Authenticate the C8 lifetime-difference and d = 3 Theorem A wrapper verdicts (check-mode batch 4)'
export GIT_AUTHOR_NAME="Dylan Roy" GIT_AUTHOR_EMAIL="th3realdyll@gmail.com" GIT_COMMITTER_NAME="Claude" GIT_COMMITTER_EMAIL="noreply@anthropic.com"
case ${1:-} in
  commit)
    cd $WT || exit 2
    rm -rf tests/__pycache__ tools/__pycache__
    grep -q '@@' reviews/c8_check_mode_wrappers_batch4_20261031/SCOPE.md && { echo "SCOPE.md still has placeholders"; exit 1; }
    git add -A tools/check_mode_replay.py tools/check_mode_contracts.json tests/test_check_mode_workflows.py .github/workflows/c8-lifetime-difference-constant.yml .github/workflows/c8-theorem-a-constants-d3.yml reviews/c8_check_mode_wrappers_batch4_20261031/SCOPE.md
    echo "== staged"; git status --short; n=$(git status --short | grep -c -E "^(M|A) "); [ "$n" = 6 ] || { echo "expected 6 staged paths, got $n"; exit 1; }
    git commit -q -F $C/commit_b4.txt && git log -1 --format='%H%n%an <%ae> / %cn <%ce>%n%s' && git show --stat --format= HEAD | tail -8 ;;
  stage)
    sha=$(git -C $WT rev-parse HEAD); echo "batch-4 commit $sha"
    cd $ST || exit 2
    git checkout -q $BR && git fetch -q origin main && [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || { echo "stage branch is not at origin/main: $(git rev-parse HEAD) vs $(git rev-parse origin/main)"; exit 1; }
    git cherry-pick $sha || exit 1
    echo "trees equal: $([ "$(git rev-parse HEAD^{tree})" = "$(git -C $WT rev-parse HEAD^{tree})" ] && echo yes || echo NO)"
    git log -1 --format='%H %an / %cn %s'
    for i in 1 2 3 4; do git push -q -u origin $BR && break || sleep $((2**i)); done
    echo "remote $(git ls-remote origin $BR | cut -c1-40) local $(git rev-parse HEAD)" ;;
  pr)
    head=$(git -C $ST rev-parse HEAD)
    grep -q '@@' $C/pr_b4_body.filled.md && { echo "PR body still has placeholders"; grep -o '@@[A-Z0-9_]*@@' $C/pr_b4_body.filled.md | sort -u; exit 1; }
    python3 -I -B -S -c "
import json; json.dump({'title': '$TITLE', 'head': '$BR', 'base': 'main', 'draft': True, 'body': open('$C/pr_b4_body.filled.md').read()}, open('$C/pr_b4_create.json', 'w'))"
    cd /home/user && gh api -X POST repos/d6g8k5htny-coder/Math-/pulls --input $C/pr_b4_create.json --jq '[.number, .html_url, .head.sha, .draft, .created_at] | @tsv' | tee $C/tmp/pr_b4.tsv ;;
  post)
    pr=${2:?PR number}; head=$(git -C $ST rev-parse HEAD)
    sed -e "s/@@PR@@/$pr/g" -e "s/@@HEAD@@/$head/g" $C/m322_batch4_delivery.filled.md > $C/m322_batch4_delivery.final.md
    grep -q '@@' $C/m322_batch4_delivery.final.md && { echo "delivery comment still has placeholders"; grep -o '@@[A-Z0-9_]*@@' $C/m322_batch4_delivery.final.md | sort -u; exit 1; }
    cd /home/user
    n=$(gh api "repos/d6g8k5htny-coder/Math-/issues/322/comments?since=2026-10-31T19:17:30Z&per_page=100" --jq '[.[] | select(.body | test("c8-lifetime|constants-d3|C8-WRAPPERS|batch 4"; "i"))] | length'); echo "fresh competing mentions on #322 since the checkpoint: $n"; [ "$n" = 0 ] || { echo "read them before posting"; exit 1; }
    python3 -I -B -S -c "import json; json.dump({'body': open('$C/m322_batch4_delivery.final.md').read()}, open('$C/m322_del4.json','w'))"
    out=$(gh api -X POST repos/d6g8k5htny-coder/Math-/issues/322/comments --input $C/m322_del4.json --jq '[.id, .created_at] | @tsv'); echo "$out" | tee $C/tmp/m322_del4.tsv; id=$(echo "$out" | cut -f1)
    python3 - "$pr" "$head" "$id" <<'PY'
import json, sys
pr, head, cid = sys.argv[1:4]
body = ("**DELIVERED + RELEASE MIRROR — `C8-WRAPPERS-REPAIR-C8-20261026`, batch 4: `c8-lifetime-difference-constant` and `c8-theorem-a-constants-d3`, the last two of the seven.** "
        "Canonical: [Math-#322/%s](https://github.com/d6g8k5htny-coder/Math-/issues/322#issuecomment-%s). Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8). Dylan Roy — delegated AI work. Credit 0. Scientific effect NONE.\n\n"
        "- **Draft PR:** [Math-#%s](https://github.com/d6g8k5htny-coder/Math-/pull/%s), head `%s`, one commit on Math- `main` `19a55750`; six files (shared helper, reference file, test file, two workflows, scope note). The PR body is the disposition, with the measured contract, the RED control and the local evidence.\n"
        "- **Needed, from other lanes:** one nonauthor engineering review at the exact head, the hosted checks at that head, then a separate integrator. C8 takes neither seat.\n"
        "- **Release:** batch 4 and the whole authoring stage `C8-WRAPPERS-REPAIR-C8-20261026` are released; all seven wrappers are delivered. I hold nothing else. GPT-6's workflow-control findings on the pre-replay source block (Math-#322/6522251138, 6522460101) are outside this stage and unclaimed.\n\n"
        "---\n_Generated by [Claude Code](https://claude.ai/code)_\n") % (cid, cid, pr, pr, head)
json.dump({'body': body}, open('/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/C14/b619_del4.json', 'w'))
PY
    gh api -X POST repos/d6g8k5htny-coder/main/issues/619/comments --input $C/b619_del4.json --jq '[.id, .created_at] | @tsv' | tee $C/tmp/b619_del4.tsv ;;
  *) echo "usage: deliver_b4.sh commit|stage|pr|post PR"; exit 2 ;;
esac
