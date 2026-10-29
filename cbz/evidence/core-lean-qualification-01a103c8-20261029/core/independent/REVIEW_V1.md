# Independent finite arithmetic-control review

Reviewer scope: readonly independent verification of the existing 13 Core arithmetic controls and installed runtime. No canonical source changes, tool fork, proof closure, adoption or final CBZ verdict.

## Actual result

The independent invocation explicitly used the original source directory as its working directory. The pinned Lean 4.34.1 compiler (commit 5045d0056413266e57c625dcd7c365b10e377c52) returned 0, produced a .olean, emitted empty stderr and exactly 13 expected, unique no-axiom messages in the expected order. Its stdout is byte-identical to the earlier result. The existing source, earlier stdout/stderr/.olean/result and compiler retained their exact bytes and stat identity across this run.

All finite propositions are consistent: Euclidean quotient/remainder controls at (5,-2),(-5,-2),(-5,2),(5,2),(5,0),(-5,0); Nat subtraction clamps 1-2 to 0 while Int subtraction is -1; Nat division and remainder by zero return 0 and the dividend respectively; 58*9=522<529=23^2 and 522<=529 are true while 522<484=22^2 is false. The last control proves the negation of that false inequality. Its successful compilation must not be represented as a successful proof of the inequality itself.

The entire existing extracted runtime was independently compared with the rehashed release archive: 17,711 regular files matched, zero missing files, zero mismatches and zero special entries. Archive size and SHA match the previously captured official release-asset metadata. This verifies more than merely the small launcher executable; it includes installed libraries used by the compiler.

## Reproduction correction

The original ACTUAL_RESULT_V1.json records argv but omits cwd. A first independent attempt from /Users/dylanroy/Documents/ChatGPT/github failed Lean's input-root containment precondition and produced no .olean. That failure is retained in the adjacent independent-20261029-2350 directory. The wrapper then failed trying to hash the missing output; its exit1 must not be relabeled as the compiler's exact exit code, which was not retained. This was an environment precondition failure and did not test arithmetic.

Portable publication should include explicit cwd (the source directory), runtime version/commit, source bytes/SHA, runtime archive URL/size/SHA, exact argv and output SHA values. A .lean.txt publication needs an instruction to save its identical bytes as CoreArithmeticControls.lean. The successful independent result records explicit cwd and the failed reproduction is preserved separately. Historical records should not be rewritten to conceal the omission.

## Limits

These are 13 finite semantic controls, each compiled with by decide, and an explicit transitive axiom print. They can serve as known-answer cases for a separately authored numerical translator. They do not demonstrate source-to-evaluator translation, randomized or pathological campaign scale, full target discovery, mathematical statement alignment, a candidate fuzzer's negative controls, whole-federation coverage, merge enforcement, project theorem CLOSED status or any final G0-G11/CBZ-0 pass.
