# Component source custody

`ALIASES.json` resolves component review and receipt source paths by both their raw path and their recorded source SHA256. The same temporary working path held multiple valid generations; path-only lookup is ambiguous. Exact aliases take precedence over broader historical aliases. Every target must still match its recorded digest within the exact published Git tree.

The combined runtime mutation_test.py is 9146cd18. Original review, notebook and identity components bind 5ded9168, fe3a1caa and cf742de4 respectively. These are distinct sources. The original review source quartet is already retained under evidence/review-boundary-fix/pre-notebook-source. Identity cf742de4 is already retained under evidence/grader-identity-fix/historical-source/exact-descriptor. Only the missing notebook component mutation_test.py is added here; matching auditor, checker and test files reuse their exact existing Git-bound paths.

The Python snapshot is a retained source audit fixture. It is not an active import, test population or runtime entrypoint, and should not be executed to recreate unchanged verification. No tests, scanner certification, kernel work or mathematical verification were rerun for this custody repair. Kernel identity runtime/test files already match their component source bindings exactly and require no extra snapshot.
