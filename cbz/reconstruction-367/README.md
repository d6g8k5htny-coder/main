# Rebuild 367 literal classification bindings

This isolated reconstruction uses already published evidence. It does not recover the unpublished 4d416a file, execute Lean, authenticate historical transcripts, prove model premises or establish CBZ-0. Duplicate declaration names in distinct Math packages retain separate context IDs.

Builder: python3.11 -B -S cbz/reconstruction-367/regenerate.py --root ROOT --output NEW.json. ROOT contains main at 82d9e480a51b946b3e74525ccf32139129d37d28 and Math- at 9f8dbb4d18b9f3ebd9ffac6b0caa42df9f343966 with Git history. It checks the 94 original IDs and 207/66 normalized replay records, all source/receipt/evidence blob identities and SHA-256s, axiom lists and combined-index entries.

Self-verification: python3.11 -B -S cbz/reconstruction-367/mutations.py runs 15 real-Git and negative controls plus five guard removals and a poisoned-loader harness mutation in each Python mode. Only selected assertion failures count. This checks finite behavior, not semantic completeness.

Independent binding audit: run ROOT/main/tools/cbz_inventory.py generate --root ROOT --repositories main Math- --classifications NEW.json --output NEW-INVENTORY.json, followed by verify --root ROOT --repositories main Math- --manifest NEW-INVENTORY.json. Use Python 3.11 with -B -S and output files outside ROOT. This checks source and evidence bindings, not kernel execution authenticity.

The native workflow .github/workflows/cbz-regenerate-367.yml publishes generated data only to its isolated branch after both inventory commands succeed. Raw logs and inventory remain in the workflow artifact. observed-result.json is absent until that happens. Owner none; serves criteria 3, 8 and 11.
