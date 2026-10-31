# Vault intake protocol — register before move

Applies to Drive vault `1VTiBaRlBvHGptiXqoEli4E5630nO7mNb` after the one-time #91 freeze is released.

1. **Identify candidate outside the vault.** Record exact Drive ID, title, MIME, byte size, parent, and reason.
2. **Resolve authority first.** Name the live successor/reference outside the vault. If no successor exists and the item has mathematical/provenance/code value, do not vault it.
3. **Git-first shelf record.** Before any Drive move, append the candidate to the machine-readable vault manifest on the live Git branch with intended classification and successor ID. CI/readback must see the row.
4. **Move, never delete.** Relocate the exact Drive ID into the vault; do not rewrite its content as part of intake.
5. **Read back membership.** Confirm the same Drive ID is now a direct vault child and the named successor remains outside.
6. **Finalize the Git row.** Add observed vault parent, timestamp, mover/session, content hash when obtainable, and `INTAKE_VERIFIED`.
7. **Fail closed.** A missing Git pre-record, missing successor, failed readback, changed bytes, or ambiguous scientific dependency aborts intake and leaves the item outside the vault.
8. **No authority inversion.** Vault presence, hashes, and duplicate classification never establish theorem correctness or current scientific status.

The one-time census remains controlling for the pre-existing 39 children until every unresolved row is reconciled. This protocol is prepared now but does not release the current freeze.
