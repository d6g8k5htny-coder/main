"""Negative controls for byte-bound museum projections (no network)."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("museum_data", ROOT / "tools/museum_data.py")
museum = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(museum)
# Immutable public index fixture. It is evidence for extraction tests, not a catalog.
INDEX_BYTES = "# Proof availability index\n\nThis file links full proof text, exact reviews, and explicitly incomplete proof obligations. Local links open the retained files in this public repository; immutable external links identify sources still on other public branches. Scientific status is unchanged by this index. Review descriptions below report the linked records, checked on 2026-09-26; they do not replace the claim manifest or the source statements.\n\n## Imported source bytes — publication only\n\n[PR70](https://github.com/d6g8k5htny-coder/Math-/pull/70) landed these public imports at Math- commit `9d7b6802424fb4715b31999066aafca8ee2f3cca`.\nOpen the [custody ledger](https://github.com/d6g8k5htny-coder/Math-/blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/imports/hardening_ebedb780/README.md) or [identity manifest](https://github.com/d6g8k5htny-coder/Math-/blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/imports/hardening_ebedb780/MANIFEST.json) for paths, sizes, Git blobs, and SHA256 digests.\nThe nine `BYTE_COPY` IDs are `EC-014`, `EC-015`, `EC-021`, `P02-LM-001`, `P02-LM-002`, `P02-LM-005`, `P02-LM-007`, `P02-LM-008`, and `P15-B`.\nTheir destination root is [`imports/hardening_ebedb780/`](imports/hardening_ebedb780/); the source is `d6g8k5htny-coder/main@ebedb7802024fa557e9071e4c9cec7cddc474b89`.\nThe separate `main#59` carrier is a labeled `TRANSCRIPTION`, not original bytes and not one of the nine byte copies.\nVerify custody with `python -B -S imports/hardening_ebedb780/verify.py`.\n**Publication is not acceptance.** Source self-labels are quoted, not adopted; no verdict, `lemma_closed`, prize, premise, or landing claim changes here.\nThe shared [public inventory](https://github.com/d6g8k5htny-coder/main/tree/main/docs/public-math) remains the catalog; this pointer creates no second inventory.\n\n## Reviewed scoped results\n\n- D2 lifetime remainder: proof [frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md](frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md); [R1–R4 review](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841270276) and [R5/R6 delta review](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206) record ACCEPT for Theorem R at its stated existential `O(1)`-remainder scope. The delta identifies its exact parent imports and does not consume the still-open Theorem A cubic selection bound. Numerical constants/radii, a second coefficient, and RN/24-jet closure are outside this verdict. [main #67](https://github.com/d6g8k5htny-coder/main/issues/67) remains open for other bundled fronts.\n- D3 SIDE24 coefficient: proof [coefficients/side24_v1/PROOF.md](coefficients/side24_v1/PROOF.md); [C1–C6 review](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490) verifies the coefficient calculation; [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841779222) closes the coefficient-review work package. Interpretation as the parent lifetime law retains the parent's separate dependency and review boundaries.\n- D4 fixed-remote RN theorem: proof [frontiers/remote_window_20260924/PROOF.md](frontiers/remote_window_20260924/PROOF.md); [source-bound review](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172) and [scoped reconciliation](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841861362) record ACCEPT on the seven fixed-rho/fixed-eta interfaces. Shrinking spatial cutoffs, pin/intermediate regions, and shrinking witness separation are outside that review.\n- D5 all-height fixed annulus: proof [frontiers/rn_annulus_bridge_20260925/PROOF.md](frontiers/rn_annulus_bridge_20260925/PROOF.md); review [reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md](reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md) records R1–R7 ACCEPT at fixed `d=2`, fixed `L`, compact positive-gap marks, and fixed annulus `1<A<B<infinity`. Pin, intermediate, and witness-collision complements remain open. This is the full reviewed candidate body, not a global D5/RN theorem.\n- D5 fixed-annulus height-window fallback: proof [frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md](frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md); review [reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md](reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md).\n- D5 two-scale S6-S21: proof [frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md](frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md); review [reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md](reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md).\n- D5 inner-belt gradient density: proof [frontiers/axial_density_20260925/PROOF.md](frontiers/axial_density_20260925/PROOF.md); review [reviews/replacement_20260925_pr19_pr21/REVIEW.md](reviews/replacement_20260925_pr19_pr21/REVIEW.md) accepts corollary (1) on `|z|<=W r^2`. That acceptance is the density bound only.\n- D5 fixed-transverse chart: proof [reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md](reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md) (byte-for-byte import of commit `32b80ee085dc6a40113d1e46e333cda50d57ba21`, blob `024d927779f79fadaf932541ea6474f2995f8c50`); review [reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md](reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md).\n- Cumulative transfer correction: proof [reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md](reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md); review [reviews/d2_cumulative_correction_20260925/REVIEW.md](reviews/d2_cumulative_correction_20260925/REVIEW.md) accepts this correction only.\n- P15 demand-one extension: exact counterexample [frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md](frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md); review pointer in [claims/LANDING_CLAIMS.json](claims/LANDING_CLAIMS.json) is [main #67](https://github.com/d6g8k5htny-coder/main/issues/67), disposition EXACT_COUNTEREXAMPLE.\n- D6 P15 full price theorem: proof [frontiers/full_price_20260924/PROOF.md](frontiers/full_price_20260924/PROOF.md); [reviews/p15_full_price_nonauthor_20260926/REVIEW.md](reviews/p15_full_price_nonauthor_20260926/REVIEW.md) records ACCEPT for Theorem F on the stated realized disjoint-capacity/clutter family, with demands at least two, independent probabilities, and the same palette. That review leaves the landing disposition untouched; it does not accept arbitrary downsets, demand-one extension, or an unrestricted prize claim.\n\n## Open or conditional results with complete proof text in GitHub\n\n- D1 parent lifetime theorem: full proof [imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md](imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md); support [imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md](imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md). [D1-A–E review](https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841570965) accepts its exact Sections 8–15 interfaces. [main #63 was reopened](https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743): Theorem A's Sections 2–7 quantitative selection chain still awaits A1–A7 review. Its A3 target must include the additive [full congruence erratum](https://github.com/d6g8k5htny-coder/Math-/blob/d573b99d8792f3d5146c952737a4ba446a549e6f/imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md), still on [PR64](https://github.com/d6g8k5htny-coder/Math-/pull/64); the immutable parent bytes remain unchanged.\n- D1 Section 9 Borel elder-mark repair: [reviews/d1_section9_borel_repair_20260925/REPAIR.md](reviews/d1_section9_borel_repair_20260925/REPAIR.md). Author-side amendment; nonauthor re-review required. The parent mirror is unchanged.\n- D4 RN count interface: [frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md](frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md).\n- D5 thin-tube candidate: [frontiers/rn_thin_tube_20260925/PROOF.md](frontiers/rn_thin_tube_20260925/PROOF.md). Independent review is open. This file is distinct from the reviewed two-scale addendum and the reviewed fixed-annulus candidate.\n- D5 contact-kernel tail note: [frontiers/contact_kernel_tail_20260925/NOTE.md](frontiers/contact_kernel_tail_20260925/NOTE.md). Author-side. Distinct from [reviews/contact_kernel_tail_20260925/](reviews/contact_kernel_tail_20260925/).\n- D6 P15 realized covers: [frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md](frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md).\n- D6 P15 restricted price theorem: [frontiers/price_budget_20260924/PROOF.md](frontiers/price_budget_20260924/PROOF.md).\n- Collision/transfer note: [reviews/collision_mechanism_20260925/NOTE.md](reviews/collision_mechanism_20260925/NOTE.md). Sections B–C: [reviews/pr25_contact_kernel_20260925/REVIEW.md](reviews/pr25_contact_kernel_20260925/REVIEW.md) accepts R1–R4. [reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md](reviews/pr25_typed_transfer_nonauthor_20260925/REVIEW.md) accepts R1–R2 and marks R3 AMEND_REQUIRED; that cumulative amendment is the separate reviewed correction above. Section A is outside both acceptances.\n- Contact-kernel work: [reviews/contact_kernel_tail_20260925/KERNEL_TAILS_AND_SMALL_GAP.md](reviews/contact_kernel_tail_20260925/KERNEL_TAILS_AND_SMALL_GAP.md) and [reviews/contact_kernel_tail_20260925/ANNULUS_ASYMPTOTIC_BRIDGE.md](reviews/contact_kernel_tail_20260925/ANNULUS_ASYMPTOTIC_BRIDGE.md). The kernel note cites `TRANSVERSE_CONTACT_ASYMPTOTIC.md`. **SOURCE NOT FOUND IN GIT / RECOVERY OPEN** at [Math #56](https://github.com/d6g8k5htny-coder/Math-/issues/56). The proof note is unchanged; neither this index nor the nearby annulus note supplies the missing exposition.\n\n## Open obligations without a complete proof yet\n\n- D5 pin neighborhoods: retained derivations are [reviews/d5_finite_r_hermite_repair_20260925/REPAIR.md](reviews/d5_finite_r_hermite_repair_20260925/REPAIR.md) and [reviews/d5_finite_r_hermite_repair_20260925/C6_REMAINDERS.md](reviews/d5_finite_r_hermite_repair_20260925/C6_REMAINDERS.md), with M1–M7 and S1–S4 accepted in [reviews/replacement_20260925_pr19_pr21/REVIEW.md](reviews/replacement_20260925_pr19_pr21/REVIEW.md). The newer public [PR53 reconnaissance note](https://github.com/d6g8k5htny-coder/Math-/blob/9a6f8a660d4508ec273b67d67e6b3871ddf2aa19/reviews/pin_neighborhood_recon_20260926/NOTE.md) has an [AMEND review in PR55](https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md). These sources are on unmerged branches. **NO COMPLETE PROOF YET:** the inner microdisk bound and the open collar to the reviewed annulus are not supplied by that reconnaissance. [Math #58](https://github.com/d6g8k5htny-coder/Math-/issues/58) tracks the microdisk derivation; the claimed summed pin-neighborhood bound remains AMEND.\n- D5 intermediate scale r << |x| << rho: bounded on opposite sides by [frontiers/remote_window_20260924/PROOF.md](frontiers/remote_window_20260924/PROOF.md) and by the reviewed fixed-annulus proofs [frontiers/rn_annulus_bridge_20260925/PROOF.md](frontiers/rn_annulus_bridge_20260925/PROOF.md) and [frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md](frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md). NO COMPLETE PROOF YET: uniform growing-scaled-radius bridge.\n- D5 shrinking multiple-witness collision: fixed-separation machinery is [frontiers/remote_window_20260924/PROOF.md](frontiers/remote_window_20260924/PROOF.md) section 6, (15)–(16), at fixed `eta>0`. NO COMPLETE PROOF YET: shrinking-separation factorial-moment/collision estimate.\n- D0 historical CH-LIFT/Piece-2/24-jet obligations outside reviewed regional bypasses: see [frontiers/downstream_gate_20260925/GRAPH.json](frontiers/downstream_gate_20260925/GRAPH.json). NO COMPLETE PROOF YET where not explicitly superseded; absent historical carriers remain ABSENT.\n\n## Repository rule\n\nA reviewed/closed theorem or lemma must have its full proof or immutable byte-bound mirror in this repository plus the exact review/certificate. An open theorem with a complete candidate proof must link it here. An open item without a complete proof must link the strongest partial derivation and state the missing proof explicitly. Superseded/refuted work stays available. Tests and hashes are evidence, not substitutes for analytic proof. A navigation refresh records source availability and existing review scope; any change to [claims/LANDING_CLAIMS.json](claims/LANDING_CLAIMS.json) or the D0-D7 graph requires its own source-bound reconciliation through the existing validators.\n".encode("utf-8")


class MuseumDataTests(unittest.TestCase):
    def setUp(self):
        self.status = (ROOT / "STATUS.md").read_bytes()

    def test_exact_fourteen_projections_and_distinct_d5_scopes(self):
        claims = museum.project(INDEX_BYTES, self.status)
        self.assertEqual(len(claims), 14)
        self.assertEqual([c["class"] for c in claims].count("ACCEPT-scoped"), 10)
        self.assertEqual([c["class"] for c in claims].count("AMEND/open"), 3)
        self.assertEqual(claims[3]["title"], "D5 all-height fixed annulus")
        self.assertEqual(claims[3]["class"], "ACCEPT-scoped")
        self.assertIsNone(claims[3]["status_quote"])
        self.assertEqual(claims[12]["class"], "AMEND/open")
        self.assertIn("inner microdisk", claims[12]["status_quote"])
        self.assertEqual(claims[9]["source_label"], "EXACT_COUNTEREXAMPLE")
        self.assertEqual(claims[9]["class"], "engineering-only")
        exact = INDEX_BYTES.decode().split("## Reviewed scoped results\n", 1)[1].split("\n## ", 1)[0]
        self.assertEqual([c["scope_quote"] for c in claims[:11]],
                         [line for line in exact.splitlines() if line.startswith("- ")])

    def test_changed_quote_refused_before_projection(self):
        for raw in (INDEX_BYTES.replace(b"only.", b"globally.", 1), INDEX_BYTES + b" "):
            with self.subTest(raw=raw[-10:]), self.assertRaisesRegex(ValueError, "identity mismatch"):
                museum.project(raw, self.status)

    def test_status_promotion_refused(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            museum.project(INDEX_BYTES, self.status.replace(b"AMEND", b"ACCEPT", 1))

    def test_misdirected_review_pointer_refused(self):
        specs = copy.deepcopy(museum.CLAIM_SPECS)
        specs[3][2] = "reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md"
        with patch.object(museum, "CLAIM_SPECS", specs), self.assertRaisesRegex(ValueError, "review pointer"):
            museum.project(INDEX_BYTES, self.status)

    def test_replay_commands_bind_source_and_checkout(self):
        claims = museum.project(INDEX_BYTES, self.status)
        for index in (0, 2, 10):
            replay = claims[index]["replay"]
            self.assertEqual(replay.get("source", {}).get("path"), "README.md")
            self.assertEqual(replay.get("working_directory"), "Math- repository root")
            self.assertEqual(replay.get("checkout"), "d6628da09384728992dcbe6e921cc28ba85aebb0")

    def test_invalid_identity_fields_refused(self):
        for field, value in (("commit", "main"), ("bytes", True), ("sha256", "0"*63), ("blob", "main")):
            identity = dict(museum.INDEX, **{field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                museum.verify_bytes(INDEX_BYTES, identity)

    def test_check_rejects_quote_identity_and_class_tampering(self):
        expected = {"claims": museum.project(INDEX_BYTES, self.status)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "museum.json"
            for field, value in (("scope_quote", "Globally accepted"), ("class", "ACCEPT"),
                                 ("source_label", "CLOSED")):
                changed = copy.deepcopy(expected)
                changed["claims"][12][field] = value
                path.write_bytes(museum.dump(changed))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "export differs"):
                    museum.check_export(path, expected)
            changed = copy.deepcopy(expected)
            changed["claims"][0]["proof"]["sha256"] = "0"*64
            path.write_bytes(museum.dump(changed))
            with self.assertRaisesRegex(ValueError, "export differs"):
                museum.check_export(path, expected)

    def test_packet_selection_excludes_unknown_fork_packet(self):
        packets = museum.packet_descriptors()
        self.assertEqual([p["id"] for p in packets], ["side24-identity-replay-20260926"])
        self.assertIsNone(packets[0]["issue"])
        self.assertEqual(packets[0]["scientific_effect"], "NONE")
        self.assertEqual(packets[0]["review_status"], "REVIEW_REQUIRED")
        self.assertEqual(packets[0]["result"]["commit"], "71400b94f6cb354a8cf7aba73ffede2138a64efa")
        # Only the frozen landed packet is selected; filesystem discovery must not add a fork.
        self.assertNotIn("showcase", json.dumps(packets).lower())

    def test_review_pointers_do_not_claim_comment_byte_custody(self):
        claims = museum.project(INDEX_BYTES, self.status)
        for claim in claims[:3]:
            self.assertTrue(claim["review"]["pointer_only"])
            self.assertEqual(claim["review"]["path"], "PROOF_INDEX.md")
            self.assertIn("#issuecomment-", claim["review"]["review_url"])
        self.assertNotIn("pointer_only", claims[3]["review"])
        self.assertEqual(claims[3]["review"]["path"],
                         "reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md")

    def test_landed_packet_can_be_read_from_pinned_sparse_checkout(self):
        source = museum.packet_descriptors()[0]["result"]
        raw = museum.read_source(source, ROOT, None)
        self.assertEqual(len(raw), 1118)
        self.assertEqual(__import__("hashlib").sha256(raw).hexdigest(),
                         "fa40a6b8afc413646081145d6d5d7b01cc4b5b4ddec32f4c6e82caa399e38cd8")

    def test_historical_snapshot_ignores_unrelated_working_tree_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "checkout"
            subprocess.run(["git", "init", "--quiet", str(checkout)],
                           check=True, capture_output=True)
            (checkout / "STATUS.md").write_bytes(self.status)
            subprocess.run(["git", "add", "STATUS.md"], cwd=checkout,
                           check=True, capture_output=True)
            subprocess.run(["git", "-c", "user.name=Museum Test Fixture",
                            "-c", "user.email=museum-fixture@example.invalid",
                            "-c", "core.hooksPath=/dev/null", "commit", "--quiet",
                            "--no-gpg-sign", "-m", "Pin historical status fixture"],
                           cwd=checkout, check=True, capture_output=True)
            fixture_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout).decode("ascii").strip()
            source = dict(museum.STATUS, commit=fixture_commit)
            changed = self.status + b"\nUnrelated later status note.\n"
            (checkout / "STATUS.md").write_bytes(changed)
            raw = museum.read_source(source, checkout, None)
            self.assertEqual(raw, self.status)
            self.assertEqual(museum.project(INDEX_BYTES, raw),
                             museum.project(INDEX_BYTES, self.status))
            self.assertEqual((checkout / "STATUS.md").read_bytes(), changed)

    def test_plain_fixture_directory_does_not_bypass_byte_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            (fixture / "STATUS.md").write_bytes(self.status + b"\nChanged fixture.\n")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                museum.read_source(museum.STATUS, fixture, None)


if __name__ == "__main__":
    unittest.main()
