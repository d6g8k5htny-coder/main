"""Offline publication fidelity, not validation of the mathematical argument."""
import hashlib
import importlib.util
from pathlib import Path, PurePosixPath
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
RECORD = "experiments/periodic_h0/EXACT_SAMPLE_SHRINKING_BIN.md"
INCOMING = {
    "experiments/periodic_h0/README.md": "EXACT_SAMPLE_SHRINKING_BIN.md",
    "docs/research-translation/20260930/MANUSCRIPT.md":
        "../../../experiments/periodic_h0/EXACT_SAMPLE_SHRINKING_BIN.md",
}
COMMENT_BINDINGS = (
    ("5976299711", "6104", "819bc7c7fd637883ca74626b01d777d40233f6b710b6f26b5d370648e09a945d"),
    ("5976527971", "9103", "ad6f20ab98ce288c2ab54f88653df908015a599f30a329f3c1f12eff93e7f16f"),
)
PINNED_INPUTS = (
    ("main", "9093629769f40db76f1311d8ee920d5a82a38b89",
     "experiments/periodic_h0/APPROXIMATION.md", "137205c149336c98dd66b0aac8d6a0374f4fb8d9",
     "17970", "dae4e01b35602077442796a1150294cce3eb544f2d0c2297f8fa7aa6ab16caef"),
    ("main", "9093629769f40db76f1311d8ee920d5a82a38b89",
     "docs/research-translation/20260930/MANUSCRIPT.md", "0e8c199d2eb48c713682d612f380d8db279c931c",
     "20601", "012d288f7bcd341174fc2670cc10ee1a59fd857f4e5495acc961ad79376ab434"),
    ("Math-", "8404169d33317cc5f01ade17c829c8036d33ea2a",
     "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md",
     "dfed3b8d318a3ab1950957f393307733a4bef3f2", "40261",
     "9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7"),
    ("Math-", "8404169d33317cc5f01ade17c829c8036d33ea2a",
     "reviews/d1_section9_borel_repair_20260925/REPAIR.md",
     "fe9b9ce4999908bb3814b500ee2d0ceb0c6f704a", "9062",
     "845abf9f9c99d672c2a10a887b5a2e7206a3d2de3d876f35f75ff6e2dc13e62f"),
)
HYPOTHESES = {
    "Field and law": ("Fixed side-24", "unconditioned", "used by M4"),
    "Sampling and filtration": ("Exact periodic vertex samples", "minimum of their corner values"),
    "Density input": ("M4–M5", "recorded hypotheses and source scope"),
    "Bin schedule": ("0 < λ < μ < ∞", "deterministic `τ_h ↓ 0`", "h² sqrt(log(1/h)) = o(τ_h)"),
}
EXCLUSIONS = (
    "spectral truncation", "FFT realization error", "floating-point sampling",
    "roundoff", "persistence-library arithmetic", "post hoc bins",
    "confidence intervals", "evaluated finite-grid threshold",
    "evaluated M4/Borell/Kac–Rice constants", "current-pilot certificate",
)

spec = importlib.util.spec_from_file_location("navigation", ROOT / "tools/navigation_check.py")
navigation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(navigation)


def publication_problems(texts):
    """Guard frozen source bindings, exclusions, and the actual reading routes."""
    problems = []
    record = texts[RECORD]
    rows = [line for line in navigation.lines_outside_fences(record) if line.startswith("|")]
    for comment, size, digest in COMMENT_BINDINGS:
        url = f"https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-{comment}"
        if not any(url in row and f"`{size}`" in row and f"`{digest}`" in row for row in rows):
            problems.append("missing exact comment binding: " + comment)
    for repository, commit, path, blob, size, digest in PINNED_INPUTS:
        url = f"https://github.com/d6g8k5htny-coder/{repository}/blob/{commit}/{path}"
        if not any(url in row and all(f"`{value}`" in row for value in (blob, size, digest))
                   for row in rows):
            problems.append("missing pinned input: " + path)
    # This local input is also frozen into existing certificate receipts.
    _, _, path, _, size, digest = PINNED_INPUTS[0]
    raw = (ROOT / path).read_bytes()
    if len(raw) != int(size) or hashlib.sha256(raw).hexdigest() != digest:
        problems.append("changed frozen certificate input: " + path)
    for label, terms in HYPOTHESES.items():
        scope_rows = [row for row in rows if row.startswith(f"| {label} |")]
        if len(scope_rows) != 1 or any(term not in scope_rows[0] for term in terms):
            problems.append("missing retained hypothesis: " + label)
    excluded = [row for row in rows if row.startswith("| Excluded |")]
    if len(excluded) != 1 or any(term not in excluded[0] for term in EXCLUSIONS):
        problems.append("missing explicit exclusion")
    for page, target in INCOMING.items():
        links = [link for line in navigation.lines_outside_fences(texts[page])
                 for link in navigation.LINK.findall(line)]
        if target not in links:
            problems.append("missing reading route: " + page)
    for page, text in texts.items():
        for line in navigation.lines_outside_fences(text):
            for target in navigation.LINK.findall(line):
                parsed = urlsplit(target)
                if parsed.scheme in ("https", "http", "mailto"):
                    continue
                if parsed.scheme or parsed.netloc:
                    problems.append("unsupported link: " + target)
                    continue
                try:
                    relative = str(PurePosixPath(page).parent / unquote(parsed.path)) if parsed.path else page
                    destination = navigation.safe_file(ROOT, relative)
                    if parsed.fragment:
                        destination_text = texts.get(destination.relative_to(ROOT).as_posix(), destination.read_text())
                        if unquote(parsed.fragment) not in navigation.anchors(destination_text):
                            raise ValueError("missing fragment")
                except (ValueError, OSError) as error:
                    problems.append(page + ": " + target + ": " + str(error))
    return problems


class ShrinkingBinPublicationTests(unittest.TestCase):
    def setUp(self):
        self.texts = {page: (ROOT / page).read_text() for page in (RECORD, *INCOMING)}

    def test_published_sources_scope_and_local_routes(self):
        self.assertEqual(publication_problems(self.texts), [])

    def test_changed_proof_identity_is_rejected(self):
        changed = dict(self.texts)
        changed[RECORD] = changed[RECORD].replace(COMMENT_BINDINGS[0][2], "0" * 64)
        self.assertIn("missing exact comment binding: 5976299711", publication_problems(changed))

    def test_lost_exclusion_is_rejected(self):
        changed = dict(self.texts)
        changed[RECORD] = changed[RECORD].replace("persistence-library arithmetic", "numerical details")
        self.assertIn("missing explicit exclusion", publication_problems(changed))

    def test_lost_exact_sample_hypothesis_is_rejected(self):
        changed = dict(self.texts)
        changed[RECORD] = changed[RECORD].replace("Exact periodic vertex samples", "Numerical vertex samples")
        self.assertIn("missing retained hypothesis: Sampling and filtration", publication_problems(changed))

    def test_broken_relative_link_is_rejected(self):
        changed = dict(self.texts)
        changed[RECORD] += "\n[Missing source](C131_MISSING_SOURCE.md)\n"
        self.assertTrue(any("C131_MISSING_SOURCE.md" in problem for problem in publication_problems(changed)))

    def test_lost_incoming_route_is_rejected(self):
        changed = dict(self.texts)
        page = "experiments/periodic_h0/README.md"
        changed[page] = changed[page].replace(INCOMING[page], "README.md")
        self.assertIn("missing reading route: " + page, publication_problems(changed))


if __name__ == "__main__":
    unittest.main()
