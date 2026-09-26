import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('public_shop_data', ROOT / 'tools/public_shop_data.py')
shop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shop)


class PublicShopDataTests(unittest.TestCase):
    def setUp(self):
        self.status = (ROOT / 'STATUS.md').read_bytes()

    def test_pinned_status_counts_and_verbatim_scope(self):
        result = shop.parse_status(self.status)
        self.assertEqual(result['counts'], {'accept': 4, 'amend': 3, 'engineering': 3})
        self.assertIn('No numerical remainder constant or numerical lifetime cutoff', result['sections'][0]['rows'][0][3])
        self.assertEqual(result['source']['sha256'], shop.STATUS['sha256'])
        self.assertFalse(result['scientific_status_authority'])

    def test_changed_source_bytes_refused_before_parsing(self):
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            shop.parse_status(self.status.replace(b'AMEND', b'ACCEPT', 1))

    def test_unknown_heading_refused_even_with_matching_identity(self):
        changed = self.status + b'\n## APPROVED\n\n| Name |\n|---|\n| X |\n'
        with self.assertRaisesRegex(ValueError, 'unknown heading'):
            shop.parse_status(changed, shop.identity_for(changed, shop.STATUS))

    def test_changed_table_schema_refused(self):
        changed = self.status.replace(b'| Explicit limits |', b'| New verdict |')
        with self.assertRaisesRegex(ValueError, 'table header'):
            shop.parse_status(changed, shop.identity_for(changed, shop.STATUS))

    def test_missing_status_section_refused(self):
        changed = self.status.replace(b'## AMEND / open', b'## AMEND moved')
        with self.assertRaisesRegex(ValueError, 'unknown heading'):
            shop.parse_status(changed, shop.identity_for(changed, shop.STATUS))

    def test_extra_table_cell_refused(self):
        changed = self.status.replace(b'| **D2 ', b'| extra | **D2 ', 1)
        with self.assertRaisesRegex(ValueError, 'table row width'):
            shop.parse_status(changed, shop.identity_for(changed, shop.STATUS))

    def test_coefficient_endpoints_stay_exact_strings(self):
        raw = json.dumps(shop.COEFFICIENT_EXPECTED).encode()
        data = shop.parse_coefficient(raw, shop.identity_for(raw, shop.COEFFICIENT))
        self.assertEqual(data['dimensions']['2']['lower'], '0.07340691930603427103')
        self.assertEqual(data['dimensions']['2']['upper'], '0.07340691930603427104')
        self.assertFalse(data['scientific_acceptance'])

    def test_coefficient_float_or_added_dimension_refused(self):
        for mutation in ('float', 'dimension', 'acceptance'):
            data = json.loads(json.dumps(shop.COEFFICIENT_EXPECTED))
            if mutation == 'float':
                data['dimensions']['2']['lower'] = 0.07340691930603427
            elif mutation == 'dimension':
                data['dimensions']['4'] = data['dimensions']['3']
            else:
                data['scientific_acceptance'] = True
            raw = json.dumps(data).encode()
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'coefficient schema'):
                shop.parse_coefficient(raw, shop.identity_for(raw, shop.COEFFICIENT))

    def test_changed_inventory_manifest_refused(self):
        raw = (ROOT / 'docs/public-math/sources.json').read_bytes() + b' '
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            shop.verify_bytes(raw, shop.INVENTORY)

    def test_inventory_uses_original_shards_and_detects_tampering(self):
        index, pages = shop.inventory_config(lambda path: (ROOT / path).read_bytes())
        self.assertEqual(sum(p['count'] for p in pages), 2138)
        self.assertEqual(len(pages), 14)
        self.assertEqual(index['path'], 'docs/public-math/sources.json')
        def damaged(path):
            raw = (ROOT / path).read_bytes()
            return raw + b' ' if path.endswith('sources-01.json') else raw
        with self.assertRaisesRegex(ValueError, 'shard identity mismatch'):
            shop.inventory_config(damaged)

    def test_raw_urls_are_commit_pinned(self):
        for source in (shop.STATUS, shop.COEFFICIENT, shop.PROOF, shop.INVENTORY, shop.IMPORTS):
            self.assertIn('/' + source['commit'] + '/', shop.with_url(source)['url'])
            self.assertEqual(len(source['commit']), 40)


if __name__ == '__main__':
    unittest.main()
