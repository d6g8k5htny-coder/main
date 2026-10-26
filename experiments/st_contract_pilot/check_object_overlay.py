"""Generic JSON overlay controls. No theorem or exponent is evaluated."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('st_object_overlay', ROOT / 'deps' / 'st_object_overlay.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
apply_overlay = MOD.apply_overlay
loads_object = MOD.loads_object
OverlayError = MOD.OverlayError


class ObjectOverlayTests(unittest.TestCase):
    def refuse(self, code, base, values, removals):
        before = copy.deepcopy((base, values, removals))
        with self.assertRaises(OverlayError) as caught:
            apply_overlay(base, values, removals)
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual((base, values, removals), before)

    def test_add_replace_and_remove(self):
        self.assertEqual(apply_overlay({'a': {'b': 1}, 'drop': 3},
                                       {'/a/b': 2, '/new': None}, ['/drop']),
                         {'a': {'b': 2}, 'new': None})

    def test_replacement_is_not_deep_merge(self):
        self.assertEqual(apply_overlay({'a': {'old': 1}}, {'/a': {'new': 2}}, []),
                         {'a': {'new': 2}})

    def test_empty_overlay_copies_every_container(self):
        original = {'a': [{'b': []}]}
        result = apply_overlay(original, {}, [])
        result['a'][0]['b'].append('new')
        self.assertEqual(original, {'a': [{'b': []}]})

    def test_inserted_container_does_not_alias_input(self):
        values = {'/a': {'b': [1]}}
        result = apply_overlay({}, values, [])
        self.assertIn('a', result)
        result['a']['b'].append(2)
        self.assertEqual(values, {'/a': {'b': [1]}})

    def test_success_does_not_mutate_inputs(self):
        base = {'a': {'b': 1}, 'drop': False}
        values, removals = {'/a/b': 2}, ['/drop']
        before = copy.deepcopy((base, values, removals))
        apply_overlay(base, values, removals)
        self.assertEqual((base, values, removals), before)

    def test_pointer_decodes_once(self):
        result = apply_overlay({'a/b': {'~key': 1}, '~1': 0, '/': 8},
                               {'/a~1b/~0key': 2, '/~01': 3}, ['/\u007e1'])
        self.assertEqual(result, {'a/b': {'~key': 2}, '~1': 3})

    def test_empty_member_and_numeric_object_key_are_valid(self):
        self.assertEqual(apply_overlay({'': {'x': 1}, '0': 2},
                                       {'//x': 3, '/0': 4}, []),
                         {'': {'x': 3}, '0': 4})
        self.assertEqual(apply_overlay({}, {'/': 9}, []), {'': 9})

    def test_malformed_pointer_refused(self):
        for pointer in ('', 'a', '/a~', '/a~2', '/a~~0'):
            with self.subTest(pointer=pointer):
                self.refuse('INVALID_POINTER', {}, {pointer: 1}, [])

    def test_nonstring_pointer_refused(self):
        self.refuse('INVALID_POINTER', {}, {}, [1])

    def test_missing_parent_refused_without_insertion(self):
        self.refuse('MISSING_PARENT', {}, {'/a/b': 1}, [])

    def test_null_or_scalar_parent_refused(self):
        for parent in (None, 1, 'text', True):
            with self.subTest(parent=parent):
                self.refuse('NON_OBJECT_PARENT', {'a': parent}, {'/a/b': 1}, [])

    def test_array_traversal_refused_but_array_value_allowed(self):
        self.refuse('NON_OBJECT_PARENT', {'a': [0]}, {'/a/0': 1}, [])
        self.assertEqual(apply_overlay({}, {'/a': [0, 1]}, []), {'a': [0, 1]})

    def test_missing_remove_refused(self):
        self.refuse('MISSING_REMOVE', {}, {}, ['/missing'])

    def test_duplicate_remove_refused(self):
        self.refuse('CONFLICTING_TARGETS', {'a': 1}, {}, ['/a', '/a'])

    def test_equal_set_remove_targets_refused(self):
        self.refuse('CONFLICTING_TARGETS', {'a': 1}, {'/a': 2}, ['/a'])

    def test_ancestor_targets_refused_in_either_order(self):
        for values in ({'/a': {}, '/a/b': 1}, {'/a/b': 1, '/a': {}}):
            with self.subTest(values=values):
                self.refuse('CONFLICTING_TARGETS', {'a': {}}, values, [])
        self.refuse('CONFLICTING_TARGETS', {'a': {'b': 1}}, {}, ['/a', '/a/b'])
        self.refuse('CONFLICTING_TARGETS', {'a': {'b': 1}}, {'/a/b': 2}, ['/a'])

    def test_prefix_matching_is_on_decoded_tokens(self):
        self.assertEqual(apply_overlay({'a': 1, 'ab': {}}, {'/ab/x': 2}, ['/a']),
                         {'ab': {'x': 2}})
        self.assertEqual(apply_overlay({'a/b': 1, 'a': {'b': 2}}, {}, ['/a~1b']),
                         {'a': {'b': 2}})
        self.refuse('CONFLICTING_TARGETS', {'a/b': {}},
                    {'/a~1b': {}, '/a~1b/c': 1}, [])

    def test_late_bad_target_is_atomic(self):
        self.refuse('MISSING_REMOVE', {'a': 1}, {'/a': 2}, ['/private-missing'])
        self.refuse('MISSING_PARENT', {'a': 1}, {'/a': 2, '/private/missing': 3}, [])

    def test_disjoint_set_order_does_not_change_result(self):
        pairs = [('/a', 1), ('/b/x', [2]), ('/c', False)]
        self.assertEqual(apply_overlay({'b': {}}, dict(pairs), []),
                         apply_overlay({'b': {}}, dict(reversed(pairs)), []))

    def test_root_and_operation_container_types(self):
        for base in ([], None, 1, 'text'):
            with self.subTest(base=base):
                self.refuse('INVALID_ROOT', base, {}, [])
        self.refuse('INVALID_OPERATIONS', {}, [], [])
        self.refuse('INVALID_OPERATIONS', {}, {}, ())

    def test_non_json_values_and_nonstring_keys(self):
        for value in ((1, 2), {1, 2}, b'bytes'):
            with self.subTest(value=repr(value)):
                self.refuse('NON_JSON_VALUE', {}, {'/a': value}, [])
        self.refuse('NON_STRING_KEY', {1: 'one'}, {}, [])
        self.refuse('NON_STRING_KEY', {}, {'/a': {1: 'one'}}, [])

    def test_nonfinite_values_refused(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                with self.assertRaises(OverlayError) as caught:
                    apply_overlay({}, {'/a': value}, [])
                self.assertEqual(caught.exception.code, 'NONFINITE_NUMBER')

    def test_cycles_refused_and_repeated_acyclic_values_allowed(self):
        base = {}; base['loop'] = base
        with self.assertRaises(OverlayError) as caught:
            apply_overlay(base, {}, [])
        self.assertEqual(caught.exception.code, 'CYCLIC_INPUT')
        self.assertIs(base['loop'], base)
        shared = {'x': []}
        result = apply_overlay({'a': shared, 'b': shared}, {}, [])
        result['a']['x'].append(1)
        self.assertEqual(result['b'], {'x': []})
        self.assertEqual(shared, {'x': []})

    def test_depth_and_node_caps(self):
        base = {}; current = base
        for _ in range(70):
            current['a'] = {}; current = current['a']
        with self.assertRaises(OverlayError) as caught:
            apply_overlay(base, {}, [])
        self.assertEqual(caught.exception.code, 'INPUT_LIMIT')
        with self.assertRaises(OverlayError) as caught:
            apply_overlay({'a': list(range(20000))}, {}, [])
        self.assertEqual(caught.exception.code, 'INPUT_LIMIT')

    def test_operation_count_cap(self):
        self.refuse('OPERATION_LIMIT', {}, {'/' + str(i): i for i in range(257)}, [])

    def test_strict_json_loader(self):
        self.assertEqual(loads_object('{"a":[null,true,1,1.5]}'),
                         {'a': [None, True, 1, 1.5]})
        for text, code in [('{"a":1,"a":2}', 'DUPLICATE_KEY'),
                           ('{"x":{"a":1,"a":2}}', 'DUPLICATE_KEY'),
                           ('{"a":NaN}', 'NONFINITE_NUMBER'),
                           ('{"a":1e9999}', 'NONFINITE_NUMBER'),
                           ('[]', 'INVALID_ROOT'), ('{', 'INVALID_JSON')]:
            with self.subTest(text=text):
                with self.assertRaises((ValueError, TypeError)) as caught:
                    loads_object(text)
                self.assertIsInstance(caught.exception, OverlayError)
                self.assertEqual(caught.exception.code, code)

    def test_loader_length_and_type_caps(self):
        for text, code in [(b'{}', 'INVALID_JSON_INPUT'), (None, 'INVALID_JSON_INPUT'),
                           (' ' * 1048577, 'INPUT_LIMIT')]:
            with self.subTest(code=code):
                with self.assertRaises((ValueError, TypeError)) as caught:
                    loads_object(text)
                self.assertIsInstance(caught.exception, OverlayError)
                self.assertEqual(caught.exception.code, code)

    def test_error_message_does_not_echo_input(self):
        secret = 'SYNTHETIC_PRIVATE_CANARY'
        with self.assertRaises(OverlayError) as caught:
            apply_overlay({}, {}, ['/' + secret])
        self.assertEqual(str(caught.exception), 'MISSING_REMOVE')
        self.assertNotIn(secret, str(caught.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
