"""Causal Unicode scalar-text domain controls. Root alone executes."""
import unittest
import artifact

class ArtifactUnicodeTests(unittest.TestCase):
    def test_surrogate_codepoint_values_and_keys_reject(self):
        for text in ('\ud800', '\udc00', '\ud83d\ude00'):
            for value in (text, {text: 'value'}, {'key': text}):
                with self.subTest(kind=type(value).__name__, text=repr(text)):
                    with self.assertRaises(ValueError):
                        artifact.dumps(value)

    def test_valid_unicode_scalar_roundtrips_without_collision(self):
        for text in ('\U0001f600', 'plain', '\u00e9'):
            value = {text: (text, True, None)}
            raw = artifact.dumps(value)
            self.assertEqual(artifact.loads(raw, len(raw)), value)
            self.assertEqual(artifact.dumps(artifact.loads(raw, len(raw))), raw)

