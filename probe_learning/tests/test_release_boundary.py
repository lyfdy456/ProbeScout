"""Keep VQA credentials out of the public source and template files."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / 'scripts/check_release.py'
spec = importlib.util.spec_from_file_location('release_boundary', path)
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class CredentialBoundaryTest(unittest.TestCase):
    def test_blank_vqa_config_and_environment_example_are_allowed(self):
        self.assertEqual(boundary.credential_issues('config.yaml', 'api_key: ""\n'), [])
        self.assertEqual(boundary.credential_issues('.env.example', 'DASHSCOPE_API_KEY=\n'), [])

    def test_nonempty_config_is_rejected_without_printing_the_value(self):
        secret = 'synthetic-test-credential'
        for name, text in [('config.yaml', f'api_key: "{secret}"'),
                           ('.env.example', f'DASHSCOPE_API_KEY={secret}')]:
            issues = boundary.credential_issues(name, text)
            self.assertTrue(issues)
            self.assertNotIn(secret, ' '.join(issues))

    def test_private_environment_files_are_rejected_even_when_empty(self):
        for name in ['.env', '.env.local', '.env.production']:
            self.assertTrue(boundary.credential_issues(name, ''))

    def test_token_patterns_are_detected_in_documentation(self):
        for secret in ['sk-' + 'A' * 32, 'hf_' + 'B' * 32]:
            issues = boundary.credential_issues('README.md', 'Example: ' + secret)
            self.assertTrue(issues)
            self.assertNotIn(secret, ' '.join(issues))
