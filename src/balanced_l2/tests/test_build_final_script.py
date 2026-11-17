"""Check build orchestration without running clustering or touching datasets."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class BuildFinalScriptTests(unittest.TestCase):
    def test_fresh_and_resumed_builds_in_both_layouts(self):
        script = Path(__file__).resolve().parents[2] / 'build_final.sh'
        for nested, parameters in ((True, None), (False, {}),
                                   (False, {'heads_from': '/deleted/comparison/native/heads.f32'})):
            with self.subTest(nested=nested, parameters=parameters), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / 'src'
                binary = source / '.venv/bin/python'
                binary.parent.mkdir(parents=True)
                (source / 'client' if nested else root / 'client').mkdir()
                shutil.copyfile(script, source / 'build_final.sh')
                binary.write_text(f'#!{sys.executable}\n' + '''import json, pathlib, sys
if sys.argv[1] == '-c':
    print(json.loads(pathlib.Path(sys.argv[3]).read_text())['parameters'].get('heads_from', ''))
else:
    with open('calls.jsonl', 'a') as stream:
        stream.write(json.dumps(sys.argv[1:]) + '\\n')
''')
                binary.chmod(0o755)
                if parameters is not None:
                    output = source / 'clusters_balanced_l2_96k_final'
                    output.mkdir()
                    (output / 'build.json').write_text(json.dumps({'parameters': parameters}))
                subprocess.run(['bash', str(source / 'build_final.sh')], check=True, capture_output=True)
                calls = [json.loads(line) for line in (source / 'calls.jsonl').read_text().splitlines()]
                self.assertEqual(len(calls), 3)
                self.assertEqual('--resume' in calls[0], parameters is not None)
                imported = (parameters or {}).get('heads_from')
                self.assertEqual('--heads-from' in calls[0], imported is not None)
                if imported:
                    self.assertEqual(calls[0][calls[0].index('--heads-from') + 1], imported)
                self.assertEqual(calls[2][-2:], ['--client', 'client' if nested else '../client'])


if __name__ == '__main__':
    unittest.main()
