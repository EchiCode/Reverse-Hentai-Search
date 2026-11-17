from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from balanced_l2.common import record_dtype
from balanced_l2 import build as pipeline
from balanced_l2.common import prepare_vectors, read_matrix, snapshot
from balanced_l2.source_records import SourceRecords
from benchmark_cluster_quantization import NormalizedVectors


class SourceRecordTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.values = np.zeros(19, record_dtype(3))
        self.values['code'] = np.arange(19) + 100
        self.values['page'] = np.arange(19) % 5
        self.values['vec'] = np.arange(57).reshape(19, 3) / 8
        self.files = [self.root / name for name in ('first.bin', 'empty.bin', 'last.bin')]
        self.values[:7].tofile(self.files[0])
        self.files[1].write_bytes(b'')
        self.values[7:].tofile(self.files[2])

    def test_record_and_field_reads_match_original_numpy_order_shapes_and_types(self):
        selectors = [0, -1, slice(None), slice(5, 12), slice(12, 5), slice(None, None, -2),
                     [], [17, 1, 17, -1], np.array([[8, 0], [18, 6]]), np.arange(19) % 3 == 0]
        with SourceRecords(self.files, 3) as source:
            self.assertEqual(len(source), 19)
            self.assertEqual(source.shape, (19,))
            self.assertEqual(source.dtype, self.values.dtype)
            self.assertEqual(source['vec'].shape, (19, 3))
            self.assertEqual(source['vec'].dtype, np.dtype('<f2'))
            for selector in selectors:
                with self.subTest(selector=repr(selector)):
                    np.testing.assert_array_equal(source[selector], self.values[selector])
                    for field in self.values.dtype.names:
                        np.testing.assert_array_equal(source[field][selector], self.values[field][selector])
            copied = source[3:10]
            vectors = source['vec'][[1, 4, 17]]
            copied['code'] = 0
            self.assertEqual(source[3]['code'], self.values[3]['code'])
        np.testing.assert_array_equal(vectors, self.values['vec'][[1, 4, 17]])
        source.close()
        with self.assertRaisesRegex(ValueError, 'closed'):
            source[:1]

    def test_bounds_and_complete_source_validation(self):
        with SourceRecords(self.files, 3) as source:
            for selector in (19, -20, [1., 2.], np.array([], dtype=float), [True, False],
                             np.array([2**64 - 1], np.uint64)):
                with self.subTest(selector=repr(selector)), self.assertRaises(IndexError):
                    source[selector]
            with self.assertRaises(KeyError):
                source['missing']
        self.files[2].write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            SourceRecords(self.files, 3)
        for files, dim in (([], 3), ([self.files[0], self.files[0]], 3), (self.files, 0)):
            with self.assertRaises(ValueError):
                SourceRecords(files, dim)

    def test_many_mappings_do_not_retain_file_descriptors_and_empty_inputs_work(self):
        descriptors = Path('/proc/self/fd')
        before = len(list(descriptors.iterdir()))
        files = []
        for index in range(96):
            path = self.root / f'{index}.bin'
            self.values[:1].tofile(path)
            files.append(path)
        with SourceRecords(files, 3) as source:
            self.assertLessEqual(len(list(descriptors.iterdir())), before + 2)
            np.testing.assert_array_equal(source['code'][[95, 0, 47]], [100, 100, 100])
        self.assertLessEqual(len(list(descriptors.iterdir())), before + 1)
        with SourceRecords([self.files[1]], 3) as source:
            self.assertEqual(source['vec'][:].shape, (0, 3))
            self.assertEqual(source[[]].shape, (0,))

    def assignment_fixture(self):
        source = self.root / 'normalized.f32'
        prepare_vectors(self.files, source, 3, batch_size=4)
        centers = read_matrix(source)[[0, 6, 18]] * np.float32(.85)
        heads = self.root / 'heads.f32'
        heads.write_bytes(struct.pack('<ii', *centers.shape) + centers.astype('<f4').tobytes())
        return source, heads

    def test_streaming_assignment_matches_the_original_normalized_file(self):
        source, heads = self.assignment_fixture()
        original, streaming = self.root / 'original.i32', self.root / 'streaming.i32'
        with patch.object(pipeline, 'GPU_SCORE_BYTES', 12):
            old = pipeline.assign(source, heads, original, gpu=False)
            with SourceRecords(self.files, 3) as records:
                normalized = NormalizedVectors(records)
                np.testing.assert_array_equal(normalized[:], read_matrix(source))
                new = pipeline.assign_vectors(normalized, heads, streaming, gpu=False)
        self.assertEqual(original.read_bytes(), streaming.read_bytes())
        self.assertEqual(new['query_batch_size'], 1)
        self.assertEqual(new['source_rows'], old['source_rows'])
        self.assertEqual(new['numeric_precision'], old['numeric_precision'])

    def test_source_validation_runs_after_reading_and_before_labels_are_published(self):
        source, heads = self.assignment_fixture()
        labels = self.root / 'changed.i32'
        before = [snapshot(path) for path in self.files]
        checks = []

        def unchanged():
            checks.append(labels.exists())
            if [snapshot(path) for path in self.files] != before:
                raise ValueError('Original sources changed')

        with SourceRecords(self.files, 3) as records:
            normalized = NormalizedVectors(records)

            class ChangedAfterRead:
                def __len__(self):
                    return len(normalized)

                def __getitem__(inner, rows):
                    result = normalized[rows]
                    with self.files[0].open('ab') as stream:
                        stream.write(b'changed')
                    return result

            with self.assertRaisesRegex(ValueError, 'sources changed'):
                pipeline.assign_vectors(ChangedAfterRead(), heads, labels, gpu=False, validate_inputs=unchanged)
        self.assertEqual(checks, [False, False])
        self.assertFalse(labels.exists())
        self.assertFalse(labels.with_name(labels.name + '.tmp').exists())

    def test_streaming_assignment_rejects_wrong_shapes_and_changed_centroids_atomically(self):
        source, heads = self.assignment_fixture()
        labels = self.root / 'wrong.i32'
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            pipeline.assign_vectors(np.ones((3, 2), np.float32), heads, labels, gpu=False)
        self.assertFalse(labels.exists())
        self.assertFalse(labels.with_name(labels.name + '.tmp').exists())
        values = read_matrix(source)

        class ChangedCentroids:
            def __len__(self):
                return len(values)

            def __getitem__(self, rows):
                heads.write_bytes(heads.read_bytes() + b'changed')
                return values[rows]

        with self.assertRaisesRegex(ValueError, 'inputs changed'):
            pipeline.assign_vectors(ChangedCentroids(), heads, labels, gpu=False)
        self.assertFalse(labels.exists())
        self.assertFalse(labels.with_name(labels.name + '.tmp').exists())


if __name__ == '__main__':
    unittest.main()
