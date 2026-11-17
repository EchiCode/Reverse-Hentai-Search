"""Compare warm CPU inference on real gallery images in fresh, sequential browsers."""

import argparse
import json
import math
import os
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile
from statistics import median
from PIL import Image


CLIENT = Path(__file__).resolve().parents[1]
ROOT = CLIENT.parent / 'src'
if not ROOT.is_dir():
    ROOT = CLIENT.parent
HARNESS = Path(__file__).with_name('browser-benchmark.py')
REPORTS = ROOT / 'build/browser-benchmarks'
OUTPUT = ROOT / 'build/inference-execution'


def prepare_images(galleries, site):
    galleries = Path(galleries).resolve()
    folders = sorted((p for p in galleries.iterdir() if p.is_dir() and p.name.isdigit()),
                     key=lambda p: int(p.name))
    if not folders:
        raise ValueError(f'No gallery folders: {galleries}')
    images = []
    (site / 'images').mkdir()
    for index in sorted({0, len(folders) // 2, len(folders) - 1}):
        folder = folders[index]
        files = sorted(p for p in folder.iterdir() if p.is_file()
                       and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.gif'})
        if not files:
            raise ValueError(f'No images: {folder}')
        for row in sorted({0, len(files) // 2, len(files) - 1}):
            source = files[row]
            with Image.open(source) as image:
                width, height = image.size
            data = source.read_bytes()
            name = f'{len(images):03d}{source.suffix.lower()}'
            (site / 'images' / name).write_bytes(data)
            images.append({'image': f'{folder.name}/{source.name}', 'url': '/images/' + name,
                           'source': str(source), 'bytes': len(data), 'width': width, 'height': height,
                           'sha256': hashlib.sha256(data).hexdigest()})
    return images


def run_variant(name, use_client, threads, site, images, *, settings_override=None,
                browser='/usr/bin/google-chrome'):
    report_name = f'gallery_inference_{name}'
    report_path = REPORTS / f'{report_name}.json'
    log_path = OUTPUT / f'{name}.log'
    settings = {'INFERENCE_BENCH_USE_CLIENT': use_client,
                'INFERENCE_BENCH_IMAGES': images,
                'INFERENCE_BENCH_REPEATS': len(images) * 4, 'INFERENCE_BENCH_WARMUPS': len(images)}
    if threads is not None:
        settings['INFERENCE_BENCH_THREADS'] = threads
    settings.update(settings_override or {})
    script = ''.join(f'globalThis.{key}={json.dumps(value)};\n' for key, value in settings.items())
    script += Path(__file__).with_suffix('.js').read_text()
    environment = {key: value for key, value in os.environ.items() if not key.startswith('RANK_')}
    with tempfile.TemporaryDirectory(prefix='dino-inference-') as temporary:
        expression = Path(temporary) / 'benchmark.js'
        expression.write_text(script)
        environment.update(RANK_SCRIPT=str(expression), RANK_REPORT=report_name,
                           RANK_CLIENT_ROOT=str(site), RANK_BROWSER_BINARY=browser)
        with log_path.open('w') as log:
            # The runner bounds CDP evaluation to 180 seconds and closes its browser in finally.
            subprocess.run([sys.executable, str(HARNESS)], cwd=ROOT, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(report_path.read_text())
    result = report.get('result', {})
    value = result.get('result', {}).get('value')
    if report.get('error') or result.get('exceptionDetails') or not isinstance(value, dict):
        raise RuntimeError(f'{name} browser benchmark failed; inspect {log_path} and {report_path}')
    if (value.get('device') != 'wasm' or value.get('dtype') != 'q8'
            or value.get('inputKind') != 'gallery'
            or len(value.get('measurements', [])) != settings['INFERENCE_BENCH_REPEATS']):
        raise ValueError(f'{name} did not run the expected CPU q8 benchmark')
    vectors = value.get('normalizedCLS', [])
    if len(vectors) != len(images) or {item['image'] for item in vectors} != {item['image'] for item in images}:
        raise ValueError(f'{name} is missing a correctness input')
    for item in vectors:
        vector = item['vector']
        if len(vector) != 384 or not all(math.isfinite(number) for number in vector):
            raise ValueError(f'{name} returned invalid CLS values')
        if abs(sum(number * number for number in vector) - 1) > 1e-6:
            raise ValueError(f'{name} returned an unnormalized CLS vector')
    summary = {'report': str(report_path), 'log': str(log_path),
               'threads': value['configuredThreads'], 'inputs': value['inputDetails'],
               'first_query_ms': value['firstInference']['pipelineMs'],
               'warm_median_ms': value['warm']['pipelineMs']['median'],
               'warm_p95_ms': value['warm']['pipelineMs']['p95'],
               'processor_median_ms': value['warm']['processorMs']['median'],
               'model_median_ms': value['warm']['modelMs']['median'],
               'library_ms': value['libraryMs'], 'model_load_ms': value['modelLoadMs'],
               'prewarm_ms': value['prewarmMs'], 'ready_ms': value['readyMs']}
    return summary, {item['image']: item['vector'] for item in vectors}


def run_full_app(galleries):
    report_path = REPORTS / 'inference_full_app_idle.json'
    log_path = OUTPUT / 'full-app.log'
    environment = {key: value for key, value in os.environ.items() if not key.startswith('RANK_')}
    with tempfile.TemporaryDirectory(prefix='gallery-inference-app-') as temporary:
        site = Path(temporary)
        for name in ('index.html', 'app.js', 'styles.css', 'fonts', 'assets', 'image-inference.js',
                     'search-release.js', 'search-quality.js', 'search.js', 'centroid-simd.js',
                     'centroid-simd.wasm', 'centroid-worker-pool.js', 'centroid-rank-worker.js',
                     'search-worker.js', 'centroids.lvq6.bin.gz'):
            target = CLIENT / name
            (site / name).symlink_to(target, target_is_directory=target.is_dir())
        images = prepare_images(galleries, site)
        if len(images) != 9:
            raise ValueError('Full-app comparison requires nine gallery images')
        expression = site / 'benchmark.js'
        expression.write_text('globalThis.INFERENCE_BENCH_IMAGES=' + json.dumps(images) + ';\n'
                              + Path(__file__).with_name('benchmark-inference-app.js').read_text())
        environment.update(RANK_CLIENT_ROOT=str(site), RANK_EXPERIMENT_APP='1', RANK_PRODUCTION='1',
                           RANK_SCRIPT=str(expression), RANK_REPORT='inference_full_app_idle')
        with log_path.open('w') as log:
            subprocess.run([sys.executable, str(HARNESS)], cwd=ROOT, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(report_path.read_text())
    result = report.get('result', {})
    value = result.get('result', {}).get('value')
    if (report.get('error') or result.get('exceptionDetails') or not isinstance(value, dict)
            or value.get('mode') != 'full-app' or len(value.get('first', [])) != 1
            or len(value.get('steady', [])) != 18):
        raise RuntimeError(f'Full-app benchmark failed; inspect {log_path} and {report_path}')
    stages = ('modelInference', 'modelPreprocessing', 'modelExecution', 'modelPipelineOther')
    labels = {image['image'] for image in images}
    for sample in value['first'] + value['steady']:
        vector = sample.get('vector', [])
        if (sample.get('image') not in labels or len(vector) != 384
                or not all(math.isfinite(number) for number in vector)
                or abs(sum(number * number for number in vector) - 1) > 1e-6
                or not all(math.isfinite(sample.get('stages', {}).get(stage, math.nan)) for stage in stages)):
            raise ValueError('Full-app result has invalid image, CLS vector or production timing')
    if {sample['image'] for sample in value['steady']} != labels:
        raise ValueError('Full-app benchmark omitted gallery images')
    value['source_images'] = images
    destination = OUTPUT / 'full-app-idle.json'
    destination.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    summary = {'report': str(destination), 'browser_report': str(report_path), 'log': str(log_path),
               'images': len(images), 'steady_queries': len(value['steady']),
               'runtime': value['runtime'], 'index_pending_at_first': value['indexPendingAtFirst'],
               'first_ms': {stage: value['first'][0]['stages'][stage] for stage in stages},
               'steady_median_ms': {stage: median(sample['stages'][stage] for sample in value['steady'])
                                    for stage in stages}}
    (OUTPUT / 'full-app-idle-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--galleries', type=Path, default=ROOT.parent / 'galleries')
    parser.add_argument('--reverse', action='store_true', help='Reverse variant order to check timing drift')
    parser.add_argument('--full-app', action='store_true', help='Measure the actual app model with uncached image queries')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.full_app:
        print(json.dumps(run_full_app(args.galleries), indent=2, allow_nan=False))
        return
    with tempfile.TemporaryDirectory(prefix='gallery-inference-site-') as temporary:
        site = Path(temporary)
        for name in ('centroid-simd.js', 'centroid-worker-pool.js', 'centroid-simd.wasm',
                     'search.js', 'image-inference.js'):
            (site / name).symlink_to(CLIENT / name)
        images = prepare_images(args.galleries, site)
        results, references, correctness = {}, None, []
        variants = [('original', False, 4), ('cpu4', True, 4), ('client', True, None)]
        for name, use_client, threads in reversed(variants) if args.reverse else variants:
            print(f'Benchmarking {name}: {len(images)} gallery images, {threads or "automatic"} CPU threads', flush=True)
            result, vectors = run_variant(name, use_client, threads, site, images)
            results[name] = result
            references = references or vectors
            for image, before in references.items():
                after = vectors[image]
                cosine = sum(a * b for a, b in zip(before, after)) / math.sqrt(
                    sum(v * v for v in before) * sum(v * v for v in after))
                maximum_error = max(abs(a - b) for a, b in zip(before, after))
                if not math.isfinite(cosine) or cosine < .999999 or maximum_error > 1e-5:
                    raise ValueError(f'CLS correctness gate failed for {name}/{image}: {cosine}, {maximum_error}')
                correctness.append({'variant': name, 'image': image, 'cosine': cosine, 'max_abs_error': maximum_error})
    summary = {'device': 'wasm', 'dtype': 'q8', 'images': images, 'repeats': len(images) * 4,
               'variants': results, 'correctness': correctness,
               'note': 'Serial CPU-only Chrome on real gallery pages. Target: warmed model execution, excluding startup/preprocessing.'}
    (OUTPUT / 'results.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'images': len(images),
                      'variants': {name: {key: result[key] for key in ('threads', 'model_median_ms',
                          'warm_median_ms', 'warm_p95_ms')} for name, result in results.items()},
                      'max_embedding_error': max(item['max_abs_error'] for item in correctness),
                      'report': str(OUTPUT / 'results.json')}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
