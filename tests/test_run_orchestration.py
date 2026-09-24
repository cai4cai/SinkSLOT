"""CPU-only tests for run.py's unit expansion, sharding, resume and merge."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run  # noqa: E402
from configs import speedup  # noqa: E402

DUMMY_MEDIANS = {key: 1.0 + i for i, key in enumerate(speedup.MEDIAN_C)}


@pytest.fixture
def cfg(tmp_path):
    import dataclasses
    return dataclasses.replace(speedup.build_config(DUMMY_MEDIANS), output_dir=str(tmp_path / "out"))


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["dataset", "tf32", "method", "n", "m", "d", "eps", "seed", "mean_ms", "oom",
               "n_iters", "srot_slices", "sample_size"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _row(unit, *, eps=None, oom=False, mean_ms="1.0"):
    return {
        "dataset": unit.dataset, "tf32": unit.tf32,
        "method": run.CSV_METHOD.get(unit.method, unit.method),
        "n": unit.n, "m": unit.n, "d": unit.d, "eps": unit.eps if eps is None else eps,
        "seed": unit.seed, "mean_ms": "OOM" if oom else mean_ms, "oom": oom, "n_iters": 20000,
        "srot_slices": unit.slices if unit.method in run.SLICED_METHODS else "N/A",
        "sample_size": unit.slices if unit.method in run.SPARSINK_METHODS else "N/A",
    }


def test_speedup_config_requires_medians():
    with pytest.raises(ValueError, match="MEDIAN_C"):
        speedup.build_config({key: None for key in speedup.MEDIAN_C})


def test_speedup_unit_count(cfg):
    units = list(run._units(cfg))
    # 5 seeds x 5 slices x 10 eps x 37 method configurations
    assert len(units) == 9250
    assert len(set(units)) == len(units)
    per_problem = [u for u in units if u.seed == 0 and u.dataset == "half_moon"
                   and u.eps == units[0].eps]
    assert len(per_problem) == 37
    assert {(u.dataset, u.d) for u in units} == set(speedup.MEDIAN_C)


@pytest.mark.parametrize("num_shards", [1, 7, 210])
def test_shards_partition_units(cfg, num_shards):
    units = list(run._units(cfg))
    shards = [units[k::num_shards] for k in range(num_shards)]
    seen = set()
    for shard in shards:
        assert seen.isdisjoint(shard)
        seen.update(shard)
    assert seen == set(units)
    assert sum(len(s) for s in shards) == len(units)


def test_dry_run_tf32_flags(cfg):
    by_group = {}
    for unit in run._units(cfg):
        cmd = run.build_command(cfg, unit.dataset, unit.eps, cfg.output_dir, method=unit.method,
                                size=unit.n, dim=unit.d, slices=unit.slices, seed=unit.seed,
                                tf32=unit.tf32)
        assert ("--tf32" in cmd) != ("--no-tf32" in cmd)
        assert cmd[cmd.index("--only") + 1] == unit.method
        if unit.method in run.FLASH_METHODS:
            group = (unit.dataset, unit.eps, unit.method, unit.seed)
            by_group.setdefault(group, set()).add("--tf32" in cmd)
        else:
            assert "--no-tf32" in cmd
    assert by_group and all(v == {False, True} for v in by_group.values())


def test_dry_run_command_contents(cfg):
    units = [u for u in run._units(cfg) if u.seed == 0]
    sym = next(u for u in units if u.method == "sinkslotcuda_symmetric")
    cmd = run.build_command(cfg, sym.dataset, sym.eps, "o", method=sym.method, size=sym.n,
                            dim=sym.d, slices=sym.slices, seed=sym.seed, tf32=sym.tf32)
    assert cmd[cmd.index("--sinkslotcuda-slices") + 1] == str(sym.slices)
    assert "--no-sinkslotcuda-symmetric" not in cmd
    expected_tol = speedup.REL_TOL * DUMMY_MEDIANS[(sym.dataset, sym.d)]
    assert float(cmd[cmd.index("--stop-tol") + 1]) == pytest.approx(expected_tol)
    for flag, value in [("--stop-mode", "potential"),
                        ("--check-every", "5"), ("--max-iter", "20000"),
                        ("--warmup", "1"), ("--warmup-iters", "10"), ("--rep", "5")]:
        assert cmd[cmd.index(flag) + 1] == value
    assert "--scaling-tol" in cmd and "--potential-tol" not in cmd
    sp = next(u for u in units if u.method == "spar_sink")
    cmd = run.build_command(cfg, sp.dataset, sp.eps, "o", method=sp.method, size=sp.n,
                            dim=sp.d, slices=sp.slices, seed=sp.seed, tf32=sp.tf32)
    assert cmd[cmd.index("--sparsink-s") + 1] == str(sp.slices)
    assert cmd[cmd.index("--sparsink-replicates") + 1] == "1"


def test_resume_skips_done_units(cfg, capsys):
    units = list(run._units(cfg))
    num_shards, shard_idx = 210, 3
    shard = units[shard_idx::num_shards]
    # Done rows spread over another shard's CSV and the top-level CSV, one of them
    # an OOM row, one with eps printed at a different precision.
    geom = next(u for u in shard if u.method == "geomloss")
    flash = next(u for u in shard if u.method in run.FLASH_METHODS)
    sliced = next(u for u in shard if u.method in run.SLICED_METHODS)
    out = Path(cfg.output_dir)
    _write_csv(out / "shards" / "shard0100" / "forward_all.csv",
               [_row(sliced), _row(geom, oom=True)])
    _write_csv(out / "forward_all.csv", [_row(flash, eps=repr(flash.eps * (1 + 1e-12)))])
    done_units = {sliced, geom, flash}

    done = run._load_done(run._existing_result_csvs(cfg.output_dir))
    assert {u for u in shard if run._is_done(u, done)} == done_units
    # Same unit at the other precision, or at another seed, is not done.
    other_tf32 = flash._replace(tf32=not flash.tf32)
    assert not run._is_done(other_tf32, done)
    assert not run._is_done(geom._replace(seed=geom.seed + 1), done)

    run.run_sweep(cfg, cfg.output_dir, dry_run=True, num_shards=num_shards, shard_idx=shard_idx)
    text = capsys.readouterr().out
    assert f"{len(units)} units total, {len(shard)} in scope, 3 done, {len(shard) - 3} remaining" in text
    assert text.count("[dry-run]") == len(shard) - 3
    assert f"shards/shard{shard_idx:04d}" in text


def test_merge_deduplicates(cfg):
    units = list(run._units(cfg))
    out = Path(cfg.output_dir)
    _write_csv(out / "shards" / "shard0000" / "forward_all.csv",
               [_row(units[0], mean_ms="5.0"), _row(units[1])])
    _write_csv(out / "shards" / "shard0001" / "forward_all.csv",
               [_row(units[2]), _row(units[0], eps=repr(float(units[0].eps)), mean_ms="5.0")])
    merged = run.merge_shards(cfg.output_dir)
    with open(merged, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    # Merging again (now including the top-level file) is idempotent.
    run.merge_shards(cfg.output_dir)
    with open(merged, newline="") as f:
        assert len(list(csv.DictReader(f))) == 3
