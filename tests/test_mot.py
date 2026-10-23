import pytest

from uavdet.eval_mot import accumulate, evaluate
from uavdet.visdrone import MotSequence


def write(path, rows):
    path.write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")
    return path


def gt_rows(n=10):
    rows = []
    for f in range(1, n + 1):
        rows.append([f, 1, 10 + 2 * f, 10, 20, 20, 1, 4, 0, 0])   # car, track 1
        rows.append([f, 2, 200, 100 + f, 20, 20, 1, 1, 0, 0])     # pedestrian, track 2
        rows.append([f, 3, 300, 300, 20, 20, 1, 2, 0, 0])         # "people": not an evaluated category
        rows.append([f, -1, 400, 0, 80, 80, 0, 0, 0, 0])          # ignored region
    return rows


def metrics(tmp_path, pred_rows):
    import motmetrics as mm

    gt = MotSequence(write(tmp_path / "gt.txt", gt_rows()))
    from uavdet.visdrone import load_mot_results

    acc = accumulate(gt, load_mot_results(write(tmp_path / "pr.txt", pred_rows)))
    return mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches", "num_false_positives"])


def test_perfect_tracks(tmp_path):
    pred = [[r[0], r[1], *r[2:6], 0.9, r[7], -1, -1] for r in gt_rows() if r[7] in (1, 4)]
    m = metrics(tmp_path, pred)
    assert m["mota"].iloc[0] == pytest.approx(1) and m["idf1"].iloc[0] == pytest.approx(1)


def test_ignored_and_unscored_predictions_are_not_false_positives(tmp_path):
    pred = [[r[0], r[1], *r[2:6], 0.9, r[7], -1, -1] for r in gt_rows() if r[7] in (1, 4)]
    for f in range(1, 11):
        pred.append([f, 50, 410, 10, 20, 20, 0.9, 4, -1, -1])   # inside the ignored region
        pred.append([f, 60, 300, 300, 20, 20, 0.9, 1, -1, -1])  # on the unscored "people" object
    m = metrics(tmp_path, pred)
    assert m["num_false_positives"].iloc[0] == 0 and m["mota"].iloc[0] == pytest.approx(1)


def test_id_switch_is_counted(tmp_path):
    pred = []
    for r in gt_rows():
        if r[7] in (1, 4):
            tid = r[1] if not (r[1] == 1 and r[0] > 5) else 99  # track 1 changes identity after frame 5
            pred.append([r[0], tid, *r[2:6], 0.9, r[7], -1, -1])
    m = metrics(tmp_path, pred)
    assert m["num_switches"].iloc[0] == 1
    assert m["mota"].iloc[0] == pytest.approx(1 - 1 / 20)
    assert m["idf1"].iloc[0] < 1


def test_evaluate_folder(tmp_path):
    (tmp_path / "ann").mkdir()
    (tmp_path / "res").mkdir()
    write(tmp_path / "ann" / "seq.txt", gt_rows())
    write(tmp_path / "res" / "seq.txt", [[r[0], r[1], *r[2:6], 0.9, r[7], -1, -1] for r in gt_rows() if r[7] == 4])
    df = evaluate(tmp_path / "res", tmp_path / "ann")
    assert list(df.index) == ["seq", "OVERALL"]
    assert df.loc["OVERALL", "mota"] == pytest.approx(0.5)  # pedestrians all missed
