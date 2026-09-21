# ruff: noqa: E501
from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from conftest import textured_image
from xoftr_pipeline import (
    CORPUS_BYTES,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    SPECIES,
    TIER_PARAMS,
    TIERS,
    build_sample_dataset,
    check_homography,
    check_split_disjoint,
    fetch_corpus,
    load_byod_dataset,
    make_pair,
    make_pairs,
    observer_overlap,
    read_corpus,
    split_dataset,
    validate_dataset,
    warp_points,
    working_size,
    write_dataset_csv,
)
from xoftr_pipeline import samples as sm


def test_pinned_corpus_constants():
    assert len(SAMPLE_RECORDS) == 360 and len(SPECIES) == 6
    labels = {r[1] for r in SAMPLE_RECORDS}
    assert labels == set(SPECIES) and all(sum(1 for r in SAMPLE_RECORDS if r[1] == k) == 60 for k in SPECIES)
    assert sum(r[5] for r in SAMPLE_RECORDS) == CORPUS_BYTES
    assert all(len(r[6]) == 64 and r[7].lower() in ("jpg", "jpeg", "png") for r in SAMPLE_RECORDS)
    assert len({r[2] for r in SAMPLE_RECORDS}) == 360
    assert SAMPLE_SPLIT == {"train": 36, "validation": 8, "test": 16}
    assert TIERS == ("easy", "hard") and TIER_PARAMS["hard"]["rotation"] > TIER_PARAMS["easy"]["rotation"]
    with pytest.raises(ValueError, match="extension"):
        sm.photo_url(3, "gif")


def test_fetch_corpus_pins_every_file_and_caches(tmp_path, monkeypatch):
    payload, pins = {}, []
    for i in range(3):
        raw = io.BytesIO()
        textured_image(i).save(raw, format="JPEG")
        data = raw.getvalue()
        payload[f"https://inaturalist-open-data.s3.amazonaws.com/photos/{100 + i}/medium.jpg"] = data
        pins.append((f"photo-{i:02d}", "song_sparrow", 100 + i, 900 + i, f"user{i}", len(data), hashlib.sha256(data).hexdigest(), "jpg"))
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(pins))
    calls = []

    def fetcher(url):
        calls.append(url)
        return payload[url]

    files = fetch_corpus(cache_dir=tmp_path, fetcher=fetcher)
    assert sorted(files) == sorted(p[0] for p in pins) and len(calls) == 3
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == files and len(calls) == 3
    (tmp_path / "101.jpg").write_bytes(b"drifted")
    fetch_corpus(cache_dir=tmp_path, fetcher=fetcher)
    assert len(calls) == 4
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "bad", fetcher=lambda url: b"tampered")
    records = read_corpus(files)
    assert [r["species"] for r in records] == ["song_sparrow"] * 3 and all("inat_observation_url" in r for r in records)


def test_working_size_and_make_pair_are_exact_and_seeded():
    assert working_size((1000, 750)) == (640, 480) and working_size((300, 641)) == (296, 640)
    image = textured_image(5, (500, 375))
    pair = make_pair(image, seed=7, tier="hard", record_id="p")
    assert pair["image0"].size == (640, 480) and pair["image1"].size == (640, 480)
    assert pair["tier"] == "hard" and pair["seed"] == 7
    homography = check_homography(pair["homography"])
    assert homography[2, 2] == 1.0
    again = make_pair(image, seed=7, tier="hard", record_id="p")
    assert again["homography"] == pair["homography"] and again["image1"].tobytes() == pair["image1"].tobytes()
    other = make_pair(image, seed=8, tier="hard", record_id="p")
    assert other["homography"] != pair["homography"]
    # the warp is what the homography says: image0 corners land where H sends them (inside or outside the canvas)
    corners = np.array([[0.0, 0.0], [639.0, 0.0], [639.0, 479.0], [0.0, 479.0]])
    moved = warp_points(corners, homography)
    assert np.all(np.isfinite(moved)) and np.abs(moved).max() < 5000
    with pytest.raises(ValueError, match="tier"):
        make_pair(image, seed=1, tier="impossible")
    easy = make_pairs([{"id": "a", "image": image}, {"id": "b", "image": image}], seed=3)
    assert [p["tier"] for p in easy] == ["easy", "hard"] and easy[0]["id"] == "a"


def _fake_corpus(n_per_species=4):
    records = []
    for s_index, species in enumerate(SPECIES):
        for i in range(n_per_species):
            records.append({"id": f"{species}-{i}", "image": textured_image(100 * s_index + i, (400, 300)), "species": species, "observer": f"u{i}"})
    return records


def test_build_sample_dataset_is_stratified_seeded_and_disjoint():
    sizes = {"train": 2, "validation": 1, "test": 1}
    splits = build_sample_dataset(_fake_corpus(), seed=1, sizes=sizes)
    assert {k: len(v) for k, v in splits.items()} == {"train": 12, "validation": 6, "test": 6}
    assert check_split_disjoint(splits) == {"train": 12, "validation": 6, "test": 6}
    assert splits["train"][0]["id"] == "train-000" and "source_id" in splits["train"][0]
    assert {r["tier"] for r in splits["train"]} == {"easy", "hard"}
    assert build_sample_dataset(_fake_corpus(), seed=1, sizes=sizes)["test"][0]["homography"] == splits["test"][0]["homography"]
    with pytest.raises(ValueError, match="only 4 records available"):
        build_sample_dataset(_fake_corpus(), sizes={"train": 4, "validation": 1, "test": 1})
    assert observer_overlap(splits)["observers"] == 4
    leaked = {"train": splits["train"], "test": [{**splits["train"][0], "id": "test-999"}]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaked)


def test_validate_dataset_reports_and_rejects(tmp_path):
    records = make_pairs([{"id": f"r{i}", "image": textured_image(i, (400, 300))} for i in range(4)])
    report = validate_dataset(records)
    assert report["n_records"] == 4 and report["tiers"] == {"easy": 2, "hard": 2}
    assert report["image_side"] == {"min": 640, "max": 640} and len(report["digest"]) == 64
    path0 = tmp_path / "a.png"
    records[0]["image0"].save(path0)
    on_disk = validate_dataset([{**records[0], "image0": str(path0)}], min_records=1)["records"][0]
    assert on_disk["image0"].size == records[0]["image0"].size
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], {**records[1], "id": records[0]["id"]}, *records[2:]])
    with pytest.raises(ValueError, match="4..5000"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="finite 3x3"):
        validate_dataset([{**records[0], "homography": [[1, 0], [0, 1]]}, *records[1:]])
    with pytest.raises(ValueError, match="singular"):
        validate_dataset([{**records[0], "homography": [[1, 0, 0], [1, 0, 0], [0, 0, 1]]}, *records[1:]])
    with pytest.raises(ValueError, match="sides must lie"):
        validate_dataset([{**records[0], "image0": records[0]["image0"].resize((32, 32))}, *records[1:]])
    with pytest.raises(ValueError, match="missing 'homography'"):
        validate_dataset([{k: v for k, v in records[0].items() if k != "homography"}, *records[1:]])
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_split_dataset_dedups_and_byod_round_trip(tmp_path):
    images = [{"id": f"img{i}", "image": textured_image(i, (400, 300))} for i in range(10)]
    images.append({"id": "dup", "image": images[0]["image"].copy()})
    splits = split_dataset(images, val_fraction=0.2, test_fraction=0.2, seed=0)
    assert sum(len(v) for v in splits.values()) == 10  # the duplicate image is dropped
    assert check_split_disjoint(splits) and all(r["tier"] in TIERS for r in splits["train"])
    for image in images[:4]:
        image["image"].save(tmp_path / f"{image['id']}.png")
    loaded = load_byod_dataset(tmp_path)
    assert [r["id"] for r in loaded] == ["img0", "img1", "img2", "img3"]
    with zipfile.ZipFile(tmp_path / "byod.zip", "w") as archive:
        for image in images[:4]:
            archive.write(tmp_path / f"{image['id']}.png", f"{image['id']}.png")
    assert [r["id"] for r in load_byod_dataset(tmp_path / "byod.zip")] == ["img0", "img1", "img2", "img3"]
    with zipfile.ZipFile(tmp_path / "empty.zip", "w") as archive:
        archive.writestr("notes.txt", "x")
    with pytest.raises(ValueError, match="no JPEG"):
        load_byod_dataset(tmp_path / "empty.zip")
    with pytest.raises(ValueError, match="directory or a .zip"):
        load_byod_dataset(tmp_path / "img0.png")
    out = write_dataset_csv(splits["train"], tmp_path / "train.csv")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("id,source_id,tier,seed,homography") and json.loads(text.splitlines()[1].split(",", 4)[4].split('",')[0].strip('"').replace('""', '"')) is not None
