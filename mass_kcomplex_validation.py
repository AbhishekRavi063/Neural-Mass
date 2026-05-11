import argparse
import json
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import mne
import numpy as np

from src.event_detection import K_complex_detection


ANNOTATION_DOI = "doi:10.5683/SP3/Y889CS"
BOREALIS_API = "https://borealisdata.ca/api"


def dataset_metadata(persistent_id=ANNOTATION_DOI):
    url = f"{BOREALIS_API}/datasets/:persistentId/?persistentId={persistent_id}"
    with urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))["data"]


def kcomplex_files(metadata):
    files = metadata["latestVersion"]["files"]
    out = []
    for item in files:
        label = item["label"]
        if label.endswith("KComplexes_E1.edf"):
            out.append(
                {
                    "label": label,
                    "id": item["dataFile"]["id"],
                    "size": item["dataFile"]["filesize"],
                    "md5": item["dataFile"]["md5"],
                }
            )
    return sorted(out, key=lambda item: item["label"])


def download_file(file_id, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BOREALIS_API}/access/datafile/{file_id}"
    if destination.exists():
        return destination
    urlretrieve(url, destination)
    return destination


def load_expert_annotations(path):
    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    return [
        {
            "onset": float(onset),
            "duration": float(duration),
            "end": float(onset + duration),
            "description": description,
        }
        for onset, duration, description in zip(
            raw.annotations.onset,
            raw.annotations.duration,
            raw.annotations.description,
        )
    ]


def mask_to_events(mask, sfreq):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    ends = np.flatnonzero(mask & np.concatenate((~mask[1:], [True])))
    return [
        {"onset": start / sfreq, "end": (end + 1) / sfreq, "duration": (end - start + 1) / sfreq}
        for start, end in zip(starts, ends)
    ]


def event_iou(a, b):
    overlap = max(0.0, min(a["end"], b["end"]) - max(a["onset"], b["onset"]))
    union = max(a["end"], b["end"]) - min(a["onset"], b["onset"])
    if union <= 0:
        return 0.0
    return overlap / union


def score_events(expert_events, detected_events, iou_threshold=0.2):
    matched_detected = set()
    true_positive = 0

    for expert in expert_events:
        best_idx = None
        best_iou = 0.0
        for idx, detected in enumerate(detected_events):
            if idx in matched_detected:
                continue
            iou = event_iou(expert, detected)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            true_positive += 1
            matched_detected.add(best_idx)

    false_positive = len(detected_events) - len(matched_detected)
    false_negative = len(expert_events) - true_positive
    precision = true_positive / len(detected_events) if detected_events else 0.0
    recall = true_positive / len(expert_events) if expert_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "expert_events": len(expert_events),
        "detected_events": len(detected_events),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_psg(psg_path, annotation_path, channel="EEG C3-LER"):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    if channel not in raw.ch_names:
        raise ValueError(f"Channel {channel!r} not found. Available channels: {raw.ch_names}")
    signal = raw.copy().pick([channel]).get_data()[0]
    sfreq = raw.info["sfreq"]
    expert_events = load_expert_annotations(annotation_path)
    detected_mask = K_complex_detection(signal, sampling_frequency=int(sfreq))
    detected_events = mask_to_events(detected_mask, sfreq)
    return score_events(expert_events, detected_events)


def main():
    parser = argparse.ArgumentParser(description="Download/read MASS SS2 K-complex annotations.")
    parser.add_argument("--download-dir", default="data/mass_ss2", help="Where annotation EDFs are stored.")
    parser.add_argument("--subject", default="01-02-0001", help="MASS SS2 subject id.")
    parser.add_argument("--psg", default=None, help="Optional local restricted PSG.edf path for full validation.")
    args = parser.parse_args()

    metadata = dataset_metadata()
    files = kcomplex_files(metadata)
    print(f"Public expert K-complex annotation files available: {len(files)}")

    selected = next((file for file in files if file["label"].startswith(args.subject)), None)
    if selected is None:
        raise ValueError(f"No K-complex annotation found for subject {args.subject}")

    destination = Path(args.download_dir) / selected["label"]
    download_file(selected["id"], destination)
    annotations = load_expert_annotations(destination)

    print(f"Downloaded/read: {destination}")
    print(f"Expert K-complex annotations: {len(annotations)}")
    print("First 5 expert events:")
    for event in annotations[:5]:
        print(
            f"- onset={event['onset']:.3f}s "
            f"duration={event['duration']:.3f}s "
            f"end={event['end']:.3f}s"
        )

    if args.psg:
        scores = evaluate_psg(args.psg, destination)
        print("\nDetector vs expert scores:")
        for key, value in scores.items():
            if isinstance(value, float):
                print(f"- {key}: {value:.3f}")
            else:
                print(f"- {key}: {value}")
    else:
        print("\nFull detector scoring needs the matching restricted MASS PSG.edf file.")
        print("Borealis lists the PSG files as restricted and requires MASS access approval.")


if __name__ == "__main__":
    main()
