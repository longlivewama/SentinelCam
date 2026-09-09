# Video-level validation

> ## Status
>
> **Evaluation framework ready; independent real-world video validation pending labelled footage.**
>
> Everything in this directory works and runs against the real production inference path. What
> does not exist yet is an independent, labelled, real-world corpus meeting the
> [dataset specification](#the-dataset-this-needs) below. Until one does, **no claim about this
> detector's operational accuracy is supported by evidence.** Any report generated here describes
> the clips it was given and nothing more — and says so, on its own front page, generated from
> the corpus rather than written by hand.
>
> Nothing here fabricates a number. A metric that cannot be computed prints `n/a`, never `0`.

The fall detector's published metrics — precision 0.846, recall 0.800, mAP@50 0.877 — are
**per-frame, on still images**. The product's claim is different: *a fall that happens raises an
alert, quickly, and ordinary activity doesn't.* Those are separate quantities, and until this
framework existed nothing in the repository measured the second set.

This directory measures them. It drives the **real production inference path** over a labelled
corpus of short clips and reports:

| | |
|---|---|
| **Incident recall** | of the falls that happened, how many raised an alert |
| **Incident precision** | of the alerts raised, how many were real falls |
| **Incident F1** | harmonic mean of the two, for ranking threshold settings |
| **True / false positives, false negatives** | the raw counts behind every rate above |
| **False alerts / hour** | over ordinary-activity footage — the operator's nuisance rate |
| **Detection latency** | seconds from the fall starting to the alert firing |
| **Missed falls** | each one, with its clip, timestamp and fall type |
| **False positives** | each one, grouped by what activity produced it |
| **Confidence distributions** | scores of true vs false detections, with a histogram |
| **Per-condition results** | recall/precision/F1 split by lighting, angle, distance, occlusion, resolution |
| **Recall by fall type** | split by direction (forward/backward/sideways) and speed (slow/fast) |
| **Threshold sweep** | the full grid, exact rather than approximated |
| **Corpus coverage** | which required buckets the corpus does *not* yet contain |

F1 is reported alongside precision and recall, never instead of them. For this product the two
errors are not interchangeable: a missed fall is someone lying on a floor unnoticed, a false alert
is an operator checking a camera for nothing. F1 weighting them equally is a convenience for
ranking configurations, not a claim that they cost the same.

---

## Quick start

**Fastest path — a published corpus, fetched for you:**

```bash
cd backend && source venv/bin/activate && cd ..
python -m ml.validation.fetch_urfd     # 10 falls + 10 ADL from the UR Fall Detection Dataset
python -m ml.validation.evaluate
```

`fetch_urfd` downloads the clips, keeps the RGB half of each published depth|RGB composite, and
derives each fall's timing from the dataset's own accelerometer peak rather than from anybody's
eyeballing. The footage is CC BY-NC-SA 4.0 (non-commercial, academic; cite Kwolek & Kepski 2014)
and is **not** redistributed here — see the script's docstring.

That corpus is one dataset, one viewpoint, two rooms and one subject, so it is a starting point
rather than a substitute for footage from your own deployment.

**Or bring your own clips:**

```bash
# 1. Create the corpus directory (it is gitignored — see Privacy below)
mkdir -p ml/data/validation/videos

# 2. Put your clips in ml/data/validation/videos/
# 3. Copy the annotation template and fill it in
cp ml/validation/annotations.example.json ml/data/validation/annotations.json

# 4. Run
cd backend && source venv/bin/activate && cd ..
python -m ml.validation.evaluate            # at production defaults
python -m ml.validation.evaluate --sweep    # plus the threshold grid
```

Outputs `ml/reports/fall_detector_video_validation.{md,json}`.

The runner needs `ultralytics` and `opencv-python`, which the **backend** venv already has —
run it from that environment, not `ml/`'s.

---

## Corpus layout

```
ml/data/validation/          ← gitignored in its entirety
├── videos/
│   ├── fall_kitchen_01.mp4
│   ├── hardneg_sitting_01.mp4
│   └── …
├── annotations.json
└── .detection_cache/        ← auto-managed, safe to delete
```

**Target: 20–50 clips, roughly half falls and half hard negatives.** Short clips (15–60 s) are
better than long ones: they are quicker to annotate accurately, and annotation accuracy is what
bounds the quality of every number here.

### Annotation format

One JSON file. Times are seconds from the start of the clip. Full template:
[`annotations.example.json`](annotations.example.json).

```json
{
  "videos": [
    {
      "filename": "fall_kitchen_01.mp4",
      "duration_seconds": 22.5,
      "category": "fall",
      "incidents": [
        {"id": "kitchen-01", "start_seconds": 8.2, "end_seconds": 10.0,
         "notes": "trips on rug, lands on side, stays down"}
      ]
    },
    {
      "filename": "hardneg_sitting_01.mp4",
      "duration_seconds": 18.0,
      "category": "hard_negative",
      "activity": "sitting",
      "incidents": [],
      "activity_intervals": [
        {"start_seconds": 3.0, "end_seconds": 7.5, "label": "sits down heavily"}
      ]
    }
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `filename` | ✅ | Must exist under `videos/` |
| `duration_seconds` | ✅ | Cross-checked against the decoded file; a >1 s disagreement is flagged |
| `category` | ✅ | `fall` or `hard_negative` |
| `incidents` | ✅ | One entry per real fall. **Must** be non-empty for `fall`, empty for `hard_negative` |
| `incidents[].start_seconds` | ✅ | When the fall *begins* — latency is measured from here |
| `incidents[].end_seconds` | ✅ | When the person is down |
| `incidents[].id` / `notes` | | Optional; `id` is auto-generated if omitted |
| `incidents[].direction` | | `forward` \| `backward` \| `sideways`. Drives recall-by-fall-type |
| `incidents[].speed` | | `slow` \| `fast`. Same |
| `activity` | | Hard negatives only. Drives the false-positive breakdown |
| `activity_intervals` | | Optional. Lets the report say *which moment* of a clip misfired |
| `conditions` | | Optional object keyed by `lighting`, `camera_angle`, `distance`, `occlusion`, `resolution`. Drives the per-condition breakdown |

Multiple falls in one clip are supported — add multiple `incidents`, and they may have different
`direction`/`speed`: a clip where someone falls forward and then, getting up, falls backward is
two different tests of the detector.

**The `category` field is checked, not trusted.** A `fall` clip with no annotated incident, or a
`hard_negative` with one, raises an error at load time rather than silently corrupting the metrics.

`direction` and `speed` are validated against their closed sets, and unknown `conditions` keys are
rejected. A typo of `"backwards"` or `"lightning"` would otherwise create a bucket of its own, and
the coverage check below — whose whole job is to say *"you have no backward falls yet"* — would
quietly report that you do. `conditions` **values** are free text, because deployments differ:
`1080p`, `4k` and `cif` are all legitimate answers to `resolution`.


---

## The dataset this needs

The framework is finished. The corpus is not. This section is the specification it is measured
against — `Corpus.missing_coverage()` implements it, the report prints what is missing in
section 1, and `python -m ml.validation.evaluate` prints it on the console. That check is
mechanical on purpose: a corpus is not adequate because it is *large*, it is adequate when it
exercises each failure mode the detector is claimed to handle.

**Overall target: 20–50 clips minimum, ≈30 annotated fall incidents, and hours (not minutes) of
hard-negative footage.** Thirty is where a single missed fall stops moving recall by several
points. Below it, the report leads with a "preliminary" banner and differences between threshold
settings are mostly noise.

### Falls

Every fall must be annotated with a `direction` and a `speed`, and the corpus must contain at
least one of each value — ideally several.

| Dimension | Required values | Why it is a separate bucket |
|---|---|---|
| `direction` | `forward` | The most common and the best represented in public data — likely the detector's easiest case |
| | `backward` | Different silhouette entirely; often ends with the subject partly under furniture |
| | `sideways` | The posture closest to "lying down", so the hardest to separate from a hard negative |
| `speed` | `fast` | A trip or collapse. Brief transition, then a sustained on-ground posture |
| | `slow` | A controlled slide down a wall or off a chair. **The case the sustained-duration gate is least suited to** — the transition is gradual, so the moment the fall "starts" is ambiguous to the model as well as the annotator |

A single overall recall figure averages these together, which is exactly how a detector that never
sees backward falls passes validation.

### Hard negatives

Each of these must be present, as its own `activity`. They are not decoration: every one produces
the visual signature the detector fires on — **a person, horizontal or low, sustained** — and the
frame-level training data contains none of them.

| `activity` | What it probes |
|---|---|
| `exercising` | **Highest priority.** Sit-ups and floor work are sustained on-ground posture that defeats the duration gate outright |
| `lying_sofa` | Horizontal, sustained, entirely ordinary |
| `sleeping` | The same, for hours, and in a care setting the most common state of all |
| `sitting` | The most frequent transition in any indoor footage |
| `bending` | Torso horizontal, feet planted — a fall's silhouette without a fall |
| `crouching` | Brief on-ground posture; probes whether the gate works at all |
| `picking_up` | Reaching to the floor and back, repeatedly |
| `dropped_object` | Motion at floor level with no person on the ground — tests whether the model keys on the person or the movement |
| `pet` | A cat or dog is a small horizontal shape at floor level |
| `occluded` | A person partly hidden by furniture, which truncates the bounding box the way a fall does |

Also useful, not required: `standing_up`, `kneeling`, `chair_transfer`, `stretching`,
`floor_activity`, `lying_bed`, `walking`.

### Environmental variation

Each dimension needs **at least two distinct values**, or there is nothing to compare across and
the per-condition table stays empty. Suggested values, but any string works:

| Dimension | Suggested values | Why it matters |
|---|---|---|
| `lighting` | `daylight`, `dim`, `night_ir`, `mixed` | IR footage is greyscale and differently textured; the training data is daylight RGB |
| `camera_angle` | `ceiling`, `high_corner`, `eye_level`, `low` | A ceiling camera sees a fall as a shape change, a corner camera sees a silhouette drop. Almost all public fall data is one angle |
| `distance` | `near`, `mid`, `far` | A distant subject is a handful of pixels; small-object recall is where detectors fail first |
| `occlusion` | `none`, `partial`, `heavy` | Furniture truncates the box, which is also what a fall does |
| `resolution` | `1080p`, `720p`, `480p`, … | Deployments run whatever cameras they already own |

### Independence

The corpus must share **no footage and no source** with the training data — see
[Separation from training data](#separation-from-training-data) below. Overlap turns every number
into a measurement of memorisation.

### What "ready" means

`missing_coverage()` returning empty is a **necessary** condition, not a sufficient one. It says
the corpus contains at least one example of each bucket. It does not say the sample is large
enough for a confidence interval anyone should act on, and it cannot say whether the footage
resembles a particular deployment. Both remain judgements about the data, and the report's
Limitations section states them on every run.

---

### Annotating well

- `start_seconds` is **loss of balance**, not impact. Latency is measured from it, so a consistent
  convention matters more than a precise one.
- `end_seconds` is when the person is on the ground. The matching window extends 30 s past it.
- If you can't tell whether something is a fall, **leave the clip out**. An ambiguous label is
  worse than a smaller corpus.

---

## Why hard negatives matter more than more fall clips

The frame-level test split contains 1,495 background images — all PASCAL VOC, all ordinary
photographs of upright people. It contains **nothing** resembling:

sit-ups · crouching to reach a low shelf · a child playing on the floor · lying on a sofa ·
kneeling to tie a shoe · yoga · stretching · getting down to plug something in

Every one of those produces the exact visual signature the detector was trained to fire on: **a
person, horizontal, at floor level, for a sustained period.** The sustained-duration gate — the
main false-positive guard — is specifically defeated by them, because unlike a brief crouch they
*persist*.

So the frame-level precision of 0.846 says almost nothing about the false-alert rate in a real
room. That is the gap this corpus exists to close, and it is why a hard negative is worth more
than another fall clip at the margin.

Suggested composition:

| Activity | Why |
|---|---|
| `exercising` | **Highest priority.** Sit-ups and floor exercise are sustained on-ground posture |
| `lying_sofa`, `lying_bed` | Horizontal and sustained, but not a fall |
| `crouching`, `kneeling`, `bending` | Brief on-ground posture — probes whether the gate works |
| `sitting`, `standing_up`, `chair_transfer` | The most common motion in a care setting |
| `floor_activity` | Reaching under furniture, playing with a pet, picking things up |
| `stretching` | Slow transitions through fall-like postures |

---

## The matching rule

A prediction is a single instant (when the sustain gate fired). Ground truth is an interval. A
prediction at time *t* matches incident `[start, end]` when:

```
start − 0.5s  ≤  t  ≤  end + 30s
```

The asymmetry is the point:

- **0.5 s before.** Firing before the fall begins isn't early detection — the evidence didn't
  exist yet. The small window absorbs annotation jitter, nothing more.
- **30 s after.** A late alert is still a *correct* alert. The product premise is that harm comes
  from lying unnoticed, so detecting at +15 s is a success, just a slower one. Bounded rather than
  infinite so a detection in a later, unrelated part of a clip isn't credited to an earlier fall.

Matching is one-to-one and greedy by time: each incident takes the **earliest** eligible
prediction (latency is what an operator waits for the *first* alert). Extra predictions inside the
same incident's window are the same fall re-firing past the debounce — counted separately as
`duplicate_detections`, not as false alerts. Predictions matching no incident are false alerts;
incidents matching no prediction are missed falls.

Implementation and full rationale: [`metrics.py`](metrics.py).

---

## What is actually being measured

The runner imports `app.services.detection.*` from the backend and calls
`video_scan.scan_video` — **the same loop the upload analyser runs**, with the same `FallPipeline`,
the same `ModelFallDetector` gate, and the same trained checkpoint. There is no evaluation-only
detector, and `verify_pipeline()` refuses to run if:

- `FALL_DETECTOR_MODEL_PATH` is unset or missing
- the checkpoint's classes aren't exactly `{0: 'Fall'}`
- the file looks like a stock Ultralytics checkpoint (`yolov8n.pt` and friends)
- the resolved fall-detection mode isn't `model`

The model's path and sha256 are printed in every report, so a result can't be read without knowing
what produced it.

### Why the threshold sweep is exact, not approximate

Re-running YOLO for every grid point would cost *grid size × corpus hours* of inference. It's
avoided without approximation:

1. `FALL_DETECTOR_MIN_SUSTAINED_SECONDS` never reaches the model — it's consumed by
   `ModelFallDetector`, downstream of inference.
2. `FALL_DETECTOR_MIN_CONFIDENCE` reaches the model only as YOLO's `conf=`, which drops boxes
   **before** NMS. NMS keeps the highest-scoring box of an overlapping group, so a box below *C*
   can never suppress one above *C*. The surviving set at score ≥ *C* is therefore identical
   whether inference ran at `conf=C` or at any lower `conf` followed by filtering.

So inference runs **once per video** at the lowest confidence in the grid, and each grid point
replays the cached boxes through a fresh gate. Bit-identical to a full re-run.

The cache is keyed on video size + mtime **and the model's sha256**, so editing a clip or swapping
the checkpoint invalidates it automatically.

### `--run-pose`

Off by default. In `model` mode `FallPipeline` feeds pose output only to the heuristic, which isn't
running — so pose inference changes no fall event and is pure cost. Pass `--run-pose` to run it
exactly as production does; the report records which was used.

---

## Privacy and data handling

**Never commit validation footage.** `ml/data/validation/` is gitignored in its entirety, and CI
fails on any committed file over 10 MB.

Video of people falling is personal data, and often health-related personal data:

- **Consent.** Get explicit, recorded consent from everyone appearing in a clip, covering use as
  test data. Staged falls by consenting adults are far easier to justify than real incident footage.
- **Real incident footage** from a care setting is special-category data under GDPR and similar
  regimes. Do not use it without a documented lawful basis and your DPO's sign-off.
- **Minimisation.** Short clips of the relevant moment, not hours of continuous recording.
- **Storage.** Keep the corpus on an encrypted volume with restricted access, and set a retention
  period.
- **Sharing.** Don't attach clips to issues, PRs or bug reports. Share the `annotations.json` and
  the generated report — they contain no imagery.

Filenames are echoed in the report, so **don't put names or room numbers in them**. Use
`fall_kitchen_01.mp4`, not `fall_mrs_smith_room_12.mp4`.

---

## Separation from training data

This corpus must share **no footage and no source** with the training data
(`ml/data/fall_detection_prepared/`, from the Roboflow `voc-fall-person_falls` export).

The training set is stills scraped from public fall datasets and PASCAL VOC. Any overlap would
make these numbers a measurement of memorisation. Concretely:

- Don't source validation clips from public fall-detection datasets — assume anything on Roboflow
  or Kaggle may already be in training.
- Prefer footage you recorded yourself, ideally at the target camera height and lighting.
- Record the provenance of each clip in its `notes`.

The one legitimate exception: the frame-level held-out **test** split
(`ml/data/fall_detection_prepared/test/`) never touched training or model selection, so it remains
valid for frame-level claims. It is stills, so it cannot answer any question this directory asks.
