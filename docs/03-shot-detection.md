# 3. Shot detection

Code: [`src/frameseek/shots.py`](../src/frameseek/shots.py)

## What we are detecting

A **shot** is a continuous run of frames from one camera take. Shots are joined by
transitions, and some things that are not transitions look like them.

| Transition | On screen | Difficulty |
|---|---|---|
| Hard cut | Frame N is shot A, frame N+1 is shot B | Easy: one huge change between two frames |
| Fade / dissolve | A blends into B over 10 to 30 frames | Hard: the change is spread out, each step is small |
| Wipe / graphic | B slides across A | Medium |

| Not a cut, but looks like one | Why it fools a detector |
|---|---|
| Fast pan or whip | Every frame differs a lot from the previous one |
| Flash or strobe (camera flash, stage lights, aarti fire) | One or two frames change completely, then the shot continues |
| Something passing close to the lens | A large region changes suddenly |

## How detection works

For every pair of consecutive frames, compute a **change score**, then decide
where the score indicates a cut.

| Detector (PySceneDetect) | Decision rule | Good at | Weak at |
|---|---|---|---|
| ContentDetector | Score above a **fixed threshold** (default 27). The score is the average change in hue, saturation and brightness (HSV). | Simple, predictable | Fast camera motion raises every score and causes false cuts |
| AdaptiveDetector | Score is several times **larger than its neighbours' average** (default 3x), and above a minimum | Ignores sustained motion such as pans | Still fooled by single-frame flashes |

Both detectors also take a **minimum shot length**: once a cut is found, no new
cut can start for that many frames. A flash shows up as two spikes in a row (the
flash, then the return to normal) followed by the same shot, so a minimum length
of around 0.5 s collapses the pair and stops the flash from creating a tiny "shot".

Dissolves remain hard for both: the change is spread across many frames, so no
single frame stands out.

## Which mistake is cheaper

No threshold catches every real cut with zero false alarms. That is the
**precision / recall trade-off**: precision is "of the cuts I reported, how many
were real", recall is "of the real cuts, how many did I report".

For Frameseek the two kinds of error cost very different amounts:

| Mistake | Cost |
|---|---|
| False alarm (extra cut) | One extra keyframe to caption (a few seconds of compute) and a near-duplicate in the index. Harmless. |
| Missed cut | Two different shots share one keyframe, so one of them is invisible to search. If the answer is only in that shot, Frameseek can never find it. |

So Frameseek tunes for **recall** and accepts a few extra keyframes.

## Choosing keyframes

- **Not the first frame** of a shot: it often still contains the transition or motion blur.
- **The middle frame** is the default for short shots.
- **Long shots get more than one.** A 90-second single take can show completely
  different things at its start, middle and end. Shots longer than `every_s`
  (10 s by default) are split into equal parts with a keyframe at the centre of each.

`plan_keyframe_times(0, 25, every_s=10)` returns `[4.167, 12.5, 20.833]`: three
parts of about 8.3 s, a frame in the middle of each.

This also covers missed cuts: if a dissolve inside a long stretch is not
detected, the extra frames still capture both sides of it.

## Making it fast

Shot detection must decode every frame, because cuts do not line up with
I-frames. Two choices keep it cheap:

1. **Analyse at low resolution.** `auto_downscale` makes PySceneDetect work on
   roughly 256 px wide frames. Cuts are obvious even when tiny, and the work
   drops by roughly 10x.
2. **Extract only the chosen keyframes at full resolution**, each with a fast
   seek (`ffmpeg -ss <t> -i video.mp4`, see Chapter 2).

A tempting shortcut is to treat the video's I-frames as cut positions, since
YouTube's encoder places I-frames at scene changes. It is only a hint: encoders
also insert periodic I-frames and miss soft transitions.

## Caching with the settings as part of the key

`shots.json` stores the detection settings alongside the result:

```json
{
  "params": {"adaptive_threshold": 3.0, "min_content_val": 15.0, "min_shot_s": 0.6, "every_s": 10.0},
  "shots": [{"index": 0, "start_s": 0.0, "end_s": 4.0, "keyframes": [{"t": 2.0, "file": "kf_0000_00.jpg"}]}]
}
```

If any setting in `config.yaml` changes, the stored params no longer match and
the step re-runs. A cache that ignores its inputs silently returns stale results.

## Checking the result by eye

`frameseek shots` writes `contact_sheet.jpg`: every keyframe in a grid, labelled
with shot number and time. It is the quickest way to spot problems such as
several near-identical frames in a row (false cuts from flashes or motion) or a
frame that clearly mixes two scenes (a missed dissolve).

## Check your understanding

1. A dhunuchi naach scene has camera flashes going off. With a low fixed
   threshold, what happens to the output, and which two settings would you change?
2. A 90-second single take moves from the pandal entrance to the idol to the
   crowd. With one middle keyframe per shot, which question could never be
   answered, and what fixes it?
3. Why does Frameseek prefer extra cuts over missed ones?
