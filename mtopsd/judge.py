"""GPT-4o multi-turn editing judge, shared by the gate and LME-Bench.

Pass 1 (score_session), one call per turn on the session so far, scoring the LAST turn:
  prompt_following - did this turn's requested change happen (judged as a delta vs the
                     previous image)?
  consistency      - is what should be preserved preserved? (for global edits: subjects,
                     identities and composition survive the transform)
  quality          - absolute visual quality, independent of the edit.
A turn succeeds iff prompt_following > 6 and consistency > 6; a session succeeds up to turn k
iff every turn <= k succeeded (MSE-Bench-style early stop). After the first failure the
remaining turns get a quality-only call. The gate's collapse axis: a session is collapsed from
the first of >= 2 consecutive turns with quality <= 6 (absorbing).

Pass 2 (score_collapse_turn, LME-Bench only), one call per turn with the source and the
current image, answering three factual questions: content_loss c in {0,1,2},
surface_corruption s in {0,1,2}, appearance_violation a in {0,1}. A turn is destroyed iff
c + s >= 2, and a session is collapsed from the first of >= 2 consecutive destroyed turns
(absorbing). This is the collapse rate (CR) reported for LME-Bench.

Needs OPENAI_API_KEY (an OpenRouter key "sk-or-..." is routed to OpenRouter automatically;
OPENAI_BASE_URL overrides the endpoint).
"""
from __future__ import annotations

import ast
import base64
import json
import os
import re
import io
import time
import traceback
from io import BytesIO

from PIL import Image

NUM_TRY = 20

PROMPT_TEMPLATE = (
    """ Assume you are an expert in evaluating multi-turn image editing. In this task, a user interacts with an image editing system across multiple turns. At the first turn, the user provides a source image and an editing prompt. The system returns the edited image. In each subsequent turn, the user supplies a new prompt, and the system generates a new image based on the output from the previous turn.
Your goal is to evaluate how successfully the editing instruction of the LAST turn (turn-{num_prompts}) has been executed.

You will be given {num_prompts} user editing prompts and {num_images} images: the first image is the original source image, and the next are the edited results from each turn for each prompt.
You should focus more on the last prompt and the last edited image, but you may also consider the previous prompts and images as context.

The {num_prompts} user editing prompts are: {editing_prompt}

Please follow these evaluation rules. For the LAST turn, assess THREE criteria by giving a reason, then assign an integer score from 0 to 10 for each:

1) prompt_following: does the last edited image fulfill the last user's editing prompt? Judge the DELTA — compare the last edited image against the IMMEDIATELY PRECEDING image (the second-to-last image provided) and verify that the SPECIFIC change the instruction asks for actually happened, instead of merely checking whether the final image happens to contain the described element. Apply this generally to the instruction type:
- Add a quantity ("add a second / another / one more X"): the COUNT of X must INCREASE by that amount versus the previous image (e.g. one X becomes two). An X that was already there does not count — there must be a newly added one.
- Remove a quantity ("remove one / a X"): the count of X must DECREASE by one versus the previous image — not "all of the X disappeared", and not "the X is unchanged".
- Change an attribute ("change X to <color/material/style>"): that specific X must actually take on the new attribute (the existing object changes), not merely that some object with that attribute now appears.
- Add or remove a target: the specific named object must actually appear (for add) or disappear (for remove) versus the previous image.
If the last image is essentially unchanged from the previous image where the instruction demanded a change, prompt_following must be low. (0 indicates the required change did not happen at all; 10 indicates it was executed exactly as asked.)

2) consistency: are the parts that should be preserved still consistent with the input reference (the source image at the first turn, or the result image at the previous turn)?
First decide whether the last instruction is a LOCAL edit (adds, removes, or alters a specific object or region) or a GLOBAL edit (intentionally changes the whole image — lighting, time of day, weather, season, color grade, or art style, e.g. "make it nighttime", "convert to black and white", "make it a sunset scene").
- LOCAL edit: consistency rates the degree of overediting. 0 indicates the scene is needlessly different from the reference; 10 indicates a minimal yet effective edit that leaves everything else intact.
- GLOBAL edit: the whole image is SUPPOSED to change, so do NOT penalize the global change itself. Instead, consistency rates whether the main subjects, their identities, the composition, and the scene structure are preserved THROUGH the transform. 0 indicates the subjects or composition are lost, replaced, or distorted; 10 indicates the same scene and subjects faithfully re-rendered under the requested global change.

3) quality: the absolute visual quality of the last edited image, INDEPENDENT of whether the edit is correct. 0 indicates the image is badly degraded (heavy noise, rainbow or color artifacts, blur, melting or duplicated structure, extreme oversaturation, or collapse into noise); 10 indicates a clean, sharp, artifact-free image. An image may follow the prompt yet still be degraded — score this axis on its own.

4) Return your results in a JSON structure, following this format:
{{"reason": "...", "prompt_following": int, "consistency": int, "quality": int}}
"""
)


QUALITY_PROMPT = (
    """You are given several images from a multi-turn image-editing sequence. All images EXCEPT THE LAST one are earlier, cleaner reference images (the original source and, when present, the previous turn). The LAST image is the current image to evaluate. Rate ONLY the visual quality of the LAST (current) image, using the earlier image(s) as references for what un-degraded quality looks like.
This is about visual degradation, NOT about content: do NOT penalize the current image for showing different objects, scene, lighting, or composition than the references — only judge whether the current image has degraded in quality.
Give a brief reason, then a quality score from 0 to 10: 0 indicates the current image is badly degraded (heavy noise, rainbow or color artifacts, blur, melting or duplicated structure, extreme oversaturation, lost detail/sharpness, or collapse into noise); 10 indicates it is clean, sharp, and artifact-free.
Return your result as JSON: {"reason": "...", "quality": int}"""
)



def make_client(api_key):
    import openai
    base_url = os.environ.get("OPENAI_BASE_URL") or (
        "https://openrouter.ai/api/v1" if api_key.startswith("sk-or-") else None)
    return openai.OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))


def _pil_to_b64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _path_to_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def session_imgs_to_b64(image_paths):
    """[source, turn1, ..., turnN] -> base64 list; the source is resized to the turn size."""
    src = Image.open(image_paths[0]).resize(Image.open(image_paths[1]).size)
    return [_pil_to_b64(src)] + [_path_to_b64(p) for p in image_paths[1:]]


def parse_reply(s):
    """First JSON object in a model reply (robust to code fences and prose)."""
    if s is None:
        return None
    candidates = [s]
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                v = parser(c)
                if isinstance(v, dict):
                    return v
            except Exception:  # noqa: BLE001
                pass
    return None


def _call(client_info, text, b64_images):
    """One Responses-API call (3 attempts with backoff); returns the output text or None."""
    api_key, model = client_info
    client = make_client(api_key)
    content = [{"type": "input_text", "text": text}] + [
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b}"} for b in b64_images]
    delay = 1
    for attempt in range(3):
        try:
            resp = client.responses.create(
                model=model, input=[{"type": "message", "role": "user", "content": content}])
            return resp.output_text
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    return None


def _quality_only(client_info, ref_b64s, cur_b64, quality_threshold):
    for _ in range(3):
        r = parse_reply(_call(client_info, QUALITY_PROMPT, list(ref_b64s) + [cur_b64]))
        if isinstance(r, dict) and "quality" in r:
            try:
                q = int(r["quality"])
            except Exception:  # noqa: BLE001
                continue
            return {"quality": q, "degraded": int(q <= quality_threshold),
                    "reason": r.get("reason", ""), "quality_only": True}
    return None


def score_session(client_info, instructions, image_paths, success_threshold=6,
                  quality_threshold=6):
    """image_paths = [source, turn1, ..., turnN]. Returns {"turn<k>": {...}} or {"error": ...}."""
    assert len(image_paths) == len(instructions) + 1
    b64 = session_imgs_to_b64(image_paths)
    res, failed = {}, False
    for i in range(len(instructions)):
        key = f"turn{i + 1}"
        if failed:
            refs = [b64[0]] + ([b64[i]] if i >= 1 else [])   # source (+ previous turn)
            q = _quality_only(client_info, refs, b64[i + 1], quality_threshold)
            if q is not None:
                res[key] = q
            continue
        prompts = {f"turn{j + 1}": instructions[j] for j in range(i + 1)}
        text = PROMPT_TEMPLATE.format(editing_prompt=str(prompts), num_prompts=i + 1,
                                      num_images=i + 2)
        for _ in range(NUM_TRY):
            try:
                r = parse_reply(_call(client_info, text, b64[: i + 2]))
                r["all"] = int(r["prompt_following"] > success_threshold
                               and r["consistency"] > success_threshold)
                r["degraded"] = int(r["quality"] <= quality_threshold)
                res[key] = r
                break
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        else:
            return {"error": f"no valid response after {NUM_TRY} attempts at turn {i + 1}"}
        if res[key]["all"] == 0:
            failed = True
    return res


def collapse_onset(per_turn, n_turn):
    """1-based first turn of the first run of >= 2 consecutive degraded turns, or None."""
    degs = [(per_turn.get(f"turn{i}") or {}).get("degraded") for i in range(1, n_turn + 1)]
    for i in range(n_turn - 1):
        if degs[i] == 1 and degs[i + 1] == 1:
            return i + 1
    return None


def aggregate(per_sample, n_turn):
    """Per-turn success / degraded / collapsed rates + quality summaries over all sessions."""
    ps = list(per_sample.values())
    n = len(ps)
    turns = [f"turn{i}" for i in range(1, n_turn + 1)]

    def ok(t, k):
        return all((t.get(f"turn{i}") or {}).get("all", 0) == 1 for i in range(1, k + 1))

    onsets = [collapse_onset(t, n_turn) for t in ps]
    quality = {k: sum((t.get(k) or {}).get("quality", 0) or 0 for t in ps) / n for k in turns}
    return {
        "success_rate": {k: sum(ok(t, i) for t in ps) / n for i, k in enumerate(turns, 1)},
        "degraded_rate": {k: sum((t.get(k) or {}).get("degraded", 0) or 0 for t in ps) / n
                          for k in turns},
        "collapsed_rate": {k: sum(1 for o in onsets if o is not None and o <= i) / n
                           for i, k in enumerate(turns, 1)},
        "quality": quality,
        "quality_auc": sum(quality.values()) / n_turn,
        "quality_last": quality[turns[-1]],
    }


# --------------------------------------------------------------------------- pass 2
COLLAPSE_PROMPT = """You are auditing one frame of a multi-turn image-editing sequence for VISUAL DEGRADATION.

You are given the ORIGINAL SOURCE image first, and the CURRENT image last.
These editing instructions have been applied, in order, to get from the source to the current image:
{instructions}

Some of those instructions deliberately change how the picture LOOKS (repaint as pixel art, as watercolour, as an oil painting, convert to black and white, shift the colour temperature, change the season or weather). A picture that is a COMPETENT rendering in the requested style is NOT degraded, however far it is from a photograph. Blocky flat colour is correct for pixel art; visible brushstrokes are correct for an oil painting. Judge craftsmanship, not photographic realism.

Degradation is the opposite: damage no competent artist would produce. It includes heavy noise, rainbow or colour speckle, blur, melting or duplicated structure, extreme oversaturation, lost detail or sharpness, collapse into noise -- and also: UNREQUESTED flat colour patches replacing texture, hard-edged colour confetti or fragments scattered over the picture, a background that has stopped depicting a place and become an abstract pattern, and a subject reduced to a featureless silhouette.

Answer these three questions about the CURRENT image, then give a brief reason.

1. content_loss -- can the things that should be in the picture still be identified?
   0 = every main object and the setting are identifiable
   1 = at least one main object, or the setting, can no longer be made out
   2 = the picture no longer reads as the scene at all

2. surface_corruption -- is the picture's surface damaged in ways the instructions did not ask for?
   0 = none worth noting
   1 = clearly present, but the picture survives it
   2 = dominates the frame; the frame reads as noise or debris rather than a picture

3. appearance_violation -- does the current look contradict what the instructions asked for
   (for example still colourful after "convert to black and white")?
   0 = consistent with the instructions
   1 = contradicts them

Return JSON only: {{"reason": "...", "content_loss": int, "surface_corruption": int, "appearance_violation": int}}"""


def _jpeg_b64(path, max_side=768):
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def score_collapse_turn(client_info, source_path, current_path, instructions):
    """{c, s, a, reason} for one frame given the instructions applied so far, or None."""
    ins = "\n".join(f"  turn{i + 1}: {x}" for i, x in enumerate(instructions)) or "  (none)"
    imgs = [_jpeg_b64(source_path), _jpeg_b64(current_path)]
    for _ in range(3):
        r = parse_reply(_call(client_info, COLLAPSE_PROMPT.format(instructions=ins), imgs))
        if isinstance(r, dict):
            try:
                return {"c": int(r["content_loss"]), "s": int(r["surface_corruption"]),
                        "a": int(r["appearance_violation"]), "reason": r.get("reason", "")}
            except Exception:  # noqa: BLE001
                continue
    return None


def lme_report(per_sample, collapse_scores, n_turn, n_total):
    """LME-Bench metrics per turn: SR (pass 1) and CR (pass 2, c + s >= 2 on >= 2 consecutive
    turns, absorbing). collapse_scores = {id: {turn(int): {"c", "s", ...}}}."""
    turns = [f"turn{i}" for i in range(1, n_turn + 1)]

    def ok(t, k):
        return all((t.get(f"turn{i}") or {}).get("all", 0) == 1 for i in range(1, k + 1))

    def destroyed(e):
        return e is not None and e["c"] + e["s"] >= 2

    onsets = []
    for d in collapse_scores.values():
        onsets.append(next((t for t in range(1, n_turn)
                            if destroyed(d.get(t)) and destroyed(d.get(t + 1))), None))
    return {
        "success_rate": {k: sum(ok(t, i) for t in per_sample.values()) / n_total
                         for i, k in enumerate(turns, 1)},
        "collapse_rate": {k: sum(1 for o in onsets if o is not None and o <= i) / n_total
                          for i, k in enumerate(turns, 1)},
    }
