from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


ISSUE_PATTERNS = [
    (r"\bglass shatter\b|\bshattered\b|\b shattered\b|\bglass broken\b", "glass_shatter"),
    (r"\bcrack\b|\bcracked\b", "crack"),
    (r"\bdent\b|\bdented\b", "dent"),
    (r"\bscrape(?:d|s)?\b|\bscratch\b|\bscraped\b|\bscuffed\b", "scratch"),
    (r"\bmissing\b|\bnot inside\b|\bnot in\b", "missing_part"),
    (r"\btorn\b|\bripped\b|\bopened\b|\bopen flap\b|\bopen seal\b", "torn_packaging"),
    (r"\bcrushed\b|\bcrush\b|\bcollapsed\b", "crushed_packaging"),
    (r"\bwater damage\b|\bwet\b|\bwater stain\b|\bliquid damage\b", "water_damage"),
    (r"\bstain\b|\bspilled\b|\bsticky\b", "stain"),
    # fallback: generic "broken" or "damage" treated as broken_part after more specific patterns
    (r"\bbroken\b|\bbroken part\b|\bdamaged\b|\bdamage\b", "broken_part"),
    (r"\bno damage\b|\bnot damaged\b|\bdoes not show damage\b|\bno visible damage\b", "none"),
]

PART_PATTERNS = {
    "car": [
        (r"\brear bumper\b|\bback bumper\b|\brear end\b", "rear_bumper"),
        (r"\bfront bumper\b|\bbefore bumper\b|\bbumper\b", "front_bumper"),
        (r"\bside mirror\b|\bmirror\b", "side_mirror"),
        (r"\bdoor\b|\bdoor panel\b", "door"),
        (r"\bhood\b|\bbonnet\b|\bfront hood\b", "hood"),
        (r"\bwindshield\b|\bwind screen\b|\bfront glass\b", "windshield"),
        (r"\bheadlight\b|\bhead light\b", "headlight"),
        (r"\btaillight\b|\btail light\b", "taillight"),
        (r"\bfender\b", "fender"),
        (r"\bquarter panel\b|\bquarter\b", "quarter_panel"),
        (r"\bbody\b|\bpanel\b|\bside\b", "body"),
    ],
    "laptop": [
        (r"\bhinge\b", "hinge"),
        (r"\bkeyboard\b|\bkeys\b", "keyboard"),
        (r"\bscreen\b|\bdisplay\b|\bmonitor\b", "screen"),
        (r"\btrackpad\b|\btouchpad\b", "trackpad"),
        (r"\blid\b|\blaptop lid\b", "lid"),
        (r"\bcorner\b", "corner"),
        (r"\bport\b|\busb\b|\bhdmi\b|\bcharging port\b", "port"),
        (r"\bbase\b|\bbody\b|\bchassis\b", "base"),
    ],
    "package": [
        (r"\bpackage corner\b|\bcorner\b", "package_corner"),
        (r"\bseal\b|\btape\b|\bopen flap\b", "seal"),
        (r"\bpackage side\b|\bside\b|\bsurface\b", "package_side"),
        (r"\blabel\b", "label"),
        (r"\bcontents\b|\bitem\b|\bproduct\b|\binside\b", "contents"),
        (r"\bbox\b|\bpackage\b|\bshipping box\b", "box"),
    ],
}

UNCERTAIN_PATTERNS = [
    r"\bnot sure\b",
    r"\bmaybe\b",
    r"\bI think\b",
    r"\bit looks like\b",
    r"\bprobably\b",
    r"\bI was confused\b",
    r"\bI am not fully sure\b",
]

LOW_SEVERITY_PATTERNS = [
    r"\bminor\b",
    r"\blight\b",
    r"\bsmall\b",
    r"\bslight\b",
    r"\bnot major\b",
    r"\bnot too bad\b",
    r"\bjust a scratch\b",
]

HIGH_SEVERITY_PATTERNS = [
    r"\bsevere\b",
    r"\bbad\b",
    r"\bhorrible\b",
    r"\bserious\b",
    r"\bpretty bad\b",
    r"\bbroken\b",
    r"\bsmashed\b",
    r"\bshattered\b",
    r"\bcrushed\b",
]

QUALITY_PATTERNS = [
    (r"\bblurry\b|\bblurred\b|\bunclear\b|\bnot clear\b", "blurry_image"),
    (r"\bglare\b|\breflection\b|\blow light\b|\bdark\b|\bpoor lighting\b", "low_light_or_glare"),
    (r"\bcropped\b|\bobstructed\b|\bcut off\b", "cropped_or_obstructed"),
    (r"\bwrong angle\b|\bside view\b|\bangle\b", "wrong_angle"),
    (r"\bwrong object\b|\bdifferent car\b|\bdifferent package\b", "wrong_object"),
    (r"\bwrong part\b|\bdifferent part\b", "wrong_object_part"),
    (r"\bnot visible\b|\bcan\'t see\b|\bdamage not visible\b", "damage_not_visible"),
]

# phrases indicating possibly non-original or edited images
NON_ORIGINAL_PATTERNS = [
    r"non[- ]?original",
    r"not original",
    r"downloaded image",
    r"from internet",
    r"stock photo",
    r"from website",
    r"from web",
    r"google image",
    r"screenshot",
    r"edited",
    r"photoshop",
    r"not my photo",
]


@dataclass
class ClaimPrediction:
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str
    evidence_standard_met: str
    evidence_standard_met_reason: str
    risk_flags: str
    issue_type: str
    object_part: str
    claim_status: str
    claim_status_justification: str
    supporting_image_ids: str
    valid_image: str
    severity: str


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def first_match(text: str, patterns: Sequence[tuple[str, str]], default: str = "unknown") -> str:
    for pattern, label in patterns:
        if re.search(pattern, text):
            return label
    return default


def find_matching_pattern(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def parse_issue_type(claim_text: str) -> str:
    text = normalize(claim_text)

    if re.search(r"\bcrack(ed)?\b", text):
        return "crack"
    elif re.search(r"\bdent(ed)?\b", text):
        return "dent"
    elif re.search(r"\bstain\b|\bspilled\b|\bsticky\b", text):
        return "stain"
    elif re.search(r"\btorn\b|\bripped\b|\bopen(ed)?\b|\bseal\b", text):
        return "torn_packaging"
    elif re.search(r"\bcrush(ed)?\b|\bcollapsed\b", text):
        return "crushed_packaging"
    elif re.search(r"\bglass shatter\b|\bshatter(ed)?\b|\bsmashed\b", text):
        return "glass_shatter"
    elif re.search(r"\bmissing\b|\bnot inside\b|\bnot in\b", text):
        return "missing_part"
    elif re.search(r"\bwater damage\b|\bwet\b|\bliquid damage\b", text):
        return "water_damage"
    elif re.search(r"\bscrape(?:d|s)?\b|\bscratch\b|\bscuffed\b", text):
        return "scratch"
    elif re.search(r"\bbroken\b|\bdamaged\b|\bbroken part\b", text):
        return "broken_part"
    else:
        return "unknown"


def parse_object_part(claim_text: str, claim_object: str) -> str:
    text = normalize(claim_text)
    if claim_object not in PART_PATTERNS:
        return "unknown"
    for pattern, label in PART_PATTERNS[claim_object]:
        if re.search(pattern, text):
            return label
    # Additional explicit checks to prefer specific parts over generic matches
    if claim_object == "car":
        # rear bumper should win if 'rear' is present
        if re.search(r"\brear\s+bumper\b|\brear bumper\b", text):
            return "rear_bumper"
        # prefer windshield / glass
        if re.search(r"\bwindshield\b|\bglass\b|\bfront glass\b", text):
            return "windshield"
        # if 'front' explicitly mentioned with bumper
        if re.search(r"\bfront\s+bumper\b|\bfront bumper\b", text):
            return "front_bumper"
        # bare 'bumper' without rear qualifier -> treat as front_bumper
        if re.search(r"\bbumper\b", text):
            return "front_bumper"
        # door panel specifically
        if re.search(r"\bdoor panel\b", text):
            return "door"
        if re.search(r"\bdoor\b", text):
            return "door"
        if re.search(r"\bhood\b|\bbonnet\b", text):
            return "hood"
        if re.search(r"\bheadlight\b|\bhead light\b", text):
            return "headlight"
        if re.search(r"\btaillight\b|\btail light\b", text):
            return "taillight"
    if claim_object == "laptop":
        if re.search(r"\bscreen\b|\bdisplay\b", text):
            return "screen"
        if re.search(r"\bkeyboard\b", text):
            return "keyboard"
        if re.search(r"\btrackpad\b|\btouchpad\b", text):
            return "trackpad"
    if claim_object == "package":
        if re.search(r"\bcontents\b|\bitem\b|\binside\b", text):
            return "contents"
        if re.search(r"\bcorner\b", text):
            return "package_corner"
        if re.search(r"\bseal\b|\btape\b", text):
            return "seal"
        if re.search(r"\blabel\b", text):
            return "label"
    return "unknown"


def parse_severity(claim_text: str, issue_type: str, claim_object: str = "", object_part: str = "") -> str:
    text = normalize(claim_text)

    if issue_type == "glass_shatter":
        return "high"
    elif issue_type in {"crack", "broken_part", "water_damage", "crushed_packaging"}:
        return "medium"
    elif issue_type == "dent":
        return "medium"
    elif issue_type == "scratch":
        return "low"
    elif issue_type == "stain":
        return "low"

    # Explicit overrides
    if issue_type == "torn_packaging" and object_part == "seal":
        return "medium"
    elif issue_type == "stain" and claim_object == "laptop" and object_part == "keyboard":
        return "medium"
    elif issue_type == "dent" and claim_object == "package" and object_part == "package_corner":
        return "low"
    elif issue_type == "crack" and claim_object == "laptop" and object_part == "screen":
        return "medium"

    # Textual modifiers
    elif find_matching_pattern(text, HIGH_SEVERITY_PATTERNS):
        return "high"
    elif find_matching_pattern(text, LOW_SEVERITY_PATTERNS):
        return "low"
    elif issue_type == "none":
        return "none"

    # Object-part overrides
    elif object_part == "rear_bumper":
        return "high"
    elif object_part == "side_mirror":
        return "medium"
    elif object_part == "windshield" and issue_type == "crack":
        return "low"
    elif claim_object == "laptop" and object_part == "hinge":
        return "high"
    else:
        return "unknown"


def parse_risk_flags(
    claim_text: str,
    history_flags: str,
    issue_type: str,
    object_part: str,
    valid_image: bool,
    existing_images: Optional[Sequence[str]] = None
) -> List[str]:
    flags: List[str] = []
    text = normalize(claim_text)

    # Include history flags first
    if history_flags and history_flags != "none":
        flags.extend([flag.strip() for flag in history_flags.split(";") if flag.strip()])

    # Apply quality patterns (blurry, wrong angle, damage not visible, etc.)
    for pattern, flag in QUALITY_PATTERNS:
        if re.search(pattern, text) and flag not in flags:
            flags.append(flag)

    # Uncertainty cues → claim mismatch
    if find_matching_pattern(text, UNCERTAIN_PATTERNS) and "claim_mismatch" not in flags:
        flags.append("claim_mismatch")

    # If both issue and part unknown but image exists → damage not visible
    if issue_type == "unknown" and object_part == "unknown" and valid_image:
        if "damage_not_visible" not in flags:
            flags.append("damage_not_visible")

    # If image invalid/missing → manual review
    if not valid_image and "manual_review_required" not in flags:
        flags.append("manual_review_required")

    # Escalate to manual review if user history risk
    if "user_history_risk" in (history_flags or "") and "manual_review_required" not in flags:
        flags.append("manual_review_required")

    # Detect non-original image hints in text
    for p in NON_ORIGINAL_PATTERNS:
        if re.search(p, text):
            if "non_original_image" not in flags:
                flags.append("non_original_image")
            if "manual_review_required" not in flags:
                flags.append("manual_review_required")

    # Inspect image filenames for non-original indicators
    if existing_images:
        for img in existing_images:
            name = Path(img).name.lower()
            if any(k in name for k in ("stock", "screenshot", "download", "google", "internet", "sample")):
                if "non_original_image" not in flags:
                    flags.append("non_original_image")
                if "manual_review_required" not in flags:
                    flags.append("manual_review_required")

    # Remove 'none' if other flags are present
    if "none" in flags and len(flags) > 1:
        flags = [f for f in flags if f != "none"]

    return flags or ["none"]



def resolve_image_path(base_dir: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if candidate.exists():
        return candidate
    direct = base_dir / image_path
    if direct.exists():
        return direct
    dataset_candidate = base_dir / "dataset" / image_path
    if dataset_candidate.exists():
        return dataset_candidate
    return candidate


def validate_images(base_dir: Path, image_paths: Sequence[str]) -> tuple[bool, List[str], List[str]]:
    existing_paths: List[str] = []
    missing: List[str] = []
    for image_path in image_paths:
        resolved = resolve_image_path(base_dir, image_path)
        if resolved.exists():
            existing_paths.append(str(resolved))
        else:
            missing.append(image_path)
    return len(missing) == 0, existing_paths, missing


def images_consistency_check(image_files: List[str]) -> Optional[bool]:
    """Return True if consistent, False if inconsistent, None if check unavailable."""
    if len(image_files) < 2:
        return None
    avgs = []
    for p in image_files:
        try:
            if PIL_AVAILABLE:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im = im.resize((64, 64))
                    pixels = list(im.getdata())
                    r = sum(px[0] for px in pixels) / len(pixels)
                    g = sum(px[1] for px in pixels) / len(pixels)
                    b = sum(px[2] for px in pixels) / len(pixels)
                    avgs.append((r, g, b))
            else:
                # fallback: use file size as a weak proxy
                sz = Path(p).stat().st_size
                avgs.append((float(sz), 0.0, 0.0))
        except Exception:
            return None
    # compute average pairwise distance
    import math
    total = 0.0
    count = 0
    for i in range(len(avgs)):
        for j in range(i + 1, len(avgs)):
            dr = avgs[i][0] - avgs[j][0]
            dg = avgs[i][1] - avgs[j][1]
            db = avgs[i][2] - avgs[j][2]
            dist = math.sqrt(dr * dr + dg * dg + db * db)
            total += dist
            count += 1
    if count == 0:
        return None
    avg_dist = total / count
    # If using real color averages (PIL), compare against absolute threshold
    if PIL_AVAILABLE:
        return avg_dist <= 40.0
    else:
        # fallback used file sizes in the first channel; compute relative difference
        mean_sz = sum(a[0] for a in avgs) / len(avgs)
        if mean_sz <= 0:
            return None
        rel = avg_dist / mean_sz
        # if average pairwise size difference > 100% of mean file size, consider inconsistent
        return rel <= 1.0


def build_justification(claim_text: str, evidence_standard_met: bool, issue_type: str, object_part: str, supporting_images: Sequence[str], claim_status: str) -> str:
    if claim_status == "not_enough_information":
        return "The submitted images do not provide enough evidence to verify the claimed issue."
    if claim_status == "contradicted":
        if issue_type != "unknown" and object_part != "unknown":
            return f"The images show {issue_type.replace('_', ' ')} on the {object_part}, which contradicts the claim."
        else:
            return "The images contradict the claim or do not match the described issue."
    if claim_status == "supported":
        if issue_type == "none":
            return "The images show the object but do not show visible damage matching the claim."
        if issue_type == "unknown":
            return "The claim is unclear from the conversation and the images do not provide a decisive visual confirmation."
        image_str = ";".join(supporting_images) if supporting_images else "none"
        if object_part != "unknown":
            return f"The images support the claim by showing {issue_type.replace('_', ' ')} on the {object_part}."
        return f"The images provide evidence consistent with {issue_type.replace('_', ' ')}."
    return "The claim could not be evaluated."

def infer_claim_status(evidence_standard_met: bool, issue_type: str, risk_flags: List[str]) -> str:
    contradiction_flags = {"claim_mismatch", "wrong_object", "wrong_object_part"}

    if not evidence_standard_met:
        # Only mark contradicted if strong mismatch flags exist
        if any(flag in risk_flags for flag in contradiction_flags):
            return "contradicted"
        return "not_enough_information"
    elif issue_type == "none":
        return "contradicted"
    elif any(flag in risk_flags for flag in contradiction_flags):
        return "contradicted"
    else:
        return "supported"



def infer_evidence_standard_met(
    valid_image: bool,
    issue_type: str,
    object_part: str,
    claim_text: str,
    claim_object: str,
    image_list: Sequence[str],
    requirements: Dict[tuple[str, str], int]
) -> tuple[bool, str]:
    if not valid_image:
        return False, "One or more submitted images are missing or unavailable for review."
    if issue_type == "unknown" and object_part == "unknown":
        return False, "The claim does not include a clear issue type or object part for reliable visual evaluation."
    if issue_type == "missing_part" and "package" not in claim_text:
        return False, "The claim suggests a missing item but package content evidence is not clearly available."
    if object_part == "unknown":
        return False, "The relevant object part cannot be inferred from the conversation."

    # Enforce minimum image evidence
    req_key = (claim_object, issue_type)
    if req_key in requirements:
        min_images = requirements[req_key]
        if len(image_list) < min_images:
            return False, f"Insufficient images to meet evidence requirements (need {min_images}, got {len(image_list)})."

    # Do NOT automatically block evidence just because risk flags exist.
    # Evidence is considered met if images are valid and requirements satisfied.
    return True, "The submitted image set appears sufficient to evaluate the claimed damage."



def get_image_ids(image_paths: Sequence[str], claim_status: str) -> str:
    if claim_status != "supported":
        return "none"
    image_ids = [Path(path).stem for path in image_paths if Path(path).stem]
    return ";".join(image_ids) if image_ids else "none"


def load_user_history(history_csv_path: Path) -> Dict[str, Dict[str, str]]:
    history: Dict[str, Dict[str, str]] = {}
    if not history_csv_path.exists():
        return history
    with history_csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history[row["user_id"]] = row
    return history


def load_claims(claims_csv_path: Path) -> List[Dict[str, str]]:
    with claims_csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def load_evidence_requirements(req_csv_path: Path) -> Dict[tuple[str, str], int]:
    requirements: Dict[tuple[str, str], int] = {}
    if not req_csv_path.exists():
        return requirements
    with req_csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["claim_object"].lower(), row["applies_to"].lower())
            try:
                min_images = int(row["minimum_image_evidence"])
                requirements[key] = min_images
            except ValueError:
                # Skip rows where minimum_image_evidence is not a number
                continue
    return requirements

def write_predictions(output_path: Path, predictions: List[ClaimPrediction]) -> None:
    fieldnames = [
        "user_id",
        "image_paths",
        "user_claim",
        "claim_object",
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pred in predictions:
            writer.writerow(pred.__dict__)


def predict_claim(row: Dict[str, str], base_dir: Path, history: Dict[str, Dict[str, str]], requirements: Dict[tuple[str, str], int]) -> ClaimPrediction:
    user_id = row.get("user_id", "")
    image_paths = row.get("image_paths", "")
    user_claim = row.get("user_claim", "")
    claim_object = row.get("claim_object", "unknown").lower()
    image_list = [p.strip() for p in image_paths.split(";") if p.strip()]

    issue_type = parse_issue_type(user_claim)
    object_part = parse_object_part(user_claim, claim_object)
    severity = parse_severity(user_claim, issue_type, claim_object, object_part)
    valid_image, existing_images, missing_images = validate_images(base_dir, image_list)
    evidence_met, evidence_reason = infer_evidence_standard_met(valid_image, issue_type, object_part, user_claim, claim_object, image_list, requirements)
    user_history = history.get(user_id, {})
    history_flags = user_history.get("history_flags", "none")

    history_flags = user_history.get("history_flags", "none")
    risk_flag_list = parse_risk_flags(user_claim, history_flags, issue_type, object_part, valid_image, existing_images)

    # Now add aggressive history risk logic
    past_claims = int(user_history.get("past_claim_count", "0"))
    recent_claims = int(user_history.get("last_90_days_claim_count", "0"))
    if past_claims > 5 or recent_claims > 2:
        if "user_history_risk" not in risk_flag_list:
            risk_flag_list.append("user_history_risk")

    # If we detected strong risk flags, adjust evidence_met conservatively
    risk_set = set(risk_flag_list)
    risky_indicators = {"wrong_object", "claim_mismatch", "manual_review_required", "non_original_image", "damage_not_visible"}
    if evidence_met and (risk_set & risky_indicators):
        evidence_met = False
        evidence_reason = "Image set raised risk flags that prevent reliable automated evaluation."
    # Check image consistency when multiple existing images are available
    if existing_images:
        consistency = images_consistency_check(existing_images)
        if consistency is False:
            if "wrong_object" not in risk_set:
                risk_flag_list.append("wrong_object")
            if "manual_review_required" not in risk_set:
                risk_flag_list.append("manual_review_required")
            evidence_met = False
            evidence_reason = "Submitted images appear inconsistent (different objects shown)."
    claim_status = infer_claim_status(evidence_met, issue_type, risk_flag_list)
    supporting_image_ids = get_image_ids(existing_images if 'existing_images' in locals() else image_list, claim_status)
    justification = build_justification(user_claim, evidence_met, issue_type, object_part, image_list if claim_status == "supported" else [], claim_status)

    if not valid_image and missing_images:
        evidence_reason = f"Missing image files: {', '.join(missing_images)}."
    if claim_status == "not_enough_information" and issue_type != "unknown" and object_part == "unknown":
        evidence_reason = "The claim cannot be evaluated because the relevant object part cannot be inferred from the conversation."

    return ClaimPrediction(
        user_id=user_id,
        image_paths=image_paths,
        user_claim=user_claim,
        claim_object=claim_object,
        evidence_standard_met=str(evidence_met).lower(),
        evidence_standard_met_reason=evidence_reason,
        risk_flags=";".join(risk_flag_list),
        issue_type=issue_type,
        object_part=object_part,
        claim_status=claim_status,
        claim_status_justification=justification,
        supporting_image_ids=supporting_image_ids,
        valid_image=str(valid_image).lower(),
        severity=severity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evidence review prediction pipeline.")
    parser.add_argument("--claims", default="dataset/claims.csv", help="Path to the claims CSV input file.")
    parser.add_argument("--history", default="dataset/user_history.csv", help="Path to the user history CSV file.")
    parser.add_argument("--output", default="output.csv", help="Path to the output CSV file.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    claims_path = Path(args.claims)
    history_path = Path(args.history)
    output_path = Path(args.output)
    req_path = base_dir / "dataset" / "evidence_requirements.csv"
    requirements = load_evidence_requirements(req_path)

    claims = load_claims(claims_path)
    history = load_user_history(history_path)
    predictions = [predict_claim(row, base_dir, history, requirements) for row in claims]
    write_predictions(output_path, predictions)
    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
