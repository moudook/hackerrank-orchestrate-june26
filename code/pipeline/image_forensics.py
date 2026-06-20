import logging
import os
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not available — EXIF extraction disabled")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("pytesseract not available — OCR scanning disabled")


INJECTION_KEYWORDS = [
    r'ignore.{0,30}(instructions|previous|prior|all)',
    r'approv(e|ed|ing|e this|e the).{0,20}(claim|payment|payout)',
    r'override.{0,20}(system|instructions|policy)',
    r'forget.{0,20}(all|previous|prior).{0,20}(instructions|rules)',
    r'(set|mark|change).{0,20}(severity|status|verdict|decision)',
    r'(approve|mark).{0,20}(this|the).{0,20}(row|claim)',
    r'system.{0,10}prompt',
    r'you.{0,10}are.{0,10}(an|a).{0,10}(ai|model|assistant)',
    r'(skip|bypass).{0,20}(manual|review|verification)',
    r'(reject|deny).{0,20}(claim|this)',
    r'escalate.{0,10}(publicly|immediately)',
    r'reopen.{0,10}tickets',
    r'follow.{0,10}(it|this).{0,10}and.{0,10}approve',
    r'accept.{0,10}(this|it).{0,10}quickly',
    r'pay.{0,10}(the|full|out).{0,10}(claim|amount|payout)',
    r'(total|full).{0,10}(loss|payout|claim)',
    r'classif(y|ied).{0,10}(this|it).{0,10}as.{0,10}(total|total loss)',
    r'exaggerat(e|ed|ing).{0,10}(damage|severity)',
    r'(fabricat|invent).{0,10}(evidence|damage)',
    r'(fake|forged|counterfeit).{0,10}(document|photo|image)',
]

INSTRUCTION_LAYOUT_PATTERNS = [
    r'^\s*\d+[\.\)]\s',  # Numbered lists: "1. ", "2) "
    r'^\s*[-•]\s',         # Bullet points
    r'(step|phase|stage)\s*\d+',  # Step instructions
]

UI_ELEMENT_PATTERNS = [
    r'progress.{0,10}(bar|indicator|tracker)',
    r'(score|scoring).{0,10}(rubric|criteria|table|dashboard)',
    r'(pass|fail).{0,10}(rate|threshold)',
    r'(model|quality).{0,10}control',
    r'(claim|review).{0,10}(id|number|dashboard)',
    r'(manual|auto).{0,10}(review|approval)',
    r'(category|priority|status).{0,10}(high|medium|low|urgent)',
]


def scan_text_in_image(image_path: str) -> Dict:
    result = {
        'ocr_text': '',
        'injection_detected': False,
        'injection_matches': [],
        'instruction_layout_detected': False,
        'ui_elements_detected': False,
        'ui_element_matches': [],
        'ocr_available': TESSERACT_AVAILABLE,
    }

    if not TESSERACT_AVAILABLE:
        return result

    try:
        ocr_text = pytesseract.image_to_string(Image.open(image_path))
        result['ocr_text'] = ocr_text

        text_lower = ocr_text.lower()

        for pattern in INJECTION_KEYWORDS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            if matches:
                result['injection_detected'] = True
                result['injection_matches'].append({
                    'pattern': pattern,
                    'matches': matches[:3],
                })

        for pattern in INSTRUCTION_LAYOUT_PATTERNS:
            if re.search(pattern, ocr_text, re.MULTILINE):
                result['instruction_layout_detected'] = True
                break

        for pattern in UI_ELEMENT_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                result['ui_elements_detected'] = True
                result['ui_element_matches'].extend(matches[:3])

    except Exception as e:
        logger.warning(f"OCR scan failed for {image_path}: {e}")

    return result


def extract_exif_metadata(image_path: str) -> Dict:
    result = {
        'has_exif': False,
        'metadata': {},
        'anomalies': [],
        'is_screenshot': False,
        'is_ai_generated_hint': False,
        'metadata_stripped': False,
    }

    if not PIL_AVAILABLE:
        return result

    try:
        with Image.open(image_path) as img:
            exif_data = img.getexif()

            if not exif_data:
                result['metadata_stripped'] = True
                result['anomalies'].append('no_exif_data')
                return result

            result['has_exif'] = True

            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if isinstance(value, bytes):
                    continue
                result['metadata'][tag_name] = str(value)

            software = result['metadata'].get('Software', '').lower()
            if any(s in software for s in ['photoshop', 'gimp', 'snapseed', 'lightroom', 'editor']):
                result['anomalies'].append(f'editing_software:{software}')

            make = result['metadata'].get('Make', '').lower()
            model = result['metadata'].get('Model', '').lower()
            if not make and not model:
                result['anomalies'].append('missing_camera_info')

            datetime_original = result['metadata'].get('DateTimeOriginal', '')
            datetime_modified = result['metadata'].get('DateTime', '')
            if datetime_original and datetime_modified:
                if datetime_modified < datetime_original:
                    result['anomalies'].append('modification_before_capture')

            if 'MacOS' in str(result['metadata'].get('Software', '')):
                result['is_screenshot'] = True
                result['anomalies'].append('possible_screenshot')

            width = int(result['metadata'].get('ImageWidth', 0))
            height = int(result['metadata'].get('ImageLength', 0))
            if width > 0 and height > 0:
                aspect = width / height
                if abs(aspect - 9/16) < 0.05 or abs(aspect - 9/19.5) < 0.05:
                    result['is_screenshot'] = True
                    result['anomalies'].append('phone_aspect_ratio')

    except Exception as e:
        logger.warning(f"EXIF extraction failed for {image_path}: {e}")
        result['metadata_stripped'] = True
        result['anomalies'].append('exif_extraction_error')

    return result


def scan_image_forensics(image_path: str) -> Dict:
    ocr_result = scan_text_in_image(image_path)
    exif_result = extract_exif_metadata(image_path)

    combined_anomalies = []
    risk_flags = []

    if ocr_result['injection_detected']:
        combined_anomalies.append('text_injection_detected')
        risk_flags.append('possible_manipulation')

    if ocr_result['instruction_layout_detected']:
        combined_anomalies.append('instruction_layout')
        risk_flags.append('possible_manipulation')

    if ocr_result['ui_elements_detected']:
        combined_anomalies.append('ui_elements_in_image')
        risk_flags.append('possible_manipulation')

    if exif_result['is_screenshot']:
        combined_anomalies.append('screenshot_detected')
        risk_flags.append('non_original_image')

    if exif_result['metadata_stripped']:
        combined_anomalies.append('metadata_stripped')

    if exif_result['anomalies']:
        combined_anomalies.extend(exif_result['anomalies'])

    if any(a in combined_anomalies for a in ['editing_software:photoshop', 'editing_software:gimp']):
        risk_flags.append('possible_manipulation')

    risk_flags = list(dict.fromkeys(risk_flags))

    return {
        'ocr': ocr_result,
        'exif': exif_result,
        'anomalies': combined_anomalies,
        'risk_flags': risk_flags,
        'has_suspicious_content': bool(risk_flags),
    }


def scan_images_batch(image_paths: List[str]) -> Dict:
    all_risk_flags = []
    all_anomalies = []
    any_suspicious = False
    ocr_texts = []

    for path in image_paths:
        if not os.path.exists(path):
            continue
        result = scan_image_forensics(path)
        all_risk_flags.extend(result['risk_flags'])
        all_anomalies.extend(result['anomalies'])
        if result['has_suspicious_content']:
            any_suspicious = True
        if result['ocr']['ocr_text']:
            ocr_texts.append(result['ocr']['ocr_text'])

    all_risk_flags = list(dict.fromkeys(all_risk_flags))
    all_anomalies = list(dict.fromkeys(all_anomalies))

    combined_ocr = '\n---\n'.join(ocr_texts) if ocr_texts else ''

    return {
        'risk_flags': all_risk_flags,
        'anomalies': all_anomalies,
        'has_suspicious_content': any_suspicious,
        'combined_ocr_text': combined_ocr,
        'images_scanned': len(image_paths),
    }
