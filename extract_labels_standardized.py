import re

LABEL_MAP = {
    "O": 0,  # Space only
    ".": 1,
    ",": 2,
    "!": 3,
    "?": 4,
    ":": 5,
    ";": 6,
    "-": 7,

    '"': 8,

    ', "': 9, # start of direct speech
    ': "': 10,
    '. "': 11,
    '"?': 12, # not citation but just quotes
    '"!': 13,

    '",': 14, # common direct speech endings
    '".': 15,
    '?"': 16,
    '!"': 17,

    '" -': 18, # dash in quotations
    '- "': 19,
    '", -': 20,
    '!" -': 21,
    '?" -': 22,
    '. -': 23,

    '""': 24,  # Double close (поступил в учреждение "школа "Эврика"".)

    "! -": 25,  # direct speech without closing "
    "? -": 26,
    ", -": 27,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def robust_clean_text(text):
    # Strip invisible / zero-width characters that corrupt gap skeletons
    text = re.sub(r'[\u200b-\u200f\u2028\u2029\u202a-\u202e\ufeff\u00ad]', '', text)
    # Unicode dashes + minus sign \u2192 ASCII hyphen
    text = re.sub(r'[\u2010-\u2015\u2212]', '-', text)
    # Multi-hyphen (common em-dash substitute in web fiction) \u2192 single
    text = re.sub(r'-{2,}', '-', text)
    # All typographic quote variants \u2192 ASCII "
    text = re.sub(r'[«»\u201c\u201d\u201e\u201f]', '"', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_canonical_label(gap_text):
    if not gap_text.strip():
        return "O"

    skeleton = re.sub(r'\s+', '', gap_text)

    if skeleton == '.': return '.'
    if skeleton == ',': return ','
    if skeleton == '!': return '!'
    if skeleton == '?': return '?'
    if skeleton == ':': return ':'
    if skeleton == ';': return ';'
    if skeleton == '-': return '-'

    if skeleton == '"': return '"'

    if skeleton == ',"': return ', "'
    if skeleton == ':"': return ': "'
    if skeleton == '."': return '. "'

    if skeleton == '",': return '",'
    if skeleton == '".': return '".'
    if skeleton == '"?': return '"?'
    if skeleton == '"!': return '"!'
    if skeleton == '?"': return '?"'
    if skeleton == '!"': return '!"'

    if skeleton == '"-': return '" -'
    if skeleton == '-"': return '- "'
    if skeleton == '",-': return '", -'
    if skeleton == '!"-': return '!" -'
    if skeleton == '?"-': return '?" -'
    if skeleton == ',"-': return '", -'

    if skeleton == '!-': return '! -'
    if skeleton == '?-': return '? -'
    if skeleton == '.-': return '. -'
    if skeleton == ',-': return ', -'

    if skeleton == '""': return '""'

    # ?! and !? are common in fiction; reduce to the primary mark and retry
    if '?' in skeleton and '!' in skeleton:
        primary = '?' if skeleton.index('?') < skeleton.index('!') else '!'
        reduced = skeleton.replace('!' if primary == '?' else '?', '')
        return _lookup_skeleton(reduced)

    return "O"


def _lookup_skeleton(skeleton):
    """Fast second-pass lookup used by the ?!/!? fallback."""
    _MAP = {
        '?': '?', '!': '!',
        '"?': '"?', '"!': '"!',
        '?"': '?"', '!"': '!"',
        '?"-': '?" -', '!"-': '!" -',
        '?-': '? -', '!-': '! -',
    }
    return _MAP.get(skeleton, 'O')


def extract_labels_standardized(text):
    text = robust_clean_text(text)

    # number:
    # -?              : Optional negative
    # \d+             : digit start
    # (?:[.,]\d+)+    : followed by a group of (dot OR comma) + digits, repeated 1+ times
    # punct doesn't divide parts of number
    number_pattern = r"-?\d+(?:[.,]\d+)+"

    # word:
    # -?           -> optional minus (in numbers like -1.5)
    # [(\[]?       -> Optional ( or [
    # [\w]+        -> word characters
    # (?:-[\w]+)*  -> Optional hyphen
    # [)\]]?       -> Optional closing ) or ]
    word_pattern = r"-?[(\[]?[\w]+(?:-[\w]+)*[)\]]?"

    # standalone brackets
    brackets_pattern = r"[()]|[\[\]]"

    full_pattern = f"{number_pattern}|{word_pattern}|{brackets_pattern}"

    word_iter = list(re.finditer(full_pattern, text))

    result_pairs = []

    for i, match in enumerate(word_iter):
        word = match.group()
        start_gap = match.end()

        if i + 1 < len(word_iter):
            end_gap = word_iter[i + 1].start()
            raw_gap = text[start_gap:end_gap]
        else:
            raw_gap = text[start_gap:]

        label_str = get_canonical_label(raw_gap)
        label_id = LABEL_MAP.get(label_str, 0)

        result_pairs.append({"word": word, "label": label_id})

    return result_pairs
