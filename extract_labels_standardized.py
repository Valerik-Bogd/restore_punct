import re

LABEL_MAP = {
    "O": 0, # Space only
    ".": 1,
    ",": 2,
    "!": 3,
    "?": 4,
    ":": 5,
    ";": 6,
    "-": 7,

    '"': 8,
    ' "': 9,
    '" ': 10,

    ', "': 11,
    ': "': 12,
    '. "': 13,

    '",': 14,
    '".': 15,
    '"?': 16,
    '"!': 17,
    '...': 18,

    '- "': 19,
    '", -': 20, # direct speech with closing "
    '!" -': 21,
    '?" -': 22,
    '. -': 23,

    '""': 24,  # Double close (поступил в учреждение "школа "Эврика"".)

    "! -": 25, # direct speech without closing "
    "? -": 26,
    ", -": 27,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

def robust_clean_text(text):
    text = re.sub(r'[\u2010-\u2015]', '-', text)
    text = re.sub(r'[«»“”„]', '"', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_canonical_label(gap_text):
    if not gap_text.strip():
        return "O"

    # capture spacing before stripping to distinguishes ' "' from '" '
    has_leading_space = gap_text.startswith(' ')
    # assume standard punctuation implies trailing space for simple marks
    # but for quotes it's different
    skeleton = re.sub(r'\s+', '', gap_text)

    if skeleton == '.': return '.'
    if skeleton == ',': return ','
    if skeleton == '!': return '!'
    if skeleton == '?': return '?'
    if skeleton == ':': return ':'
    if skeleton == ';': return ';'
    if skeleton == '-': return '-'

    if skeleton == '"':
        # If leading space => Open Quote
        if has_leading_space:
            return ' "'
        # otherwise assume Close Quote
        return '" '

    if skeleton == ',"': return ', "'
    if skeleton == ':"': return ': "'
    if skeleton == '."': return '. "'

    if skeleton == '",': return '",'
    if skeleton == '".': return '".'
    if skeleton == '"?': return '"?'
    if skeleton == '"!': return '"!'

    if skeleton == '-"': return '- "'
    if skeleton == '",-': return '", -'
    if skeleton == '!"-': return '!"-'
    if skeleton == '?"-': return '?"-'

    if skeleton == '!-': return '! -'
    if skeleton == '?-': return '? -'
    if skeleton == '.-': return '. -'
    if skeleton == ',-': return ', -'

    if skeleton == '""': return '""' # Fallback
    if skeleton == '"",': return '"",'

    return "O"


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
            end_gap = word_iter[i+1].start()
            raw_gap = text[start_gap:end_gap]
        else:
            raw_gap = text[start_gap:]

        label_str = get_canonical_label(raw_gap)
        label_id = LABEL_MAP.get(label_str, 0)

        result_pairs.append({"word": word, "label": label_id})

    return result_pairs