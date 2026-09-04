"""Consolidated emotion and feeling_it annotations from Luana's scripts.

Source: luana-Crocodile-with-data/script_[1-4]S.py, script_1X.py
Each session's annotations are stored as range tuples (start, end) where
end is exclusive (matching Python's range() convention).
"""

# Emotion indices per session, stored as {label: [(start, end), ...]}
SESSION_EMOTIONS = {
    '1S': {
        'nul': [(54, 291), (372, 416), (551, 605), (1879, 1897), (5282, 5303),
                (6356, 6750), (6987, 7006), (9304, 9316), (10425, 10450),
                (10863, 10876), (12114, 12142), (12676, 12691), (18009, 19048),
                (22491, 22513), (22635, 22717), (22937, 23147), (24921, 24944)],
        'war': [(0, 54), (291, 372), (416, 551), (605, 1300)],
        'tra': [(8617, 8982), (13507, 14188), (17568, 18009), (19048, 20461),
                (22717, 22880)],
        'rlx': [(1300, 1879), (1897, 4958)],
        'con': [(4958, 5282), (5303, 6356), (6750, 6988), (7006, 8617)],
        'cfd': [(8982, 9304), (9316, 10425), (10450, 10863), (10876, 12114),
                (12253, 12676), (12691, 13507)],
        'gra': [(14188, 17568)],
        'joy': [(20461, 22491), (22513, 22635), (22880, 22937), (23147, 24921),
                (24944, 25112)],
    },
    '2S': {
        'nul': [(0, 325), (481, 527), (760, 1037), (14451, 16812), (23429, 23647)],
        'tra': [(325, 481), (527, 760), (4020, 4199), (5622, 5745),
                (19534, 20037), (23058, 23429)],
        'rlx': [(1037, 2354)],
        'sup': [(2354, 4020)],
        'sun': [(4199, 5622)],
        'dst': [(5745, 8473)],
        'anx': [(8473, 11578)],
        'fea': [(11578, 14451)],
        'dsg': [(16812, 19534)],
        'ang': [(20037, 23058)],
    },
    '3S': {
        'nul': [(68, 335), (455, 510), (13458, 28038), (30262, 30747),
                (31534, 31778), (36320, 36671)],
        'tra': [(0, 68), (335, 455)],
        'war': [(510, 1195)],
        'sha': [(1195, 3883)],
        'pai': [(3883, 5950)],
        'dsp': [(5950, 9104)],
        'sad': [(9104, 13458)],
        'tir': [(28038, 30262), (30747, 31534), (31778, 32174)],
        'rlf': [(32174, 36320)],
    },
    '4S': {
        'nul': [(0, 331), (587, 639), (9854, 10431), (16025, 16395),
                (35593, 35812), (36280, 36481), (39036, 40563), (43112, 43612)],
        'tra': [(331, 587), (13231, 13408), (17268, 17491), (29233, 29449),
                (32509, 33321), (35812, 36062), (40563, 40714)],
        'war': [(639, 1218)],
        'aro': [(1218, 4148)],
        'neu': [(4148, 5350), (9327, 9854), (10431, 11355), (15430, 15778),
                (15821, 16025), (16395, 17268), (22353, 23890), (29449, 30292),
                (30409, 32509)],
        'joy': [(5350, 9327), (15778, 15821), (30292, 30409)],
        'sup': [(11355, 12651), (12848, 13231)],
        'sun': [(12651, 12848), (13408, 15430)],
        'ang': [(17491, 22353)],
        'fea': [(23890, 29233)],
        'gri': [(33321, 35593)],
        'sad': [(36062, 36280), (36481, 39036)],
        'tir': [(40714, 43112)],
    },
    '1X': {
        'nul': [(0, 312), (2457, 2834), (12119, 12241), (12582, 12749),
                (14895, 17579), (21147, 21340), (25913, 26011), (27881, 27916),
                (28583, 29147), (29251, 29284), (33400, 33931), (38147, 39649),
                (41028, 41540)],
        'tra': [(312, 622), (26509, 26699), (26895, 27000), (35041, 35400)],
        'rlx': [(622, 2457), (2834, 3453)],
        'con': [(3453, 4481)],
        'cfd': [(4481, 5965)],
        'pri': [(5965, 7231)],
        'gra': [(7231, 9003)],
        'joy': [(9003, 10417)],
        'aro': [(10417, 12000)],
        'sup': [(12000, 12119), (12241, 12582), (12749, 13374)],
        'sun': [(13374, 13761)],
        'dst': [(13761, 14895), (17579, 18377)],
        'anx': [(18377, 19792)],
        'fea': [(19792, 21147), (21340, 21818)],
        'dsg': [(21818, 23401)],
        'ang': [(23401, 25913), (26011, 26509), (26699, 26895)],
        'sha': [(27000, 27881), (27916, 28583)],
        'gri': [(29147, 29251), (29284, 31603)],
        'lau': [(31603, 33400)],
        'pai': [(33931, 35041)],
        'dsp': [(35400, 36785)],
        'sad': [(36785, 38147)],
        'tir': [(39649, 41028)],
    },
}

# Feeling_it indices per session, stored as {0_or_1: [(start, end), ...]}
SESSION_FEELING_IT = {
    '1S': {
        0: [(0, 1897), (4958, 5303), (6356, 7006), (8617, 9315),
            (10425, 10876), (12114, 12691), (13507, 14188), (17568, 20846),
            (22491, 23350), (24921, 25112)],
        1: [(1897, 4958), (5303, 6356), (7006, 8617), (9315, 10425),
            (10876, 12114), (12691, 13507), (14188, 17568), (20846, 22491),
            (23348, 24921)],
    },
    '2S': {
        0: [(0, 2354), (2386, 5174), (5390, 5987), (8473, 9009),
            (11061, 12603), (14275, 17029), (19317, 20576), (21187, 22006),
            (23058, 23647)],
        1: [(2354, 2386), (5174, 5390), (5987, 8473), (9009, 11061),
            (12603, 14275), (17029, 19317), (20576, 21187), (22006, 23058)],
    },
    '3S': {
        0: [(0, 1428), (2740, 2984), (3883, 6103), (7554, 7980),
            (8817, 9175), (10265, 10712), (11701, 12512), (13458, 28131),
            (30262, 32459), (36320, 36671)],
        1: [(1428, 2740), (2984, 3883), (6103, 7554), (7980, 8817),
            (9175, 10265), (10712, 11701), (12512, 13458), (28131, 30262),
            (32459, 36320)],
    },
    '4S': {
        0: [(0, 1407), (3885, 4284), (5350, 5797), (7187, 8007),
            (8777, 10929), (11355, 13616), (14792, 16563), (17268, 17594),
            (19165, 19514), (21306, 22929), (23832, 24443), (25585, 26461),
            (29228, 30498), (31883, 36100), (38267, 40834), (43112, 43612)],
        1: [(1407, 3885), (4284, 5350), (5797, 7187), (8007, 8777),
            (10929, 11355), (13616, 14792), (16563, 17268), (17594, 19165),
            (19514, 21306), (22929, 23832), (24443, 25585), (26461, 29228),
            (30498, 31883), (36100, 38267), (40834, 43112)],
    },
    '1X': {
        0: [(0, 41540)],
    },
}

# Recording start times (seconds from video start to biodata start)
# Source: luana-Crocodile-with-data/script_[1-4]S_biodata.py
RECORDING_START_TIMES = {
    '1S': 19.332,
    '2S': 17.582,
    '3S': 17.127,
    '4S': 22.201,
}

# Mapping from session to continuous_features.csv session_id
SESSION_TO_FEATURES_ID = {
    '1S': 'emotion_biodata_laurence_main_1',
    '2S': 'emotion_biodata_laurence_main_2',
    '3S': 'emotion_biodata_laurence_main_3',
    '4S': 'emotion_biodata_laurence_main_4',
}

# Labels that are NOT real emotions (used for filtering)
NON_EMOTION_LABELS = {'nul', 'war', 'tra', 'coo', 'neu'}

# Invalid label — frames with makeup, clapperboard, tech issues etc.
INVALID_LABEL = 'nul'


def _find_in_ranges(ranges, index):
    """Binary-ish search through sorted range tuples."""
    for start, end in ranges:
        if start <= index < end:
            return True
        if index < start:
            return False
    return False


def get_emotion(session, frame_number):
    """Get emotion label for a given session and frame number.

    Returns the emotion abbreviation string, or None if the frame is
    not covered by any annotation range.
    """
    emotions = SESSION_EMOTIONS.get(session)
    if emotions is None:
        raise ValueError(f"Unknown session: {session}")
    for label, ranges in emotions.items():
        if _find_in_ranges(ranges, frame_number):
            return label
    return None


def get_feeling_it(session, frame_number):
    """Get feeling_it value for a given session and frame number.

    Returns True if feeling_it=1, False if feeling_it=0,
    None if frame is not covered.
    """
    feeling_it = SESSION_FEELING_IT.get(session)
    if feeling_it is None:
        raise ValueError(f"Unknown session: {session}")
    if 1 in feeling_it and _find_in_ranges(feeling_it[1], frame_number):
        return True
    if 0 in feeling_it and _find_in_ranges(feeling_it[0], frame_number):
        return False
    return None


def is_invalid(session, frame_number):
    """Check if a frame is invalid (nul) and should be excluded from training.

    Invalid frames contain makeup application, clapperboard, tech issues, etc.
    """
    emotion = get_emotion(session, frame_number)
    return emotion == INVALID_LABEL


def get_emotion_label(session, frame_number):
    """Get emotion label suitable for training.

    Returns the emotion abbreviation, or 'none' for non-emotion labels
    (nul, war, tra, coo, neu). Returns 'none' if frame has no annotation.
    """
    emotion = get_emotion(session, frame_number)
    if emotion is None or emotion in NON_EMOTION_LABELS:
        return 'none'
    return emotion
