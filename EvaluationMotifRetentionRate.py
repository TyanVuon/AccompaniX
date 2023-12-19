import music21
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict, Counter
def load_musicxml(file_path):
    return music21.converter.parse(file_path)

def extract_pitches(score, part_index):
    part = score.parts[part_index]
    return [note.pitch for note in part.recurse().notes if note.isNote]

def compute_pitch_diffs(pitches):
    return [pitches[i + 1].midi - pitches[i].midi for i in range(len(pitches) - 1)]

def has_matching_signs(*seqs):
    for seq1, seq2 in itertools.combinations(seqs, 2):
        if not all((x > 0 and y > 0) or (x < 0 and y < 0) for x, y in zip(seq1, seq2)):
            return False
    return True

def within_tolerance(seq1, seq2, tolerance_levels):
    for tolerance, max_exceptions in tolerance_levels:
        exceptions = 0
        for x, y in zip(seq1, seq2):
            ratio = y / x if x != 0 else 0
            if not (1 / tolerance <= abs(ratio) <= tolerance):
                exceptions += 1
                if exceptions > max_exceptions:
                    break
        else:
            return tolerance
    return None


def analyze_and_merge_themes(themes):
    # Group themes by pitch differences
    grouped_themes = defaultdict(list)
    for theme, info in themes.items():
        grouped_themes[theme].extend(info['indices'])

    # Analyze the density of indices for each theme group
    merged_themes = {}
    for theme, indices in grouped_themes.items():
        count_indices = Counter(indices)
        most_common_indices = count_indices.most_common()  # List of (index, count) tuples
        density_info = {'indices': most_common_indices, 'total_count': len(indices)}
        merged_themes[theme] = density_info

    return merged_themes
def detect_recurring_themes(tracks_with_pitches, voice_names, window_size=8, tolerance_levels=[(1, 3), (2, 3), (3, 0)]):
    themes = defaultdict(lambda: {'count': 0, 'indices': [], 'match_levels': [], 'first_appearance': None})
    for track_index, (pitches, pitch_diffs) in enumerate(tracks_with_pitches):
        for i in range(len(pitch_diffs) - window_size + 1):
            window = tuple(pitch_diffs[i:i + window_size])
            for other_track_index, (_, other_pitch_diffs) in enumerate(tracks_with_pitches):
                if track_index != other_track_index:
                    other_window = tuple(other_pitch_diffs[i:i + window_size])
                    match_level = within_tolerance(window, other_window, tolerance_levels)
                    if match_level is not None:
                        themes[window]['count'] += 1
                        themes[window]['indices'].append(i)
                        themes[window]['match_levels'].append(match_level)
                        if not themes[window]['first_appearance']:
                            themes[window]['first_appearance'] = (track_index, i)
    return themes

def analyze_index_density(themes):
    index_counter = Counter()
    for theme, info in themes.items():
        index_counter.update(info['indices'])
    return index_counter.most_common()

def detect_recurring_themes(tracks_with_pitches, voice_names, window_size=8, tolerance_levels=[(1, 3), (2, 3), (3, 0)]):
    themes = defaultdict(lambda: {'count': 0, 'indices': [], 'match_levels': [], 'first_appearance': None})
    for track_index, (pitches, pitch_diffs) in enumerate(tracks_with_pitches):
        for i in range(len(pitch_diffs) - window_size + 1):
            window = tuple(pitch_diffs[i:i + window_size])
            for other_track_index, (_, other_pitch_diffs) in enumerate(tracks_with_pitches):
                if track_index != other_track_index:
                    other_window = tuple(other_pitch_diffs[i:i + window_size])
                    match_level = within_tolerance(window, other_window, tolerance_levels)
                    if match_level is not None:
                        themes[window]['count'] += 1
                        themes[window]['indices'].append(i)
                        themes[window]['match_levels'].append(match_level)
                        if not themes[window]['first_appearance']:
                            themes[window]['first_appearance'] = (track_index, i)

    # Group themes by pitch differences for merging
    pitch_diff_groups = defaultdict(list)
    for theme, info in themes.items():
        pitch_diff_groups[theme].append(info)

    # Print themes with pitch differences and pitch names
    for pitch_diff, theme_infos in pitch_diff_groups.items():
        print(f"Pitch Difference: {pitch_diff}")
        for info in theme_infos:
            track_index, first_index = info['first_appearance']
            track_name = voice_names[track_index] if track_index < len(voice_names) else f"Cross-Voice Range {track_index - len(voice_names) + 1}"
            first_pitch_sequence = revert_to_notes(tracks_with_pitches[track_index][0], pitch_diff, first_index)
            print(f"  Theme: {first_pitch_sequence}, Track: {track_name}, Initial Index: {first_index}, Count: {info['count']}")
            sub_group_counts = defaultdict(int)
            for level in info['match_levels']:
                sub_group_counts[level] += 1
            for sub_ratio, sub_count in sub_group_counts.items():
                print(f"    Sub-Ratio {sub_ratio}: {sub_count} matches")
            # Density analysis
            index_counter = Counter(info['indices'])
            most_common_index, density = index_counter.most_common(1)[0]
            print(f"    Most Common Index: {most_common_index}, Density: {density}")
    return themes


def merge_themes_by_density(themes, top_indices):
    merged_themes = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'themes': []}))

    for theme, info in themes.items():
        for index in top_indices:
            if index in info['indices']:
                for i, idx in enumerate(info['indices']):
                    if idx == index:
                        merged_themes[index][theme]['count'] += 1
                        if info['first_appearance'][1] == idx:
                            merged_themes[index][theme]['themes'].append(info['first_appearance'])

    return merged_themes


def revert_to_notes(pitches, pitch_diffs, index):
    sequence = [pitches[index]]
    for diff in pitch_diffs[:7]:
        next_pitch = music21.pitch.Pitch()
        next_pitch.midi = sequence[-1].midi + diff
        sequence.append(next_pitch)
    return ' '.join(p.nameWithOctave for p in sequence)
def main(file_path):
    score = load_musicxml(file_path)
    voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
    cross_voice_ranges = [('Soprano-Alto Cross', (65, 70.5)), ('Alto-Tenor Cross', (58.5, 65)), ('Tenor-Bass Cross', (50.5, 58.5))]

    all_tracks_with_pitches = []
    for index, voice_name in enumerate(voice_names):
        pitches = extract_pitches(score, index)
        all_tracks_with_pitches.append((pitches, compute_pitch_diffs(pitches)))

    for cross_voice_name, (lower, upper) in cross_voice_ranges:
        cross_voice_pitches = []
        for part in score.parts:
            cross_voice_pitches.extend([note.pitch for note in part.recurse().notes if lower <= note.pitch.midi <= upper])
        all_tracks_with_pitches.append((cross_voice_pitches, compute_pitch_diffs(cross_voice_pitches)))
        voice_names.append(cross_voice_name)  # Append the name for the cross-voice range

    themes = detect_recurring_themes(all_tracks_with_pitches, voice_names)
    top_indices = [index for index, _ in analyze_index_density(themes)[:10]]
    merged_themes = merge_themes_by_density(themes, top_indices)

    # Printing merged theme information
    for index, pitch_diffs_info in merged_themes.items():
        print(f"Most Common Index: {index}")
        for pitch_diff, info in pitch_diffs_info.items():
            print(f"  Pitch Difference: {pitch_diff}, Count: {info['count']}")
            for track_index, first_index in info['themes']:
                track_name = voice_names[track_index] if track_index < len(voice_names) else f"Cross-Voice Range {track_index - len(voice_names) + 1}"
                theme_sequence = revert_to_notes(all_tracks_with_pitches[track_index][0], pitch_diff, first_index)
                print(f"    Theme: {theme_sequence}, Track: {track_name}, Initial Index: {first_index}")
        print()

if __name__ == "__main__":
    file_path = "C:\\Users\\Tyan\\AppData\\Local\\Temp\\music21\\reindexedDefaulttrain512lgnth.musicxml"
    main(file_path)
