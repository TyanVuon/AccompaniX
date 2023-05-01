from music21 import converter, stream


def get_melody(score):
    # Extract the melody from each part
    melody_parts = []
    for part in score.parts:
        melody_stream = stream.Stream([part])
        melody = list(melody_stream.flatten().notes)
        int_seq = [0] * (len(melody) - 1)
        for i in range(1, len(melody)):
            int_seq[i-1] = melody[i].pitch.midi - melody[i-1].pitch.midi
        melody_parts.append(int_seq)
    return melody_parts


def main():
    # Load a MusicXML file
    score = converter.parse(r'C:\Users\Tyan\DPBCT\DeepBachTyan\melody_extraction\sophisticationn.mxml', format='musicxml')

    # Extract the melody from each part
    melody_parts = get_melody(score)

    # Print each pitch difference sequence for all parts
    for i, part_seq in enumerate(melody_parts):
        print(f"Part {i+1}: {part_seq}")


if __name__ == '__main__':
    main()
