from music21 import converter, stream


def get_melody(score):
    # Assume that the melody is the highest-pitched part
    melody_part = max(score.parts, key=lambda p: p.highestTime)
    melody_stream = stream.Stream([melody_part])
    melody = melody_stream.flat.notes
    int_seq = [0] * (len(melody) - 1)
    for i in range(1, len(melody)):
        int_seq[i-1] = melody[i].pitch.midi - melody[i-1].pitch.midi

    print(int_seq)
    return int_seq



def main():
    # Load a MusicXML file
    score = converter.parse(r'C:\Users\Tyan\DPBCT\DeepBachTyan\melody_extraction\sophisticationn.mxml', format='musicxml')


    # Extract the melody from the score
    int_seq = get_melody(score)

    # Use the int_seq list in further processing steps
    # ...


if __name__ == '__main__':
    main()



