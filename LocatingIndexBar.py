from music21 import converter, stream

# Load the MusicXML file into a music21 stream
score = converter.parse(r"C:\\Users\\Tyan\\AppData\\Local\\Temp\\music21\\reindexedDefaulttrain512lgnth.musicxml")

# Initialize counters for each voice
soprano_counter, alto_counter, tenor_counter, bass_counter = (0, 0, 0, 0)

# Function to find the bar number for a given index
def find_bar_number(voice, index):
    counter = 0
    for note in voice.recurse().notes:
        if counter == index:
            return note.measureNumber
        counter += 1
    return None

# Assuming 'score' is ordered Soprano, Alto, Tenor, Bass
soprano_bar_89 = find_bar_number(score.parts[0], 89)
alto_bar_89 = find_bar_number(score.parts[1], 89)
tenor_bar_89 = find_bar_number(score.parts[2], 89)
bass_bar_89 = find_bar_number(score.parts[3], 89)

soprano_bar_111 = find_bar_number(score.parts[0], 111)
alto_bar_111 = find_bar_number(score.parts[1], 111)
tenor_bar_111 = find_bar_number(score.parts[2], 111)
bass_bar_111 = find_bar_number(score.parts[3], 111)

# Print the bar numbers
print(f"Soprano Bar at Index 89: {soprano_bar_89}")
print(f"Alto Bar at Index 89: {alto_bar_89}")
print(f"Tenor Bar at Index 89: {tenor_bar_89}")
print(f"Bass Bar at Index 89: {bass_bar_89}")

print(f"Soprano Bar at Index 111: {soprano_bar_111}")
print(f"Alto Bar at Index 111: {alto_bar_111}")
print(f"Tenor Bar at Index 111: {tenor_bar_111}")
print(f"Bass Bar at Index 111: {bass_bar_111}")
