def preprocess_input(self, tensor_chorale, tensor_metadata):
    """
    :param tensor_chorale: (batch_size, num_voices, chorale_length_ticks)
    :param tensor_metadata: (batch_size, num_metadata, chorale_length_ticks)
    :return: (notes, metas, label) tuple
    where
    notes = (left_notes, central_notes, right_notes)
    metas = (left_metas, central_metas, right_metas)
    label = (batch_size)
    right_notes and right_metas are REVERSED (from right to left)
    """
    batch_size, num_voices, chorale_length_ticks = tensor_chorale.size()

    # random shift! Depends on the dataset
    offset = random.randint(0, self.dataset.subdivision)
    time_index_ticks = chorale_length_ticks // 2 + offset

    # split notes
    notes, label = self.preprocess_notes(tensor_chorale, time_index_ticks)
    metas = self.preprocess_metas(tensor_metadata, time_index_ticks)
    return notes, metas, label


def preprocess_notes(self, tensor_chorale, time_index_ticks):
    """

    :param tensor_chorale: (batch_size, num_voices, chorale_length_ticks)
    :param time_index_ticks:
    :return:
    """
    batch_size, num_voices, _ = tensor_chorale.size()
    left_notes = tensor_chorale[:, :, :time_index_ticks]
    right_notes = reverse_tensor(
        tensor_chorale[:, :, time_index_ticks + 1:],
        dim=2)
    if self.num_voices == 1:
        central_notes = None
    else:
        central_notes = mask_entry(tensor_chorale[:, :, time_index_ticks],
                                   entry_index=self.main_voice_index,
                                   dim=1)
    label = tensor_chorale[:, self.main_voice_index, time_index_ticks]
    return (left_notes, central_notes, right_notes), label


def preprocess_metas(self, tensor_metadata, time_index_ticks):
    """

    :param tensor_metadata: (batch_size, num_voices, chorale_length_ticks)
    :param time_index_ticks:
    :return:
    """

    left_metas = tensor_metadata[:, self.main_voice_index, :time_index_ticks, :]
    right_metas = reverse_tensor(
        tensor_metadata[:, self.main_voice_index, time_index_ticks + 1:, :],
        dim=1)
    central_metas = tensor_metadata[:, self.main_voice_index, time_index_ticks, :]
    return left_metas, central_metas, right_metas

    def empty_score_tensor(self, score_length):
        start_symbols = np.array([note2index[START_SYMBOL]
                                  for note2index in self.note2index_dicts])
        start_symbols = torch.from_numpy(start_symbols).long().clone()
        start_symbols = start_symbols.repeat(score_length, 1).transpose(0, 1)
        return start_symbols

    def random_score_tensor(self, score_length):
        chorale_tensor = np.array(
            [np.random.randint(len(note2index),
                               size=score_length)
             for note2index in self.note2index_dicts])
        chorale_tensor = torch.from_numpy(chorale_tensor).long().clone()
        return chorale_tensor

    def tensor_to_score(self, tensor_score,
                        fermata_tensor=None):
        """
        :param tensor_score: (num_voices, length)
        :return: music21 score object
        """
        slur_indexes = [note2index[SLUR_SYMBOL]
                        for note2index in self.note2index_dicts]

        score = music21.stream.Score()
        num_voices = tensor_score.size(0)
        name_parts = (num_voices == 4)
        part_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

        for voice_index, (voice, index2note, slur_index) in enumerate(
                zip(tensor_score,
                    self.index2note_dicts,
                    slur_indexes)):
            add_fermata = False
            if name_parts:
                part = stream.Part(id=part_names[voice_index],
                                   partName=part_names[voice_index],
                                   partAbbreviation=part_names[voice_index],
                                   instrumentName=part_names[voice_index])
            else:
                part = stream.Part(id='part' + str(voice_index))
            dur = 0
            total_duration = 0
            f = music21.note.Rest()
            for note_index in [n.item() for n in voice]:
                # if it is a played note
                if not note_index == slur_indexes[voice_index]:
                    # add previous note
                    if dur > 0:
                        f.duration = music21.duration.Duration(dur / self.subdivision)

                        if add_fermata:
                            f.expressions.append(music21.expressions.Fermata())
                            add_fermata = False

                        part.append(f)

                    dur = 1
                    f = standard_note(index2note[note_index])
                    if fermata_tensor is not None and voice_index == 0:
                        if fermata_tensor[0, total_duration] == 1:
                            add_fermata = True
                        else:
                            add_fermata = False
                    total_duration += 1

                else:
                    dur += 1
                    total_duration += 1
            # add last note
            f.duration = music21.duration.Duration(dur / self.subdivision)
            if add_fermata:
                f.expressions.append(music21.expressions.Fermata())
                add_fermata = False

            part.append(f)
            score.insert(part)
        return score