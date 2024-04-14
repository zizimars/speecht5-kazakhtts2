from preprocessing import create_dataset
import code

def transliterate_text():
    replacements = [
    ('«', ''),
    ('и', 'i'),
    ('р', 'r'),
    ('7', 'jeti'),
    ('х', 'kh'),
    ('ъ', ''),
    ('”', ''),
    ('0', 'nol'),
    ('о', 'o'),
    ('8', 'segis'),
    ('\n', ' '),
    ('у', 'oo'),
    (' ', ' '),
    ('ү', 'u'),
    ('ў', 'u'),
    ('э', 'e'),
    ('к', 'k'),
    ('щ', 'shsh'),
    ('с', 's'),
    ('ы', 'y'),
    ('−', '-'),
    ('ә', 'a'),
    ('ʨ', ''),
    ('5', 'bes'),
    ('я', 'ya'),
    ('ц', 'ts'),
    ('ч', 'tch'),
    ('г', 'g'),
    ('4', 'tort'),
    ('ң', 'ng'),
    ('л', 'l'),
    ('»', ''),
    ('ө', 'o'),
    ('в', 'v'),
    ('е', 'e'),
    ('н', 'n'),
    ('б', 'b'),
    ('－', ''),
    ('м', 'm'),
    ('қ', 'k'),
    ('д', 'd'),
    ('–', '-'),
    ('і', 'i'),
    ('2', 'yeki'),
    ('ё', 'yo'),
    ('●', ''),
    ('ұ', 'u'),
    ('ф', 'f'),
    ('ж', 'j'),
    ('“', ''),
    ('й', 'i'),
    ('ь', ''),
    ('3', 'Z'),
    ('6', 'alty'),
    ('т', 't'),
    ('̆', ''),
    ('һ', 'kh'),
    ('1', 'bir'),
    ('ɕ', ''),
    ('…', ' '),
    ('з', 'z'),
    ('а', 'a'),
    ('ю', 'yu'),
    ('ə', 'a'),
    ('ғ', 'g'),
    ('п', 'p'),
    ('ш', 'sh'),
    ]

    dataset_vocab, tokenizer_vocab, dataset = create_dataset()

    def cleanup_text(inputs):
        for src, dst in replacements:
            inputs["normalized_text"] = inputs["normalized_text"].replace(src, dst)
        return inputs

    dataset_new = dataset.map(cleanup_text)
    # code.interact(local=dict(globals(), **locals()))
    # print(dataset_new[5])
    return dataset_new

if __name__ == '__main__':
    _ = transliterate_text()
