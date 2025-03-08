import re
import argparse
import numpy as np
import h5py
from nltk.tokenize import word_tokenize
import spacy
import json
from random import shuffle, seed
from collections import defaultdict
from pathlib import Path

def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !$#@~()*&^%;\[\]/\\+<>\n=])", sentence) if
            i != '' and i != ' ' and i != '\n']

def prepro_question(imgs, params):
    print('Example processed tokens:')
    for i, img in enumerate(imgs):
        s = img['question']
        # Tokenize using the selected method
        if params['token_method'] == 'nltk':
            txt = word_tokenize(str(s).lower())
        elif params['token_method'] == 'spacy':
            # Tokenize with spaCy
            doc = params['spacy'](s)
            txt = [token.norm_ for token in doc]
        else:
            txt = tokenize(s)
        img['processed_tokens'] = txt
        if i < 10:
            print(txt)
        if i % 1000 == 0:
            print(f"Processing {i}/{len(imgs)} ({i * 100.0 / len(imgs):.2f}% done)")

    return imgs

def build_vocab_question(imgs, params):
    # Build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # Count up the number of words using defaultdict to simplify the process
    counts = defaultdict(int)
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] += 1

    # Sort the words by frequency (descending order)
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print(f"Top words and their counts:")
    print('\n'.join(map(str, cw[:20])))

    # Print some stats
    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)

    print(f"Total words: {total_words}")
    print(f"Number of bad words: {len(bad_words)}/{len(counts)} = {len(bad_words) * 100.0 / len(counts):.2f}%")
    print(f"Number of words in vocab would be {len(vocab)}")
    print(f"Number of UNKs: {bad_count}/{total_words} = {bad_count * 100.0 / total_words:.2f}%")

    # Insert the special UNK token into the vocab
    print(f"Inserting the special UNK token")
    vocab.append('UNK')

    # Update each image with the final question tokens
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w, 0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs, vocab

def apply_vocab_question(imgs, wtoi):
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if w in wtoi else 'UNK' for w in txt]
        img['final_question'] = question

    return imgs

def get_top_answers(imgs, params):
    # Count the frequency of each answer
    counts = {}
    for img in imgs:
        ans = img['ans']
        counts[ans] = counts.get(ans, 0) + 1

    # Sort answers by frequency (descending)
    cw = sorted([(count, ans) for ans, count in counts.items()], reverse=True)
    print(f"Top answers and their counts:")
    print('\n'.join(map(str, cw[:20])))

    # Get top 'num_ans' answers
    vocab = [ans for _, ans in cw[:params['num_ans']]]

    return vocab


def encode_question(imgs, params, wtoi):
    max_length = params['max_length']
    N = len(imgs)

    # Initialize numpy arrays
    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        question_id[i] = img['ques_id']
        final_question = img['final_question']
        label_length[i] = min(max_length, len(final_question))

        for k, w in enumerate(final_question[:max_length]):
            label_arrays[i, k] = wtoi[w]

    return label_arrays, label_length, question_id

def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi[img['ans']]

    return ans_arrays


# def encode_mc_answer(imgs, atoi):
#     N = len(imgs)
#     mc_ans_arrays = np.zeros((N, 18), dtype='uint32')
#
#     for i, img in enumerate(imgs):
#         for j, ans in enumerate(img['MC_ans']):
#             mc_ans_arrays[i, j] = atoi.get(ans, 0)
#     return mc_ans_arrays


def filter_question(imgs, atoi):
    # Filter out images with invalid answers
    new_imgs = [img for img in imgs if img['ans'] in atoi]

    # Print the change in number of questions
    print(f'Question number reduced from {len(imgs)} to {len(new_imgs)}')

    return new_imgs


def get_unique_img(imgs):
    unique_img = list({img['img_path'] for img in imgs})
    imgtoi = {w: i for i, w in enumerate(unique_img)}
    img_pos = np.array([imgtoi.get(img['img_path'], -1) for img in imgs],
                       dtype='uint32')
    return unique_img, img_pos


def main(params):
    # Load spaCy tokenizer
    if params['token_method'] == 'spacy':
        print('Loading spaCy tokenizer for NLP...')
        params['spacy'] = spacy.load('en_core_web_sm')  # Load the spaCy model directly

    # Read training and test images
    with open(params['input_train_json'], 'r') as f:
        imgs_train = json.load(f)

    with open(params['input_test_json'], 'r') as f:
        imgs_test = json.load(f)

    # Get top answers
    top_ans = get_top_answers(imgs_train, params)
    atoi = {w: i for i, w in enumerate(top_ans)}
    itoa = {i : w for i, w in enumerate(top_ans)}

    # Filter questions not in top answers
    imgs_train = filter_question(imgs_train, atoi)

    # Make the process reproducible
    seed(123)
    shuffle(imgs_train)  # Shuffle the order

    # Tokenization and preprocessing for training and testing questions
    imgs_train = prepro_question(imgs_train, params)
    imgs_test = prepro_question(imgs_test, params)

    # Build vocabulary for questions
    imgs_train, vocab = build_vocab_question(imgs_train, params)
    itow = {i : w for i, w in enumerate(vocab)}
    wtoi = {w: i for i, w in enumerate(vocab)}

    # Encode questions for training and testing
    ques_train, ques_length_train, question_id_train = encode_question(imgs_train, params, wtoi)
    imgs_test = apply_vocab_question(imgs_test, wtoi)
    ques_test, ques_length_test, question_id_test = encode_question(imgs_test, params, wtoi)

    # Get unique images for train and test
    unique_img_train, img_pos_train = get_unique_img(imgs_train)
    unique_img_test, img_pos_test = get_unique_img(imgs_test)

    # Get the answer encoding
    A = encode_answer(imgs_train, atoi)

    # Create output h5 file for training set
    output_h5_path = Path(params['output_h5'])
    with h5py.File(output_h5_path, "w") as f:
        f.create_dataset("ques_train", dtype='uint32', data=ques_train)
        f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train)
        f.create_dataset("answers", dtype='uint32', data=A)
        f.create_dataset("question_id_train", dtype='uint32', data=question_id_train)
        f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)

        f.create_dataset("ques_test", dtype='uint32', data=ques_test)
        f.create_dataset("ques_length_test", dtype='uint32', data=ques_length_test)
        f.create_dataset("question_id_test", dtype='uint32', data=question_id_test)
        f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)

    print(f'Wrote to {output_h5_path}')

    # Create output JSON file
    output_json_path = Path(params['output_json'])
    out = {
        'ix_to_word': itow,
        'ix_to_ans': itoa,
        'unique_img_train': unique_img_train,
        'unique_img_test': unique_img_test
    }
    with open(output_json_path, 'w') as f:
        json.dump(out, f)

    print(f'Wrote to {output_json_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--num_ans', required=True, type=int,
                        help='number of top answers for the final classifications.')

    parser.add_argument('--output_json', default='../data/data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/data_prepro.h5', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=26, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_test', default=0, type=int,
                        help='number of test images (to withold until very very end)')
    parser.add_argument('--token_method', default='nltk', help='token method. set "spacy" for unigram paraphrasing')
    parser.add_argument('--spacy_data', default='spacy_data', help='location of spacy NLP model')

    parser.add_argument('--batch_size', default=10, type=int)

    args = parser.parse_args()
    params = vars(args)
    print(f"Parsed input parameters:\n{json.dumps(params, indent=2)}")
    main(params)