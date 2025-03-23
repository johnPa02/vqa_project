import re
import argparse
import numpy as np
import h5py
from nltk.tokenize import word_tokenize
import json
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
    # vocab += ['UNK']

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
        ans_arrays[i] = atoi.get(img['ans'], len(atoi) - 1) # default: UNK

    return ans_arrays


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
    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))

    # imgs_train = imgs_train[:5000]
    # imgs_test = imgs_test[:5000]
    # get top answers
    top_ans = get_top_answers(imgs_train, params)
    atoi = {w: i for i, w in enumerate(top_ans)}
    itoa = {i: w for i, w in enumerate(top_ans)}

    # filter question, which isn't in the top answers.
    imgs_train = filter_question(imgs_train, atoi)

    # tokenization and preprocessing training question
    imgs_train = prepro_question(imgs_train, params)
    # tokenization and preprocessing testing question
    imgs_test = prepro_question(imgs_test, params)

    # create the vocab for question
    imgs_train, vocab = build_vocab_question(imgs_train, params)
    itow = {i: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i for i, w in enumerate(vocab)}  # inverse table

    ques_train, ques_length_train, question_id_train = encode_question(imgs_train, params, wtoi)

    imgs_test = apply_vocab_question(imgs_test, wtoi)
    ques_test, ques_length_test, question_id_test = encode_question(imgs_test, params, wtoi)

    # get the unique image for train and test
    unique_img_train, img_pos_train = get_unique_img(imgs_train)
    unique_img_test, img_pos_test = get_unique_img(imgs_test)

    # get the answer encoding.
    ans_train = encode_answer(imgs_train, atoi)

    ans_test = encode_answer(imgs_test, atoi)

    # get the split
    N_train = len(imgs_train)
    N_test = len(imgs_test)
    # since the train image is already suffled, we just use the last val_num image as validation
    # train = 0, val = 1, test = 2
    split_train = np.zeros(N_train)
    # split_train[N_train - params['val_num']: N_train] = 1

    split_test = np.zeros(N_test)
    split_test[:] = 2

    # create output h5 file for training set.
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("ques_train", dtype='uint32', data=ques_train)
    f.create_dataset("ques_test", dtype='uint32', data=ques_test)

    f.create_dataset("answers", dtype='uint32', data=ans_train)
    f.create_dataset("ans_test", dtype='uint32', data=ans_test)

    f.create_dataset("ques_id_train", dtype='uint32', data=question_id_train)
    f.create_dataset("ques_id_test", dtype='uint32', data=question_id_test)

    f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
    f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)

    f.create_dataset("ques_len_train", dtype='uint32', data=ques_length_train)
    f.create_dataset("ques_len_test", dtype='uint32', data=ques_length_test)
    f.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['ix_to_ans'] = itoa
    out['unique_img_train'] = unique_img_train
    out['unique_img_test'] = unique_img_test
    with open(params['output_json'], 'w') as f:
        json.dump(out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='../data/cocoqa_raw_train.json',
                        help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='../data/cocoqa_raw_test.json',
                        help='input json file to process into hdf5')

    parser.add_argument('--output_json', default='../data/cocoqa_data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='../data/cocoqa_data_prepro.h5', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=26, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='token method, nltk is much more slower.')
    parser.add_argument("--num_ans", default=20, type=int, help="Number of top answers to use")
    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)