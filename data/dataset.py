import h5py, json, torch
from torch.utils.data import Dataset

class VQADataset(Dataset):
    def __init__(self, ques_h5, img_h5, json_file, split='train'):
        self.h5_img = h5py.File(img_h5, 'r')
        self.h5_ques = h5py.File(ques_h5, 'r')
        self.json_data = json.load(open(json_file))
        self.vocab_size = len(self.json_data['ix_to_word'])
        dataset_key = 'train' if split == 'train' else 'test'
        self.img_pos = self.h5_ques[f'img_pos_{dataset_key}'][:]
        self.fv_ims = self.h5_img[f'images_{dataset_key}'][:]
        self.ques = self.h5_ques[f'ques_{dataset_key}'][:]
        self.ques_len = self.h5_ques[f'ques_len_{dataset_key}'][:]
        self.ques_id = self.h5_ques[f'ques_id_{dataset_key}'][:]
        self.answer = self.h5_ques['answers'][:] if dataset_key == 'train' else self.h5_ques[f'ans_{dataset_key}'][:]

    def __len__(self):
        return len(self.ques_id)

    def __getitem__(self, idx):
        return {
            'image': torch.tensor(self.fv_ims[self.img_pos[idx]], dtype=torch.float),
            'question': torch.tensor(self.ques[idx], dtype=torch.long),
            'question_len': torch.tensor(self.ques_len[idx], dtype=torch.long),
            'answer': torch.tensor(self.answer[idx], dtype=torch.long)
        }

def collate_fn(batch):
    batch.sort(key=lambda x: x['question_len'], reverse=True)
    questions = torch.stack([item['question'] for item in batch])
    lengths = torch.tensor([item['question_len'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    answers = torch.tensor([item['answer'] for item in batch])
    return questions, lengths, images, answers
