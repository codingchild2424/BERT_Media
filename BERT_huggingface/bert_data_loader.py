import torch
from torch.utils.data import Dataset

class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text = True):
        self.tokenizer = tokenizer #허깅페이스의 토크나이저
        self.max_length = max_length #bert가 최대로 받을 수 있는 문장 길이를 줌(보통 255?)
        self.with_text = with_text #??

    #collate function을 불러오는 것
    #samples에는 TextClassificationDataset의 리턴 값이 리스트(안에는 딕셔너리)로 들어있음
    def __call__(self, samples):
        texts = [ s['text'] for s in samples ] #딕셔너리 값을 활용해서 텍스트의 리스트를 만듦
        labels = [ s['label'] for s in samples ] #딕셔너리 값을 활용해서 레이블의 리스트를 만듦

        #tokenizer 활용
        encoding = self.tokenizer( #괄호를 열면 call이 호출됨
            texts, #텍스트
            padding = True, #패딩, 미니배치 내에서 가장 긴 길이를 기준으로 패딩을 줌
            truncation = True, #maximum length에 따라서 자르게 됨
            return_tensors = "pt", #pytorch 타입이라는 뜻
            max_length = self.max_length
        )
        #이렇게 처리하면, 각 미니배치 안에 있는 데이터가 같은 길이로 정리됨, (N, L, 1) => (샘플별, 타임스텝별, 단어 인덱스)

        """
        *패딩 관련
        튜토리얼에서는 collate func 대신, Dataloader함수에서 max_length로 처리하는 경우가 있지만,
        실제로는 미니배치 안에서 가장 큰 것에 대해서만 처리하는 것이 성능상 이점을 가짐, 따라서 위와 같이 패딩 처리를 하는 것이 좋음
        """

        return_value = {
            #encoding 값
            'input_ids': encoding['input_ids'],
            #패드가 들어간 위치는 제외하고 attention weight를 주도록 함(패드에 attn이 들어가면 안됨), 이렇게 쓰면 자동으로 처리됨
            'attention_mask': encoding['attention_mask'], 
            #리스트를 torch long 타입의 텐서로 바꿈
            'labels': torch.tensor(labels, dtype = torch.long),
        }

        #텍스트 원문이 필요한 경우에는, return_value에 text로 집어 넣음
        if self.with_text:
            return_value['text'] = texts
        
        return return_value


class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts #전체 텍스트 데이터셋(코퍼스)
        self.labels = labels #샘플별 레이블

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item): #item은 인덱스
        text = str(self.texts[item])
        label = self.labels[item]

        #여기서 바로 tensor를 return해도 되지만, collate를 거치도록 딕셔너리로 반환함
        return {
            'text': text,
            'label': label,
        }
        #여기서 나온 것은 문장마다 길이가 다르므로, collate function을 활용해서 사용
        #hugging face tokenizer가 알아서 해줌