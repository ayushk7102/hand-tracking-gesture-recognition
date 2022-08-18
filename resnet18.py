import torch
import torchvision
from torch import nn, Tensor
from torchvision import models
from torchvision.transforms import Compose
from torchvision.transforms import functional as F
from torchmetrics.functional import f1_score, confusion_matrix

from PIL import Image, ImageOps

import os
import pandas as pd
import random
import json
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import seaborn as sns

from tqdm import tqdm
from collections import defaultdict

class_names = [
    'call',
    'dislike',
    'fist',
    'four',
    'like',
    'mute',
    'ok',
    'one',
    'palm',
    'peace',
    'rock',
    'stop',
    'stop_inverted',
    'three',
    'two_up',
    'two_up_inverted',
    'three2',
    'peace_inverted',
    'no_gesture'
]

random_seed = 42
num_classes = len(class_names)
batch_size = 64
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
show_heat_map = True


class GestureDataset(torch.utils.data.Dataset):

    def __init__(self, path, is_train=True, transform=None):

        self.path = path

        self.is_train = is_train

        self.transform = transform

        self.labels = {label: num for (label, num) in

                       zip(class_names, range(len(class_names)))}

        self.leading_hand = {'right': 0, 'left': 1}

        self.annotations = self.__read_annotations(self.path)

        users = self.annotations['user_id'].unique()

        users = sorted(users)

        random.Random(42).shuffle(users)


        train_users = users[:int(len(users) * 0.8)]

        test_users = users[int(len(users) * 0.8):]


        self.annotations = self.annotations.copy()


        if self.is_train:

            self.annotations = self.annotations[self.annotations['user_id'].isin(train_users)]

        else:

            self.annotations = self.annotations[self.annotations['user_id'].isin(test_users)]

            

    def __read_annotations(self, path):

        json_annotations = json.load(open(os.path.join(self.path, "ann_subsample.json")))

        json_annotations = [dict(annotation, **{'name': f'{name}.jpg'}) for name, annotation in zip(json_annotations, json_annotations.values())]

        annotations = pd.DataFrame(json_annotations)

        

        labels = list(annotations['labels'])

        targets = []

        for label in labels:

            targets.append([item for item in label if item != 'no_gesture'][0])


        annotations['target'] = targets

        return annotations

    

    def __prepare_image_target(self, target, name, bboxes, labels, leading_hand):


        image_pth = os.path.join(self.path, target, name)

        image = Image.open(image_pth).convert('RGB')


        width, height = image.size


        choice = np.random.choice(['gesture', 'no_gesture'], p=[0.7, 0.3])


        bboxes_by_class = {}


        for i, bbox in enumerate(bboxes):

            x1, y1, w, h = bbox

            bbox_abs = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]

            if labels[i] == 'no_gesture':

                bboxes_by_class['no_gesture'] = (bbox_abs, labels[i])

            else:

                bboxes_by_class['gesture'] = (bbox_abs, labels[i])


        if choice not in bboxes_by_class:

            choice = list(bboxes_by_class.keys())[0]


        box_scale = 1.0

        image_cropped, bbox_orig = self.get_crop_from_bbox(image, bboxes_by_class[choice][0], box_scale=box_scale)


        image_resized = ImageOps.pad(image_cropped, tuple([224, 224]), color=(0, 0, 0))


        gesture = bboxes_by_class[choice][1]


        leading_hand_class = leading_hand

        if gesture == 'no_gesture':

            leading_hand_class = 'right' if leading_hand == 'left' else 'left'


        return image_resized, gesture, leading_hand_class

    

    @staticmethod

    def get_crop_from_bbox(image, bbox, box_scale=1.):


        int_bbox = np.array(bbox).round().astype(np.int32)


        x1 = int_bbox[0]

        y1 = int_bbox[1]

        x2 = int_bbox[2]

        y2 = int_bbox[3]

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2


        w = h = max(x2 - x1, y2 - y1)

        x1 = max(0, cx - box_scale * w // 2)

        y1 = max(0, cy - box_scale * h // 2)

        x2 = cx + box_scale * w // 2

        y2 = cy + box_scale * h // 2

        x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))


        crop_image = image.crop((x1, y1, x2, y2))

        bbox_orig = np.array([x1, y1, x2, y2]).reshape(2, 2)


        return crop_image, bbox_orig


    def __len__(self):

        return self.annotations.shape[0]


    def __getitem__(self, index):


        row = self.annotations.iloc[[index]].to_dict('records')[0]


        image_resized, gesture, leading_hand = self.__prepare_image_target(

            row['target'],

            row['name'],

            row['bboxes'],

            row['labels'],

            row['leading_hand']

        )


        label = {'gesture': self.labels[gesture],

                 'leading_hand': self.leading_hand[leading_hand]}


        if self.transform is not None:

            image_resized = self.transform(image_resized)


        return image_resized, label



class ToTensor(nn.Module):
    @staticmethod
    def forward(image):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image

def get_transform():
    transforms = [ToTensor()]
    return Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=False, freezed=False):
        super().__init__()
        torchvision_model = models.resnet18(pretrained=pretrained)

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(
            torchvision_model.conv1,
            torchvision_model.bn1,
            torchvision_model.relu,
            torchvision_model.maxpool,
            torchvision_model.layer1,
            torchvision_model.layer2,
            torchvision_model.layer3,
            torchvision_model.layer4,
            torchvision_model.avgpool
        )

        num_features = torchvision_model.fc.in_features

        self.classifier = nn.Sequential(nn.Linear(num_features, num_classes))
        self.leading_hand = nn.Sequential(nn.Linear(num_features, 2))

    def forward(self, img):
        x = self.backbone(img)
        x = torch.flatten(x, 1)
        
        gesture = self.classifier(x)
        leading_hand = self.leading_hand(x)

        return {'gesture': gesture, 'leading_hand': leading_hand}

train_data = GestureDataset(path='/home/ayush/Desktop/Stuff/DVCON/HAGRID Dataset', is_train=True, transform=get_transform())
test_data = GestureDataset(path='/home/ayush/Desktop/Stuff/DVCON/HAGRID Dataset', is_train=False, transform=get_transform())

#test me
dataset_path = '/home/ayush/Desktop/Stuff/DVCON/HAGRID Dataset'

path_arr = []

for folder in os.listdir(dataset_path):
    if '.' not in folder:
        fold = (os.path.join(dataset_path, folder))
        for file in os.listdir(fold):
            path_arr.append(os.path.join(fold, file))

# for idx in range(1000):
#     img_test = test_data.__getitem__(idx)
#     print(class_names[img_test[1]['gesture']])
        # cv.imshow(class_names[img_test[1]['gesture']], cv.resize(cv.imread(path_arr[idx]), (224, 224)))
        # cv.waitKey(20)
        # print(img_test[0].numpy())
        # plt.imshow(img_test[0].numpy().shape)

short_class_names = []

for name in class_names:
    if name == 'stop_inverted':
        short_class_names.append('stop inv.')
    elif name == 'peace_inverted':
        short_class_names.append('peace inv.')
    elif name == 'two_up':
        short_class_names.append('two up')
    elif name == 'two_up_inverted':
        short_class_names.append('two up inv.')
    elif name == 'no_gesture':
        short_class_names.append('no gesture')
    else:
        short_class_names.append(name)


i = 0
plt.rcParams['figure.figsize'] = (18, 5)
plt.subplots_adjust(wspace=0, hspace=0)

# for k in range(18):
#     s = random.randint(0, len(test_data) - 1)
#     sample = test_data[s]
#     image = sample[0]
#     label = sample[1]
#     image = np.swapaxes(image, 0, 1)
#     image = np.swapaxes(image, 1, 2)
    
#     plt.subplot(2, 9, i + 1)
#     plt.title(f"{short_class_names[label['gesture']]}", fontsize=15)
#     plt.subplots_adjust(wspace=0.1, hspace=0)
#     plt.imshow(image)
#     plt.axis('off')
#     i += 1


train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

model = ResNet18(num_classes=19, pretrained=True, freezed=False).to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

model_path = '/home/ayush/Downloads/resnet18.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print('.......Loaded model state.........')

print(model)
params = []

for param in model.parameters():
    params.append(param.view(-1))
params = torch.cat(params)
print('Total no of params', params.shape)
print('Max weight: ', torch.max(params))
print('Min weight: ', torch.min(params))
print('mean: ', torch.mean(params))
print('std: ', torch.std(params))
# print('numpy', params.detach().numpy())
# exit()

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# print("PREDICTIONS")
# i = 0
# for k in range(18):
#     s = random.randint(0, len(test_data) - 1)
#     sample = test_data[s]
#     image = sample[0]
#     label = sample[1]
#     image = np.swapaxes(image, 0, 1)
#     image = np.swapaxes(image, 1, 2)

#     transform = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) 

#     img_orig = torch.reshape(image, (224, 224, 3))
#     image = torch.reshape(image, (3, 224, 224))
#     image = transform(image)
    
#     output = model(image[None, ...])
#     preds = (output['gesture'])
#     prediction = class_names[torch.argmax(preds)]
    
#     plt.subplot(2, 9, i + 1)
#     plt.title(f"pred:{prediction}", fontsize=15)
#     plt.subplots_adjust(wspace=0.1, hspace=0)
#     plt.imshow(img_orig)
#     plt.axis('off')
#     print('next', i)
#     i+=1

# plt.show()
#     i += 1
# for k in range(18):
#     s = random.randint(0, len(test_data) - 1)
#     sample = test_data[s]
#     image = sample[0]
#     label = sample[1]
    

#     image = np.swapaxes(image, 0, 1)
#     image = np.swapaxes(image, 1, 2)
#     img_orig = torch.reshape(image, (224, 224, 3))

#     image = torch.reshape(image, (3, 224, 224))
#     print('i am trying to send this ', image)
#     print('i:::details\nmean: ', torch.mean(image))
#     print('i:::details\nstd: ', torch.std(image))

#     output = model(image[None, ...])
#     preds = (output['gesture'])
#     prediction = class_names[torch.argmax(preds)]
#     print(preds)
#     print(torch.argmax(preds))
#     # cv.imshow(class_names[label['gesture']] + '::: predicted = ' + prediction, img_orig.numpy())

#     # cv.waitKey(500)
total = 0
correct = 0
batchno = 0
with torch.no_grad():
    model.eval()
    
    for data in test_dataloader:

        images, labels = data
        print('Batch : ',batchno, f'  (size = {len(images)})')
        batchno+=1

        images = torch.stack(images, dim=0)
        outputs = model(images)
        predicted, args = torch.max(outputs['gesture'], dim=1)
        
        pred_label_args = [class_names[i] for i in args.tolist()]
        label_args = [class_names[i['gesture']] for i in labels]

        # print(predicted, args)
        # print(pred_label_args, label_args)
        # print(labels)

        total += len(labels)
        correct += sum(p == t for p, t in zip(pred_label_args, label_args))
        print('accuracy so far: ', 100*(correct/total))
        
    predicts, targets = defaultdict(list), defaultdict(list)
    for i, (images, labels) in enumerate(test_dataloader):
        images = torch.stack(list(image.to(device) for image in images))
        output = model(images)

        for target in list(labels)[0].keys():
            target_labels = [label[target] for label in labels]
            predicts[target] += list(output[target].detach().cpu().numpy())
            targets[target] += target_labels
    
    f1 = {}
    for target in targets.keys():
        f1[target] = float(f1_score(torch.argmax(torch.tensor(predicts[target]), dim=1), 
                                    torch.tensor(targets[target]), average='weighted', num_classes=num_classes))
        
    print(f"f1 gestures: {round(f1['gesture'], 3)}")

    
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
if show_heat_map:
    cm = confusion_matrix(torch.tensor(predicts['gesture']), torch.tensor(targets['gesture']), num_classes)
    df_cm = pd.DataFrame(cm.numpy(), index=[i for i in short_class_names], columns=[i for i in short_class_names])

    plt.figure(figsize=(10, 8))
    hm = sns.heatmap(df_cm, annot=True, fmt='.5g', cmap='YlGnBu').get_figure()
    plt.show()


