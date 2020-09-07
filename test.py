import glob
from torch.utils.data import DataLoader
from tqdm import tqdm, tqdm_notebook
from torch.autograd import Variable
from baseline_CRNN import load_model, Resize, TextDataset, text_collate
from torchvision.transforms import Compose
import pandas as pd

test_path = glob.glob('../tianchi_data/mchar_test_a/*')
test_label = [[1]] * len(test_path)
test_path.sort()


# 预测代码
# In[11]:


def predict(net, data, abc, cuda, visualize, batch_size=50):
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=text_collate)

    count = 0
    tp = 0
    avg_ed = 0
    out = []
    iterator = tqdm_notebook(data_loader)
    for sample in iterator:
        imgs = Variable(sample["img"])
        if cuda:
            imgs = imgs.cuda()
        out += net(imgs, decode=True)
        # print(out)
        # break
    return out


if __name__ == '__main__':
    model = load_model('0123456789', seq_proj=[7, 30], backend='resnet18', snapshot='crnn_resnet18_0123456789_best', cuda=True)

    transform = Compose([
        # Rotation(),
        # Translation(),
        # Scale(),
        Resize(size=(200, 100))
        ])
    test_data = TextDataset(test_path, test_label, transform=transform)

    model.training = False
    test_predict = predict(model, test_data, '0123456789', True, False, batch_size=50)



    df_submit = pd.read_csv('../tianchi_data/mchar_sample_submit_A.csv')
    print(test_predict)
    df_submit['file_code'] = test_predict
    df_submit.to_csv('../tianchi_data/mchar_sample_submit_A.csv', index=None)







