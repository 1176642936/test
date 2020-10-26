import torch
from  flask import Flask,request
from model import BLSTMCTCNetwork
from lengyue_dl.utils import encode
from  lengyue_dl.utils.image import load,to_rgb
import torchvision.transforms as T

app = Flask(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
CAPTCHA_MAX_LENGTH = 5                   # 验证码长度
CHAR_SET = '_2345678abcdefghkmnpqrsuvwxyz'
CAPTCHA_CHARS = len(CHAR_SET)                # 分类数
IMAGE_SHAPE = (40, 120)

model = BLSTMCTCNetwork(IMAGE_SHAPE, CAPTCHA_CHARS)
model.load_state_dict(torch.load("./models/save_23.models"))
model = model.to(DEVICE)
model.eval()  # 转入测试模式, 更新BN计算方法

@app.route("/ocr", methods=["POST"])
def api():
    binary = request.data
    images = load(binary, None)
    images = T.Compose([
            to_rgb,
            T.ToPILImage(),
            T.Resize(IMAGE_SHAPE),
            T.ToTensor(),
            T.Normalize((0.91585117, 0.87477658, 0.92081706), (0.13103009, 0.19668907, 0.13226289))
        ])(images)
    images = images.to(DEVICE)
    images = images.unsqueeze(0)
    predicts = model(images)
    predict_text = ''
    for i in range(predicts.shape[1]):
        predict = predicts[:, i, :]
        predict = predict.argmax(1)
        predict = predict.contiguous()
        predict_text = encode.ctc_to_str(predict, CHAR_SET)
    return  predict_text

if __name__ == '__main__':
    app.run()
