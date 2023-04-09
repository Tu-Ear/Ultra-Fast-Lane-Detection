import torch, os, cv2
from PIL import Image
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor

if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------
    infer_work_dir = '/data/ldp/zjf/infer'
    backbone = '18'

    # test_model = '/data/ldp/zjf/log/Tusimple/Official/tusimple_18.pth'
    # row_anchor = tusimple_row_anchor
    # griding_num = 100
    # cls_num_per_lane = 56

    test_model = '/data/ldp/zjf/log/CULane/Official/culane_18.pth'
    row_anchor = culane_row_anchor
    griding_num = 200
    cls_num_per_lane = 18

    # CULane
    img_w, img_h = 1640, 590
    img_path = '/data/ldp/zjf/dataset/CULane/driver_193_90frame/06042022_0515.MP4/'
    img_name = '00810.jpg'

    # Tusimple
    # img_w, img_h = 1280, 720
    # img_path = '/data/ldp/zjf/dataset/Tusimple/clips/0313-2/10020'
    # img_name = '11.jpg'
    # ----------------------------------------------------------------------------------------------

    torch.backends.cudnn.benchmark = True

    dist_print('start testing...')

    net = parsingNet(pretrained=False, backbone=backbone, cls_dim=(griding_num + 1, cls_num_per_lane, 4),
                     use_aux=False).cuda()  # we dont need auxiliary segmentation in testing

    state_dict = torch.load(test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    img = Image.open(os.path.join(img_path, img_name))
    img = img_transforms(img)
    img = np.reshape(img, (1, 3, 288, 800))
    img = img.cuda()

    if not os.path.exists(infer_work_dir):
        os.makedirs(infer_work_dir)

    with torch.no_grad():
        out = net(img)

    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]

    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc

    # import pdb; pdb.set_trace()
    vis = cv2.imread(os.path.join(img_path, img_name))
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                           int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1)
                    cv2.circle(vis, ppp, 5, (0, 255, 0), -1)

    cv2.imwrite(os.path.join(infer_work_dir, img_name), vis)