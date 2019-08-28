import torch

from cbam import CAM

def test_CAM(input_tensor):
    c,h,w = input_tensor.shape[1:]
    cam = CAM(channels=c, h=h, w=w)
    return cam(input_tensor)

def main():
    batch_size, c, h, w = 8, 64, 16, 16
    print('Initialize tensor with shape [{},{},{},{}]'.format(batch_size, c, h, w))
    dummy_tensor = torch.autograd.Variable(torch.ones(batch_size, c, h, w))

    out = test_CAM(dummy_tensor)
    print('Shape after CAM: ', out.shape)

if __name__ == "__main__":
    main()