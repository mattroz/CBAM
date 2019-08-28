import torch

from cbam import CAM, SAM, CBAM

def test_CAM(input_tensor):
    c,h,w = input_tensor.shape[1:]
    cam = CAM(channels=c, h=h, w=w)
    return cam(input_tensor)

def test_SAM(input_tensor):
    c,h,w = input_tensor.shape[1:]
    sam = SAM(channels=c, h=h, w=w)
    return sam(input_tensor)

def test_CBAM(input_tensor):
    c,h,w = input_tensor.shape[1:]
    cbam = CBAM(channels=c, h=h, w=w)
    return cbam(input_tensor)

def main():
    batch_size, c, h, w = 8, 64, 10, 10
    print('Initialize tensor with shape [{},{},{},{}]'.format(batch_size, c, h, w))
    dummy_tensor = torch.autograd.Variable(torch.ones(batch_size, c, h, w))

    out = test_CAM(dummy_tensor)
    print('Shape after CAM: ', out.shape)

    out = test_SAM(dummy_tensor)
    print('Shape after SAM: ', out.shape)

    out = test_CBAM(dummy_tensor)
    print('Shape after CBAM: ', out.shape)


if __name__ == "__main__":
    main()