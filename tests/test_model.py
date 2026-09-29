import torch

from facial_keypoints.model import INPUT_SIZE, OUTPUT_DIM, Net


def test_forward_pass_output_shape():
    net = Net()
    net.eval()
    x = torch.randn(2, 1, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (2, OUTPUT_DIM)


def test_forward_pass_single_sample():
    net = Net()
    net.eval()
    x = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out = net(x)
    assert out.shape == (1, OUTPUT_DIM)


def test_output_is_finite():
    net = Net()
    net.eval()
    x = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        out = net(x)
    assert torch.isfinite(out).all()


def test_gradients_flow_in_train_mode():
    net = Net()
    net.train()
    x = torch.randn(2, 1, INPUT_SIZE, INPUT_SIZE, requires_grad=True)
    out = net(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
