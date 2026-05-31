import torch
from pocketgrad.engine import Scalar


def assert_close(actual, expected, tol=1e-6):
    assert abs(actual - expected) < tol


def test_chain_expr():
    x = Scalar(-3.5)
    y = Scalar(1.25)

    out = ((x * y) + x.relu()) * (y + 2.0) + (x - y) ** 2
    out.backward()

    x_pg, y_pg, out_pg = x, y, out

    x = torch.tensor([-3.5], dtype=torch.double, requires_grad=True)
    y = torch.tensor([1.25], dtype=torch.double, requires_grad=True)

    out = ((x * y) + x.relu()) * (y + 2.0) + (x - y) ** 2
    out.backward()

    x_th, y_th, out_th = x, y, out

    assert_close(out_pg.data, out_th.item())
    assert_close(x_pg.grad, x_th.grad.item())
    assert_close(y_pg.grad, y_th.grad.item())


def test_reused_subgraph():
    a = Scalar(2.0)
    b = Scalar(-1.5)

    shared = a * b + a + 0.5
    out = shared * shared + shared.relu() + b / a
    out.backward()

    a_pg, b_pg, out_pg = a, b, out

    a = torch.tensor([2.0], dtype=torch.double, requires_grad=True)
    b = torch.tensor([-1.5], dtype=torch.double, requires_grad=True)

    shared = a * b + a + 0.5
    out = shared * shared + shared.relu() + b / a
    out.backward()

    a_th, b_th, out_th = a, b, out

    assert_close(out_pg.data, out_th.item())
    assert_close(a_pg.grad, a_th.grad.item())
    assert_close(b_pg.grad, b_th.grad.item())


def test_mixed_ops():
    p = Scalar(-0.75)
    q = Scalar(3.0)
    r = Scalar(0.5)

    m = (p + q) * r
    n = (m.relu() + q / 2.0) * (p - r)
    out = n + p**2 - (q**2) / 3.0 + 4.0
    out.backward()

    p_pg, q_pg, r_pg, out_pg = p, q, r, out

    p = torch.tensor([-0.75], dtype=torch.double, requires_grad=True)
    q = torch.tensor([3.0], dtype=torch.double, requires_grad=True)
    r = torch.tensor([0.5], dtype=torch.double, requires_grad=True)

    m = (p + q) * r
    n = (m.relu() + q / 2.0) * (p - r)
    out = n + p**2 - (q**2) / 3.0 + 4.0
    out.backward()

    p_th, q_th, r_th, out_th = p, q, r, out

    assert_close(out_pg.data, out_th.item())
    assert_close(p_pg.grad, p_th.grad.item())
    assert_close(q_pg.grad, q_th.grad.item())
    assert_close(r_pg.grad, r_th.grad.item())