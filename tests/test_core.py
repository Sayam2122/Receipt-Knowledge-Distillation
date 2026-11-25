import math, numpy as np, torch
from sroie_data import norm_amount, values_equal, build_candidates, find_targets, make_box, assign_rows, DATE_RE
from features import featurize, FEATURE_DIM
from student import CandidateStudent, kd_loss, n_params
from teacher_qwen import combine_digit_probs, entropy
from metrics import iou


def boxes(rows):
    bs = []
    for text, x, y in rows:
        bs.append(make_box(len(bs), text, [x, x + 100], [y, y + 10], 400, 800))
    return assign_rows(bs)


def test_amount_and_equality():
    assert norm_amount("RM 1,234.5") == "1234.50"
    assert values_equal("12.50", "RM12.5", "total")
    assert not values_equal("12.50", "12.05", "total")
    assert values_equal("23-01-2019", "23/01/2019", "date")


def test_date_regex_month_names():
    assert DATE_RE.search("12 JAN 2019") and DATE_RE.search("2019-01-12")


def test_candidates_and_multi_target():
    bs = boxes([("SHOP SDN BHD", 10, 10), ("TOTAL", 10, 200), ("9.90", 300, 200), ("CASH", 10, 220), ("9.90", 300, 220)])
    tc = build_candidates(bs, "total")
    assert [c["value"] for c in tc] == ["9.90", "9.90"]
    assert find_targets(tc, "9.90", "total") == [0, 1]            # both printed copies are valid answers
    assert "total" in tc[0]["left"].lower()                        # label found on the same ROW
    assert "cash" in tc[1]["left"].lower()


def test_feature_shape_and_model_size():
    bs = boxes([("A SDN BHD", 10, 10), ("TOTAL", 10, 200), ("9.90", 300, 200)])
    for f in ("vendor", "date", "total"):
        c = build_candidates(bs, f)
        if c:
            assert featurize(c, f).shape == (len(c), FEATURE_DIM)
    assert n_params(CandidateStudent(FEATURE_DIM)) < 30000


def test_combine_digit_probs_sums_to_one_and_matches():
    p1 = [0.1, 0.7, 0.2] + [0.0] * 7
    p2 = {1: [0.0, 1.0] + [0.0] * 8}
    pr = combine_digit_probs(p1, p2, 25, 2)
    assert abs(sum(pr) - 1) < 1e-9 and pr[11] > 0.6
    one = combine_digit_probs([0.5, 0.5] + [0] * 8, {}, 2, 1)
    assert one == [0.5, 0.5] and entropy(one) == 1.0


def test_kd_loss_reduces_to_ce_and_kl_is_zero_when_equal():
    s = torch.randn(2, 4); mask = torch.ones(2, 4, dtype=torch.bool)
    tm = torch.zeros(2, 4, dtype=torch.bool); tm[0, 1] = True; tm[1, 2] = True
    zeros = torch.zeros(2, 4)
    ce = kd_loss(s, mask, tm, zeros, lam=0.5)
    ref = -torch.log_softmax(s, -1)[[0, 1], [1, 2]].mean()
    assert torch.allclose(ce, ref, atol=1e-5)
    q = torch.softmax(s, -1)       # teacher == student -> KL term 0, loss = (1-lam)*CE
    kd = kd_loss(s, mask, tm, q, lam=0.5)
    assert torch.allclose(kd, 0.5 * ref, atol=1e-5)


def test_kd_loss_padding_and_missing_targets():
    s = torch.randn(2, 5); mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    tm = torch.zeros(2, 5, dtype=torch.bool); tm[0, 0] = True                      # row 1 has no GT -> ignored
    l = kd_loss(s, mask, tm, torch.zeros(2, 5), lam=0.3)
    assert torch.isfinite(l)
    l.backward() if l.requires_grad else None


def test_iou():
    assert iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0 and iou([0, 0, .1, .1], [.5, .5, 1, 1]) == 0.0


def test_clean_vendor_matches_diagnosed_cases():
    from sroie_data import clean_vendor
    cases = {
        "STAR GROCER SDN BHD (1171881-W)": "STAR GROCER SDN BHD",
        "ECONSAVE CASH & CARRY (FC) S/B(930311-W)": "ECONSAVE CASH & CARRY (FC) S/B",
        "URBAN IDEA SDN BHD (417940-M),": "URBAN IDEA SDN BHD",
        "SEGI CASH & CARRY SDN. BHD.(317041-W)": "SEGI CASH & CARRY SDN. BHD.",
        "KING'S CONFECTIONERY S/B 273500-U (GKJ)": "KING'S CONFECTIONERY S/B",
        "TSH POWER HARDWARE": "TSH POWER HARDWARE",
        "ABC (M) SDN BHD": "ABC (M) SDN BHD",
        "TEL: 03-1234567": "TEL: 03-1234567",
    }
    for raw, want in cases.items():
        assert clean_vendor(raw) == want, (raw, clean_vendor(raw))


def test_soft_from_json_mapping():
    import numpy as np
    from soft_from_json import map_to_candidates, build
    c = [{"value": "TAN WOON YANN"}, {"value": "INDAH GIFT & HOME DECO"}, {"value": "JOHOR."}]
    q = map_to_candidates("vendor", "Indah Gift & Home Deco", c)
    assert q.argmax() == 1 and abs(q.sum() - 1) < 1e-9
    assert map_to_candidates("vendor", "TOTALLY UNRELATED STRING", c) is None
    t = [{"value": "9.00"}, {"value": "10.00"}, {"value": "9.00"}]
    q = map_to_candidates("total", "RM 9.00", t)
    assert list(np.round(q, 2)) == [0.5, 0.0, 0.5]
    assert map_to_candidates("total", "77.00", t) is None


def test_teacher_only_group_gets_pure_kl():
    s = torch.randn(1, 4); mask = torch.ones(1, 4, dtype=torch.bool); tm = torch.zeros(1, 4, dtype=torch.bool)
    q = torch.softmax(s, -1)
    assert torch.allclose(kd_loss(s, mask, tm, q, lam=0.5), torch.tensor(0.0), atol=1e-6)   # q == student -> 0
    q2 = torch.tensor([[1.0, 0, 0, 0]])
    assert kd_loss(s, mask, tm, q2, lam=0.5) > 0
    assert kd_loss(s, mask, tm, torch.zeros(1, 4), lam=0.5) == 0                            # neither signal -> ignored
