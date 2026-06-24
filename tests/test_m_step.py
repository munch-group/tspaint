"""Closed-form M-step updates (CLAUDE.md §3.4). These are pure and so testable
ahead of the E-step orchestration (Rungs 4-5)."""
import numpy as np

from tslai.em import m_step_Q, m_step_pi, m_step_w


def test_m_step_Q_basic():
    S_dwell = np.array([2.0, 4.0])
    S_jumps = np.array([[0.0, 1.0], [2.0, 0.0]])
    Q = m_step_Q(S_dwell, S_jumps)
    assert np.isclose(Q[0, 1], 0.5)        # 1 / 2
    assert np.isclose(Q[1, 0], 0.5)        # 2 / 4
    assert np.isclose(Q[0, 0], -0.5)
    assert np.isclose(Q[1, 1], -0.5)
    assert np.allclose(Q.sum(axis=1), 0.0)


def test_m_step_Q_zero_dwell_is_safe():
    Q = m_step_Q(np.array([0.0, 1.0]), np.array([[0.0, 5.0], [1.0, 0.0]]))
    assert np.allclose(Q[0], 0.0)          # no dwell -> no outgoing rate, no div-by-zero
    assert np.isclose(Q[1, 0], 1.0)
    assert np.allclose(Q.sum(axis=1), 0.0)


def test_m_step_pi():
    assert np.allclose(m_step_pi(np.array([3.0, 1.0])), [0.75, 0.25])
    assert np.allclose(m_step_pi(np.array([0.0, 0.0])), [0.5, 0.5])   # degenerate fallback


def test_m_step_w_matches_beta_map():
    w = m_step_w(agree=8.0, disagree=2.0, alpha=5.0, beta=1.0)
    assert np.isclose(w, (5 - 1 + 8) / (5 + 1 - 2 + 8 + 2))           # 12 / 14
    assert 0.0 <= w <= 1.0
    # a strong Beta(alpha>>1, 1) prior keeps credibility high under mild disagreement
    assert m_step_w(agree=10.0, disagree=1.0, alpha=50.0, beta=1.0) > 0.95
