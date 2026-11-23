"""Exact Discrete Gaussian Mechanism Sampler (Canonne, Kamath & Steinke, NeurIPS 2020).

Implements Algorithms 1, 2, and 3 from:
  Canonne, Kamath, Steinke (2020), "The Discrete Gaussian for Differential Privacy", NeurIPS 2020.

Design invariants:
1. Zero floating-point arithmetic and zero math.exp/random.gauss in noise sampling.
   All probability parameters are exact `fractions.Fraction` values and all coin flips
   use `secrets.SystemRandom()` (OS CSPRNG via os.urandom) by default.
   Inner loops use exact integer (numerator, denominator) pairs to avoid repeated
   Fraction/GCD allocations while remaining 100% exact.
2. Exact integer tail-probability threshold calibration via high-precision conservative
   directed rounding (Decimal prec=60, monotonic multiplicative recurrence with
   ROUND_CEILING/.next_plus() upper bounds and ROUND_FLOOR/.next_minus() lower bounds),
   guaranteeing Pr[1 + Z > tau] = Pr[Z >= tau] <= delta_j strictly.
3. Conservative exact rational zCDP budget target rho_T derived from (epsilon, delta/2)
   using Decimal(prec=100) directed rounding and verified without ad-hoc 1-1e-12 hacks.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from functools import lru_cache
import math
import os
import secrets
from typing import Dict, Optional, Protocol, Tuple, Union


class IntegerRNG(Protocol):
    def randrange(self, stop: int) -> int: ...


class BufferedOSCSPRNG:
    """Pure os.urandom() CSPRNG buffered in 64KB blocks for syscall efficiency.

    Strictly cryptographic OS entropy (identical source to secrets.SystemRandom);
    uses zero PRNG state expansion and consumes bits sequentially.
    """

    def __init__(self, chunk_bytes: int = 65536) -> None:
        self._chunk = chunk_bytes
        self._buf = b""
        self._pos = 0

    def _getrandbits(self, k: int) -> int:
        nbytes = (k + 7) >> 3
        if self._pos + nbytes > len(self._buf):
            rem = self._buf[self._pos :]
            need = max(self._chunk, nbytes)
            self._buf = rem + os.urandom(need)
            self._pos = 0
        raw = self._buf[self._pos : self._pos + nbytes]
        self._pos += nbytes
        val = int.from_bytes(raw, "big")
        excess = (nbytes << 3) - k
        return val >> excess if excess else val

    def randrange(self, stop: int) -> int:
        k = stop.bit_length()
        while True:
            r = self._getrandbits(k)
            if r < stop:
                return r


_DEFAULT_CSPRNG = secrets.SystemRandom()


def _sample_bernoulli_int(num: int, den: int, rng: IntegerRNG) -> int:
    if num <= 0:
        return 0
    if num >= den:
        return 1
    return 1 if rng.randrange(den) < num else 0


def _sample_bernoulli_exp1_int(num: int, den: int, rng: IntegerRNG) -> int:
    """CKS20 Algorithm 1 for 0 <= num/den <= 1: exact Bernoulli(exp(-num/den))."""
    if num == 0:
        return 1
    k = 1
    den_k = den
    while True:
        # Bernoulli((num / den) / k) == Bernoulli(num / (den * k))
        if rng.randrange(den_k) >= num:
            return k & 1
        k += 1
        den_k += den


def _sample_bernoulli_exp_int(num: int, den: int, rng: IntegerRNG) -> int:
    """CKS20 Algorithm 1 for rational gamma = num/den >= 0: exact Bernoulli(exp(-gamma))."""
    if num == 0:
        return 1
    floor_g, rem_num = divmod(num, den)
    for _ in range(floor_g):
        # exp(-1): k=1.. with 1/k
        k = 1
        while True:
            if rng.randrange(k) >= 1:
                if (k & 1) == 0:
                    return 0
                break
            k += 1
    return _sample_bernoulli_exp1_int(rem_num, den, rng)


def sample_bernoulli(p: Fraction, rng: Optional[IntegerRNG] = None) -> int:
    """Sample Bernoulli(p) for exact rational p in [0, 1] using integer CSPRNG."""
    if not isinstance(p, Fraction):
        raise TypeError(f"p must be Fraction, got {type(p).__name__}")
    if p < 0 or p > 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    r = _DEFAULT_CSPRNG if rng is None else rng
    return _sample_bernoulli_int(p.numerator, p.denominator, r)


def sample_bernoulli_exp(gamma: Fraction, rng: Optional[IntegerRNG] = None) -> int:
    """CKS20 Algorithm 1 (gamma >= 0 rational): exact Bernoulli(exp(-gamma))."""
    if not isinstance(gamma, Fraction):
        raise TypeError(f"gamma must be Fraction, got {type(gamma).__name__}")
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    r = _DEFAULT_CSPRNG if rng is None else rng
    return _sample_bernoulli_exp_int(gamma.numerator, gamma.denominator, r)


def sample_dlaplace(scale: Fraction, rng: Optional[IntegerRNG] = None) -> int:
    """CKS20 Algorithm 2: exact Discrete Laplace DLap(scale) with PMF propto exp(-|x|/scale).

    Works for any positive rational scale = r / s in O(1) expected rejections.
    """
    if not isinstance(scale, Fraction):
        raise TypeError(f"scale must be Fraction, got {type(scale).__name__}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    r_rng = _DEFAULT_CSPRNG if rng is None else rng
    r_num = scale.numerator
    s_den = scale.denominator
    while True:
        while True:
            u = r_rng.randrange(r_num)
            if _sample_bernoulli_exp1_int(u, r_num, r_rng) == 1:
                break
        v = 0
        while True:
            k = 1
            while True:
                if r_rng.randrange(k) >= 1:
                    res = k & 1
                    break
                k += 1
            if res == 0:
                break
            v += 1
        z = r_num * v + u
        w = z // s_den if s_den != 1 else z
        b = r_rng.randrange(2)
        if b == 1 and w == 0:
            continue
        return -w if b == 1 else w


def sample_dgauss(sigma2: Fraction, rng: Optional[IntegerRNG] = None) -> int:
    """CKS20 Algorithm 3: exact Discrete Gaussian N_Z(0, sigma2) with PMF propto exp(-x^2/(2*sigma2)).

    Theorem 4 / Corollary 5 (Canonne, Kamath & Steinke 2020):
    Adding Z ~ N_Z(0, sigma2) to a sensitivity-1 integer statistic satisfies
    exact rho = 1 / (2 * sigma2)-zCDP with zero floating-point approximation error.
    """
    if not isinstance(sigma2, Fraction):
        raise TypeError(f"sigma2 must be Fraction, got {type(sigma2).__name__}")
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be > 0, got {sigma2}")
    r_rng = _DEFAULT_CSPRNG if rng is None else rng
    s2_num = sigma2.numerator
    s2_den = sigma2.denominator
    # Integer proposal scale t = floor(sigma) + 1 >= 1, computed via exact integer sqrt.
    t = math.isqrt(s2_num // s2_den) + 1
    # gamma = (t*|y| - s2_num/s2_den)^2 / (2 * t^2 * s2_num/s2_den)
    #       = (t * s2_den * |y| - s2_num)^2 / (2 * t^2 * s2_num * s2_den)
    t_s2_den = t * s2_den
    gamma_den = 2 * t * t * s2_num * s2_den
    while True:
        # Sample Y ~ DLap(t) (integer scale t => s_den=1)
        while True:
            while True:
                u = r_rng.randrange(t)
                if _sample_bernoulli_exp1_int(u, t, r_rng) == 1:
                    break
            v = 0
            while True:
                k = 1
                while True:
                    if r_rng.randrange(k) >= 1:
                        res = k & 1
                        break
                    k += 1
                if res == 0:
                    break
                v += 1
            w = t * v + u
            b = r_rng.randrange(2)
            if not (b == 1 and w == 0):
                break
        diff = t_s2_den * w - s2_num
        gamma_num = diff * diff
        if _sample_bernoulli_exp_int(gamma_num, gamma_den, r_rng) == 1:
            return -w if b == 1 else w


def conservative_eps_from_rho(rho: Fraction, delta_conv: Fraction) -> Decimal:
    """Rigorous upper bound on rho + 2*sqrt(rho*ln(1/delta_conv)) via Decimal(prec=100, ROUND_CEILING)."""
    if rho <= 0 or not (0 < delta_conv < 1):
        raise ValueError("rho > 0 and 0 < delta_conv < 1 required")
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.rounding = ROUND_CEILING
        rho_d = Decimal(rho.numerator) / Decimal(rho.denominator)
        inv_d = Decimal(delta_conv.denominator) / Decimal(delta_conv.numerator)
        L_hi = inv_d.ln().next_plus().next_plus()
        return rho_d + Decimal(2) * (rho_d * L_hi).sqrt().next_plus()


def rational_rho_budget(
    epsilon: Union[Fraction, int],
    delta_conv: Fraction,
    scale_denom: int = 10**15,
) -> Fraction:
    """Compute exact rational rho_T <= rho_bar(epsilon, delta_conv) verified by conservative_eps_from_rho."""
    eps_f = Fraction(epsilon)
    if eps_f <= 0 or not (0 < delta_conv < 1):
        raise ValueError("positive epsilon and 0 < delta_conv < 1 required")
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.rounding = ROUND_CEILING
        eps_d = Decimal(eps_f.numerator) / Decimal(eps_f.denominator)
        inv_d = Decimal(delta_conv.denominator) / Decimal(delta_conv.numerator)
        L_hi = inv_d.ln().next_plus().next_plus()
    with localcontext() as ctx:
        ctx.prec = 100
        ctx.rounding = ROUND_FLOOR
        s_lo = (L_hi + eps_d).sqrt() - L_hi.sqrt()
        rho_approx = s_lo * s_lo
        n = int((rho_approx * Decimal(scale_denom)).to_integral_value(rounding=ROUND_FLOOR))
    rho_rat = Fraction(max(1, n - 2), scale_denom)
    while Fraction(conservative_eps_from_rho(rho_rat, delta_conv)) > eps_f:
        rho_rat -= Fraction(1, scale_denom)
        if rho_rat <= 0:
            raise ValueError("Failed to find positive rational rho_T")
    return rho_rat


@lru_cache(maxsize=None)
def exact_dgauss_tau(sigma2: Fraction, delta_j: Fraction) -> Tuple[int, Dict[str, str]]:
    """Compute exact minimum integer tau >= 1 such that Pr_{Z~N_Z(0,sigma2)}[1 + Z > tau] <= delta_j.

    Because Z is integer-valued, {1 + Z > tau} is identical to {Z >= tau}.
    Uses rigorous directed rounding (prec=60) with a monotonic multiplicative recurrence
    for exp(-k^2/(2*sigma2)) (2 Decimal.exp calls total), guaranteeing upper and lower bounds.
    """
    if not isinstance(sigma2, Fraction) or not isinstance(delta_j, Fraction):
        raise TypeError("sigma2 and delta_j must be Fractions")
    if sigma2 <= 0 or not (0 < delta_j < Fraction(1, 2)):
        raise ValueError("sigma2 > 0 and 0 < delta_j < 1/2 required")

    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_CEILING
        s2_hi = Decimal(sigma2.numerator) / Decimal(sigma2.denominator)
        two_s2_hi = Decimal(2) * s2_hi
        delta_d = Decimal(delta_j.numerator) / Decimal(delta_j.denominator)
        # Cut off where exp(-K^2/(2*sigma2)) < delta_j * 1e-20
        target_exp = (-delta_d.ln()) + Decimal(50)
        k_max = max(20, int(((two_s2_hi * target_exp).sqrt()).to_integral_value(rounding=ROUND_CEILING)) + 5)

        step1_hi = (-Decimal(1) / two_s2_hi).exp().next_plus().next_plus()
        r0_hi = (-Decimal(1) / s2_hi).exp().next_plus().next_plus()

        w_hi = [Decimal(1)] * (k_max + 1)
        step = step1_hi
        for k in range(1, k_max + 1):
            w_hi[k] = (w_hi[k - 1] * step).next_plus()
            step = (step * r0_hi).next_plus()

        k_next = k_max + 1
        term_next = (-Decimal(k_next * k_next) / two_s2_hi).exp().next_plus().next_plus()
        ratio_hi = (-Decimal(k_next) / s2_hi).exp().next_plus().next_plus()
        assert ratio_hi < Decimal(1), ratio_hi
        tail_rem_hi = term_next / (Decimal(1) - ratio_hi).next_minus()

    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_FLOOR
        s2_lo = Decimal(sigma2.numerator) / Decimal(sigma2.denominator)
        two_s2_lo = Decimal(2) * s2_lo
        delta_d_lo = Decimal(delta_j.numerator) / Decimal(delta_j.denominator)

        step1_lo = (-Decimal(1) / two_s2_lo).exp().next_minus().next_minus()
        r0_lo = (-Decimal(1) / s2_lo).exp().next_minus().next_minus()
        if step1_lo < 0:
            step1_lo = Decimal(0)
        if r0_lo < 0:
            r0_lo = Decimal(0)

        w_lo = [Decimal(1)] * (k_max + 1)
        step = step1_lo
        for k in range(1, k_max + 1):
            val = (w_lo[k - 1] * step).next_minus()
            w_lo[k] = val if val > Decimal(0) else Decimal(0)
            s_next = (step * r0_lo).next_minus()
            step = s_next if s_next > Decimal(0) else Decimal(0)

        denom_lo = Decimal(1) + Decimal(2) * sum(w_lo[1:])

    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_CEILING
        denom_hi = Decimal(1) + Decimal(2) * (sum(w_hi[1:]) + tail_rem_hi)

    suf_hi = [Decimal(0)] * (k_max + 2)
    suf_lo = [Decimal(0)] * (k_max + 2)
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_CEILING
        acc = tail_rem_hi
        suf_hi[k_max + 1] = acc
        for k in range(k_max, 0, -1):
            acc = acc + w_hi[k]
            suf_hi[k] = acc
    with localcontext() as ctx:
        ctx.prec = 60
        ctx.rounding = ROUND_FLOOR
        acc = Decimal(0)
        for k in range(k_max, 0, -1):
            acc = acc + w_lo[k]
            suf_lo[k] = acc

    chosen_tau = None
    chosen_upper = None
    prev_lower = None
    with localcontext() as ctx:
        ctx.prec = 60
        for tau in range(1, k_max + 1):
            ctx.rounding = ROUND_CEILING
            p_hi = suf_hi[tau] / denom_lo
            if p_hi <= delta_d_lo:
                chosen_tau = tau
                chosen_upper = p_hi
                prev_hi = ((suf_hi[tau - 1] if tau > 1 else 1 + suf_hi[1]) / denom_lo)
                ctx.rounding = ROUND_FLOOR
                chosen_lo = suf_lo[tau] / denom_hi
                prev_lower = ((suf_lo[tau - 1] if tau > 1 else 1 + suf_lo[1]) / denom_hi)
                break

    if chosen_tau is None or Fraction(prev_lower) <= delta_j:
        raise ArithmeticError("Insufficient precision to certify the minimum threshold")

    def endpoint(value, rounding):
        with localcontext() as ctx:
            ctx.prec = 15
            ctx.rounding = rounding
            f = Fraction(+value)
            return f"{f.numerator}/{f.denominator}"

    bounds = {
        "tail_lower": endpoint(chosen_lo, ROUND_FLOOR),
        "tail_upper": endpoint(chosen_upper, ROUND_CEILING),
        "prev_tail_lower": endpoint(prev_lower, ROUND_FLOOR),
        "prev_tail_upper": endpoint(prev_hi, ROUND_CEILING),
    }
    if Fraction(bounds["tail_upper"]) > delta_j or Fraction(bounds["prev_tail_lower"]) <= delta_j:
        raise ArithmeticError("Export precision insufficient for threshold certificate")
    return chosen_tau, bounds


def dgauss_pmf_table(sigma2: Fraction, support_radius: int) -> Dict[int, float]:
    """Exact theoretical PMF of N_Z(0, sigma2) truncated for unit-test statistical checks."""
    with localcontext() as ctx:
        ctx.prec = 60
        s2 = Decimal(sigma2.numerator) / Decimal(sigma2.denominator)
        two_s2 = Decimal(2) * s2
        k_lim = max(support_radius + 20, int((two_s2 * Decimal(140)).sqrt()) + 10)
        weights = {k: (-Decimal(k * k) / two_s2).exp() for k in range(-k_lim, k_lim + 1)}
        z = sum(weights.values())
        return {k: float(weights[k] / z) for k in range(-support_radius, support_radius + 1)}
