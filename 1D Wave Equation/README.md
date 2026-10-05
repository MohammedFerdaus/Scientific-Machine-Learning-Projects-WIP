# Project 04 — 1D Wave Equation

Part of the SciML Foundations series — a structured sequence of scientific machine learning projects built entirely from scratch in Julia.

**Series:** Phase 1 Foundations (Projects 1–7)
1. PINN — Harmonic Oscillator
2. NODE — Lotka-Volterra Neural ODE
3. DeepONet — Anti-derivative Operator
4. PINN — 1D Wave Equation ← you are here
5. FNO — 1D Burgers Equation
6. GNN — Spring-Mass System
7. DeepONet — ODE Solution Operator

---

## The Problem

The 1D wave equation describes small transverse vibrations of a taut string:

$$
\frac{\partial^2 u}{\partial t^2} = c^2\,\frac{\partial^2 u}{\partial x^2}, \qquad (x,t) \in [0,1]^2
$$

with initial conditions, fixed ends, and analytical solution:

$$
\begin{aligned}
u(x,0) &= \sin(\pi x), & u_t(x,0) &= 0, \\
u(0,t) &= u(1,t) = 0, & u(x,t) &= \sin(\pi x)\cos(\pi c t).
\end{aligned}
$$

This is the first PDE in the series: two independent variables, a second derivative in each, and a new failure mode. The loss treats all times equally, so the network can fit late times before it has learned the initial condition. This project builds a PINN for it from scratch and tests causal training against an unweighted baseline.

---

## Repository Structure

```
p04_wave_equation/
├── hyperdual.jl     — HyperDual number type and arithmetic
├── network.jl       — MLP parameters, Xavier init, plain and hyperdual forward passes
├── sampling.jl      — Latin Hypercube Sampling for interior, IC and BC points
├── losses.jl        — Time binning and cumulative causal weights
├── autodiff.jl      — Second-order forward jet, backward pass, gradient assembly
├── optimizer.jl     — Adam
├── evaluate.jl      — Analytical solution, error metrics, plots
├── train.jl         — Training loop and logging (entry point)
└── gradcheck.jl     — Finite-difference verification of all gradients
```

---

## Method

The network $\hat{u}_\theta(x,t)$ is an MLP with architecture `[2, 20, 20, 20, 1]`, tanh hidden layers and a linear output, with Xavier uniform initialization. Tanh is required because the loss contains second derivatives of the network, and ReLU would make the PDE term vanish.

A PINN needs two separate kinds of derivative:

| Derivative | What | Mechanism |
|------------|------|-----------|
| Physics | $`u_{tt}`$, $`u_{xx}`$ — output with respect to inputs $`(x,t)`$ | Forward mode: hyperdual numbers |
| Training | $`\partial\mathcal{L}/\partial\theta`$ — loss with respect to every weight | Reverse mode: manual backprop |

Forward mode suits the first because there are only two inputs. Reverse mode suits the second because there are thousands of parameters and one scalar loss.

### Forward Mode — Hyperdual Numbers

A hyperdual number has the form

$$
a + b\,\varepsilon_1 + c\,\varepsilon_2 + d\,\varepsilon_1\varepsilon_2, \qquad \varepsilon_1^2 = \varepsilon_2^2 = 0.
$$

Seeding both $`\varepsilon`$'s along the same input makes the $`\varepsilon_1\varepsilon_2`$ coefficient of any smooth function exactly its second derivative, with no step size and no truncation error.

Multiplication is the product rule:

$$
(fg)_{12} = f\,g_{12} + f_1 g_2 + f_2 g_1 + f_{12}\,g
$$

and tanh applies the chain rule:

$$
d^{\text{out}}_{12} = \tanh'(a)\,d_{12} + \tanh''(a)\,d_1 d_2 .
$$

One pass seeded on $`x`$ gives $`u_{xx}`$, one seeded on $`t`$ gives $`u_{tt}`$.

The hyperdual pass is kept as an independent reference. Training uses the equivalent plain-vector jet below, which is faster and can be reversed.

### Reverse Mode Through the Jet

**Notation.** $`\sigma = \tanh`$. For a fixed seed direction $`s \in \{x, t\}`$, each layer carries the value $`h`$, first derivative $`\dot h`$ and second derivative $`\ddot h`$ (all taken with respect to $`s`$). A hidden layer maps them as:

$$
\begin{aligned}
z &= W h_{\text{prev}} + b, & \dot z &= W \dot h_{\text{prev}}, & \ddot z &= W \ddot h_{\text{prev}}, \\
h &= \sigma(z), & \dot h &= \sigma'(z)\,\dot z, & \ddot h &= \sigma'(z)\,\ddot z + \sigma''(z)\,\dot z^{2}.
\end{aligned}
$$

The derivative streams couple in one direction only, so the backward pass runs $`\ddot h \to \dot h \to h`$. Given adjoints $`H_0, H_1, H_2`$ at a layer's output (for $`h, \dot h, \ddot h`$ respectively):

$$
\begin{aligned}
\bar z &= H_0\,\sigma' + H_1\,\sigma''\,\dot z + H_2\left(\sigma''\,\ddot z + \sigma'''\,\dot z^{2}\right), \\
\bar{\dot z} &= H_1\,\sigma' + 2 H_2\,\sigma''\,\dot z, \\
\bar{\ddot z} &= H_2\,\sigma'.
\end{aligned}
$$

The parameter gradients are then

$$
\begin{aligned}
dW &= \bar z\,h_{\text{prev}}^{\top} + \bar{\dot z}\,\dot h_{\text{prev}}^{\top} + \bar{\ddot z}\,\ddot h_{\text{prev}}^{\top}, \\
db &= \bar z .
\end{aligned}
$$

Training on a loss containing $`u_{ss}`$ needs the third derivative of the activation:

$$
\sigma''' = (1-\sigma^2)(6\sigma^2 - 2).
$$

One backward function takes three output adjoints $`(g_0, g_1, g_2)`$ on $`(u, u_s, u_{ss})`$. Each loss term is a different choice of seeds:

| Loss term | Pass | $`g_0`$ | $`g_1`$ | $`g_2`$ |
|-----------|------|---------|---------|---------|
| IC (shared pass) | seed $`t`$ | $`2\lambda_{ic0}\,(u - \sin\pi x)/N_{ic}`$ | $`2\lambda_{ic1}\,u_t/N_{ic}`$ | $`0`$ |
| BC | seed $`t`$ | $`2\lambda_{bc}\,u/N_{bc}`$ | $`0`$ | $`0`$ |
| PDE, $`t`$-pass | seed $`t`$ | $`0`$ | $`0`$ | $`\bar r_i`$ |
| PDE, $`x`$-pass | seed $`x`$ | $`0`$ | $`0`$ | $`-c^2\,\bar r_i`$ |

where $`\bar r_i = 2 w_i r_i / N_f`$ and $`r_i = (u_{tt} - c^2 u_{xx})_i`$ is the PDE residual at collocation point $`i`$.

### Loss

$$
\mathcal{L} = \mathcal{L}_{\text{pde}} + \lambda_{ic0}\,\mathcal{L}_{ic0} + \lambda_{ic1}\,\mathcal{L}_{ic1} + \lambda_{bc}\,\mathcal{L}_{bc}
$$

with the four terms defined as:

## Loss Terms

| Term | Equation | Role |
|:--|:--|:--|
| $\mathcal{L}_{\text{pde}}$ | $\frac{1}{N_f}\sum_{i=1}^{N_f} w_i\,\big(u_{tt} - c^2 u_{xx}\big)_i^{2}$ | Newton's law in the interior |
| $\mathcal{L}_{ic0}$ | $\text{mean}\big[u(x,0) - \sin\pi x\big]^2$ | Initial shape |
| $\mathcal{L}_{ic1}$ | $\text{mean}\big[u_t(x,0)\big]^2$ | Released from rest |
| $\mathcal{L}_{bc}$ | $\text{mean}\big[u(0,t)^2 + u(1,t)^2\big]$ | Clamped ends |

## Causal Training

The collocation points are binned into $M$ ordered time windows. With $\mathcal{L}_j$ the mean squared residual of bin $j$, bin $i$ is weighted by

$$w_i = \exp\left(-\epsilon \sum_{j=1}^{i-1} \mathcal{L}_{j}\right)$$

so a later bin only matters once every earlier bin is well fit. The weights are recomputed every epoch and treated as constants in the backward pass (stop-gradient). $\epsilon$ is annealed geometrically from $10^{-2}$ to $10^{1}$.

### Training Setup

| Setting | Value |
|---------|-------|
| Collocation points | $`N_f = 2000`$ interior, $`N_{ic} = 100`$, $`N_{bc} = 100`$, all Latin Hypercube Sampled |
| Optimizer | Adam, $`\beta_1 = 0.9`$, $`\beta_2 = 0.999`$ |
| Learning rate | $`10^{-3}`$ decaying geometrically to $`10^{-4}`$ |
| Epochs | 5000 |

---

## Verification

Every differentiation component is checked before training. The backward pass is checked with central finite differences one output adjoint at a time, so a failure localizes to a specific term of the derivation.

| Check | Compares | Result |
|-------|----------|--------|
| Hyperdual arithmetic | Hand-derived $`x^3`$, $`\tanh(x)`$, $`\tanh(2x+1)`$, $`x^2 t`$ | Exact |
| Forward agreement | Jet vs hyperdual vs plain forward | Max diff `3.3e-16` |
| Backward, $`u`$ / $`u_s`$ / $`u_{ss}`$ | Backprop vs central differences | `~4e-9` / `~1e-8` / `~2e-9` |
| Full pipeline, $`c = 1`$ | Assembled gradient vs finite differences | `~9e-10` |
| Full pipeline, $`c = 3`$, unequal $`\lambda`$ | Assembled gradient vs finite differences | `~3e-9` |

The full-pipeline check runs with causal weights off, because finite differences would differentiate through the weights while the backward pass deliberately does not.

---

## Results

Two runs share the same seed, network and collocation points: an unweighted baseline and cumulative causal training, at $`c = 1`$.

| Run | Final total loss | Final PDE loss | Final relative $`L_2`$ error | Wall time |
|-----|------------------|----------------|------------------------------|-----------|
| Baseline | `2.08e-4` | `5.75e-5` | `0.0167` | 1208 s |
| Causal | `2.10e-4` | `5.92e-5` | `0.0169` | 1268 s |

- **No measurable difference at $`c = 1`$.** The errors are 1.67% and 1.69% on a single seed, and the loss curves and per-bin heatmaps are visually identical.
- **The baseline showed no causality failure.** At $`c = 1`$ the solution is half a period of $`\cos(\pi t)`$, which the unweighted loss fits across all times together.
- **The causal weights were probably close to 1.** Per-bin residuals are $`10^{-3}`$ to $`10^{-5}`$ and $`\epsilon`$ only reaches 10, so the exponent stays near zero. The minimum weight was not logged, so this is an inference.
- **Error is concentrated at late times.** In both runs the final error is roughly an order of magnitude larger in the last time bins than the first, with a maximum pointwise error of about 0.025.
- **Not fully converged.** The loss and $`L_2`$ curves are still decreasing slowly at 5000 epochs.

**Next:** run $`c \in \{2, 4\}`$ with the minimum causal weight logged and a larger final $`\epsilon`$, since the causality problem should only appear once $`\cos(\pi c t)`$ oscillates faster.

---

## Libraries

| Purpose | Library |
|---------|---------|
| Sampling, statistics, logging, checkpoints | `Random`, `Statistics`, `Printf`, `Serialization` (stdlib) |
| Visualization | `CairoMakie` |

The network, hyperdual numbers, forward jet, backward pass, sampler, causal weighting and Adam are all implemented from scratch.

---

## How to Run

```bash
julia train.jl
```

Trains the baseline and causal runs in sequence and writes `log.txt`, `params.jls`, `solution.png` and `training.png` to `results/<run name>/`.

```bash
julia gradcheck.jl
```

Runs the full gradient verification suite.
