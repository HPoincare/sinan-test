"""
activation_maximization.py

Finds the input that maximally activates each of the 5 output neurons
of the Sinan CNN (fc4 layer, latency percentiles p90/p95/p98/p99/p99.9).

Uses projected gradient ascent with adaptive LR decay.
"""

import os
import time
import numpy as np
import mxnet as mx

_HERE       = os.path.dirname(os.path.abspath(__file__))
SYMBOL_PATH = os.path.join(_HERE, 'model', 'cnv-symbol.json')
PARAMS_PATH = os.path.join(_HERE, 'model', 'cnv-0200.params')

# data1: (1, 6, 28, 5)  -- 6 system-metric channels x 28 microservices x 5 time steps
# data2: (1, 5, 5)      -- 5 latency-percentile channels x 5 time steps
# data3: (1, 28)        -- planned CPU allocations per microservice (next step)
INPUT_SHAPES = [
    ('data1', (1, 6, 28, 5)),
    ('data2', (1, 5, 5)),
    ('data3', (1, 28)),
]
INPUT_NAMES = [name for name, _ in INPUT_SHAPES]

# Upper bounds are TBD -- set to inf until per-channel ranges are confirmed from training data.
BOUNDS = {
    'data1': (0.0, float('inf')),
    'data2': (0.0, float('inf')),
    'data3': (0.0, float('inf')),
}

PERCENTILE_NAMES = ['p90', 'p95', 'p98', 'p99', 'p99.9']
N_OUTPUTS        = len(PERCENTILE_NAMES)

# Optimizer settings
LR          = 1.0    # initial step size; gradients are normalized to max-abs=1 before scaling
N_STEPS     = 20000
CONV_WINDOW = 50     # convergence: N consecutive steps below threshold
CONV_TOL    = 0.1

# LR decay: if best activation hasn't improved in PATIENCE steps, multiply LR by FACTOR
LR_DECAY_PATIENCE = 500
LR_DECAY_FACTOR   = 0.5
LR_MIN            = 0.01
MIN_EXPLORE_STEPS = 12000


def load_params(params_path):
    """Split MXNet checkpoint into arg_params (weights) and aux_params (BN running stats)."""
    save_dict  = mx.nd.load(params_path)
    arg_params = {k[4:]: v for k, v in save_dict.items() if k.startswith('arg:')}
    aux_params = {k[4:]: v for k, v in save_dict.items() if k.startswith('aux:')}
    return arg_params, aux_params


def build_executor(arg_params, aux_params, ctx=mx.cpu()):
    """
    Bind the fc4_output subgraph to an MXNet executor with gradients enabled for the inputs.

    We attach to fc4_output rather than 'latency' because the latency node wraps fc4 in
    BlockGrad, which zeros gradients during backprop.

    grad_req is 'write' for data1/data2/data3 and 'null' for frozen weight tensors.

    BatchNorm must run with is_train=False so it uses learned running stats instead of
    per-batch stats. With batch_size=1 and is_train=True, batch variance is 0 and all
    BN gradients vanish.
    """
    sym     = mx.sym.load(SYMBOL_PATH)
    fc4_sym = sym.get_internals()['fc4_output']

    all_args = fc4_sym.list_arguments()
    grad_req = {
        name: ('write' if name in INPUT_NAMES else 'null')
        for name in all_args
    }

    shape_kwargs = {name: shape for name, shape in INPUT_SHAPES}
    executor = fc4_sym.simple_bind(ctx=ctx, grad_req=grad_req, **shape_kwargs)
    executor.copy_params_from(arg_params, aux_params)

    return executor


def activation_maximization(executor, neuron_idx,
                             lr=LR, n_steps=N_STEPS,
                             conv_window=CONV_WINDOW, conv_tol=CONV_TOL,
                             seed=None):
    """
    Projected gradient ascent to maximize fc4[neuron_idx].

    At each step:
        grad = d(fc4_i)/d(input)
        grad = grad / max(|grad|)   # normalize to prevent float32 overflow through BN
        input = clip(input + lr * grad, lo, hi)

    LR decays by LR_DECAY_FACTOR if best activation hasn't improved in LR_DECAY_PATIENCE steps.

    Returns:
        best_inputs : dict of {name -> np.ndarray}
        final_act   : best activation value seen
        curve       : {'steps', 'current', 'best'} logged every step
        meta        : {'converged': bool, 'steps_taken': int}
    """
    rng = np.random.default_rng(seed)

    def _init_inputs():
        # start at zeros -- safe baseline confirmed by sanity check in run()
        out = {}
        for (name, shape), (lo, hi) in zip(INPUT_SHAPES, BOUNDS.values()):
            out[name] = np.clip(np.zeros(shape, dtype=np.float32), lo, hi)
        return out

    inputs      = _init_inputs()
    prev_act    = None
    plateau_cnt = 0
    final_act   = None
    converged   = False

    # track the best state seen so far -- the optimizer can overshoot and never return
    best_act    = -np.inf
    best_inputs = _init_inputs()
    steps_since_improvement = 0

    curve = {'steps': [], 'current': [], 'best': []}

    for step in range(n_steps):

        # forward pass
        for name, _ in INPUT_SHAPES:
            executor.arg_dict[name][:] = mx.nd.array(inputs[name])
        executor.forward(is_train=False)

        output      = executor.outputs[0].asnumpy()
        current_act = float(output[0, neuron_idx])

        # reset on NaN -- shouldn't happen with zero init but guard anyway
        if not np.isfinite(current_act):
            inputs      = _init_inputs()
            prev_act    = None
            plateau_cnt = 0
            continue

        # update best; decay LR if we've been stuck for too long
        if current_act > best_act:
            best_act    = current_act
            best_inputs = {name: arr.copy() for name, arr in inputs.items()}
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            if steps_since_improvement >= LR_DECAY_PATIENCE and lr > LR_MIN and step >= MIN_EXPLORE_STEPS:
                lr = max(lr * LR_DECAY_FACTOR, LR_MIN)
                steps_since_improvement = 0
                inputs = {name: arr.copy() for name, arr in best_inputs.items()}
                print(f"      lr -> {lr:.4f} at step {step}")

        curve['steps'].append(step)
        curve['current'].append(current_act)
        curve['best'].append(best_act)

        # convergence check on current trajectory
        if prev_act is not None:
            if abs(current_act - prev_act) < conv_tol:
                plateau_cnt += 1
                if plateau_cnt >= conv_window:
                    print(f"    converged at step {step:5d}  fc4[{neuron_idx}] = {best_act:.4f}")
                    final_act = best_act
                    converged = True
                    break
            else:
                plateau_cnt = 0
        prev_act = current_act

        # backward pass -- seed with e_i to isolate d(fc4_i)/d(input)
        out_grad                = mx.nd.zeros((1, N_OUTPUTS))
        out_grad[0, neuron_idx] = 1.0
        executor.backward(out_grads=[out_grad])

        # normalized gradient ascent step
        for name, _ in INPUT_SHAPES:
            lo, hi = BOUNDS[name]
            grad   = executor.grad_dict[name].asnumpy()
            g_max  = np.abs(grad).max()
            if g_max > 1e-12:
                grad = grad / g_max
            inputs[name] = np.clip(inputs[name] + lr * grad, lo, hi)

    if final_act is None:
        final_act = best_act
        print(f"    max steps reached        fc4[{neuron_idx}] = {final_act:.4f}")

    meta = {'converged': converged, 'steps_taken': len(curve['steps'])}
    return best_inputs, final_act, curve, meta


def run(ctx=mx.cpu(), seed=42):
    """Run activation maximization for all 5 latency percentiles and save results."""
    print("Loading model parameters ...")
    arg_params, aux_params = load_params(PARAMS_PATH)

    print("Building executor (fc4_output) ...")
    executor = build_executor(arg_params, aux_params, ctx=ctx)

    print("Sanity check (zero input) ...")
    for name, shape in INPUT_SHAPES:
        executor.arg_dict[name][:] = mx.nd.zeros(shape)
    executor.forward(is_train=False)
    check = executor.outputs[0].asnumpy()
    if not np.isfinite(check).all():
        raise RuntimeError(f"Model produces non-finite output on zero input: {check}")
    print(f"  fc4(zeros) = {check[0]}")

    results = {}
    for i, pname in enumerate(PERCENTILE_NAMES):
        print(f"\nNeuron {i}  ({pname})")
        t0 = time.perf_counter()
        opt_inputs, final_act, curve, meta = activation_maximization(
            executor, neuron_idx=i, seed=seed + i
        )
        elapsed = time.perf_counter() - t0

        timing = {
            'elapsed_s':   elapsed,
            'steps_taken': meta['steps_taken'],
            'converged':   meta['converged'],
            's_per_step':  elapsed / max(meta['steps_taken'], 1),
        }
        results[pname] = {
            'data1':      opt_inputs['data1'],
            'data2':      opt_inputs['data2'],
            'data3':      opt_inputs['data3'],
            'activation': final_act,
            'curve':      curve,
            'timing':     timing,
        }
        print(f"    activation = {final_act:.4f} ms  ({pname})")

    # timing summary
    print(f"\n{'neuron':<8} {'conv?':<7} {'steps':>7} {'elapsed':>10} {'s/step':>9}")
    print("-" * 45)
    for pname, r in results.items():
        t = r['timing']
        conv_str = "yes" if t['converged'] else "no"
        print(f"{pname:<8} {conv_str:<7} {t['steps_taken']:>7d} "
              f"{t['elapsed_s']:>9.1f}s {t['s_per_step']:>8.4f}s")
    print(f"{'total':<8} {'':7} {'':>7} "
          f"{sum(r['timing']['elapsed_s'] for r in results.values()):>9.1f}s")

    out_path = os.path.join(_HERE, 'activation_maximization_results.npy')
    np.save(out_path, results, allow_pickle=True)
    print(f"\nResults saved -> {out_path}")

    return results


if __name__ == '__main__':
    run()
