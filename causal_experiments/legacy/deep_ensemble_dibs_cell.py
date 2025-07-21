

import jax
import jax.random as random
import numpy as np
from typing import List, Dict, Any
import time

from dibs.target import make_nonlinear_gaussian_model
from dibs.inference import JointDiBS
from dibs.metrics import expected_shd, threshold_metrics, neg_ave_log_likelihood
from dibs.utils import visualize_ground_truth

# Set random seed for reproducibility
main_key = random.PRNGKey(42)
print(f"JAX backend: {jax.default_backend()}")

# Generate ground truth nonlinear Gaussian model
print("Generating ground truth nonlinear Gaussian model...")
key, subk = random.split(main_key)
data, graph_model, likelihood_model = make_nonlinear_gaussian_model(
    key=subk, 
    n_vars=20, 
    graph_prior_str="sf"
)

print(f"Ground truth graph has {np.sum(data.g)} edges")
print("Visualizing ground truth...")
try:
    visualize_ground_truth(data.g)
except:
    print("Visualization skipped (may not work in all environments)")

# Experiment parameters
n_ensemble_runs = 20
n_particles_svgd = 20
n_steps = 2000
callback_every = 500

print(f"\n" + "="*60)
print("EXPERIMENT SETUP")
print(f"  Deep Ensemble: {n_ensemble_runs} runs × 1 particle each")
print(f"  SVGD: 1 run × {n_particles_svgd} particles")
print(f"  Training steps: {n_steps}")
print(f"  Variables: {data.x.shape[1]}")
print(f"  Training samples: {data.x.shape[0]}")
print(f"  Test samples: {data.x_ho.shape[0]}")
print("="*60)

# Storage for results
ensemble_results = []
ensemble_metrics = {
    'eshd_empirical': [],
    'auroc_empirical': [],
    'negll_empirical': [],
    'eshd_mixture': [],
    'auroc_mixture': [],
    'negll_mixture': [],
    'training_time': []
}

# NEW: Storage for true ensemble (combining all samples)
all_ensemble_gs = []
all_ensemble_thetas = []

print("\n" + "="*60)
print("DEEP ENSEMBLE APPROACH (20 runs × 1 particle)")
print("="*60)

# Deep Ensemble: 20 runs with 1 particle each
for run_idx in range(n_ensemble_runs):
    print(f"\nRun {run_idx + 1}/{n_ensemble_runs}")
    
    # Use different seed for each run
    key, subk = random.split(key)
    
    start_time = time.time()
    
    # Create DiBS instance
    dibs = JointDiBS(
        x=data.x, 
        interv_mask=None, 
        graph_model=graph_model, 
        likelihood_model=likelihood_model
    )
    
    # Sample with 1 particle
    gs, thetas = dibs.sample(
        key=subk, 
        n_particles=1, 
        steps=n_steps, 
        callback_every=callback_every
    )
    
    training_time = time.time() - start_time
    
    # Get distributions
    dibs_empirical = dibs.get_empirical(gs, thetas)
    dibs_mixture = dibs.get_mixture(gs, thetas)
    
    # Compute metrics
    # Empirical
    eshd_emp = expected_shd(dist=dibs_empirical, g=data.g)
    auroc_emp = threshold_metrics(dist=dibs_empirical, g=data.g)['roc_auc']
    negll_emp = neg_ave_log_likelihood(
        dist=dibs_empirical, 
        eltwise_log_likelihood=dibs.eltwise_log_likelihood_observ, 
        x=data.x_ho
    )
    
    # Mixture
    eshd_mix = expected_shd(dist=dibs_mixture, g=data.g)
    auroc_mix = threshold_metrics(dist=dibs_mixture, g=data.g)['roc_auc']
    negll_mix = neg_ave_log_likelihood(
        dist=dibs_mixture, 
        eltwise_log_likelihood=dibs.eltwise_log_likelihood_observ, 
        x=data.x_ho
    )
    
    # Store results
    run_result = {
        'run_idx': run_idx,
        'eshd_empirical': eshd_emp,
        'auroc_empirical': auroc_emp,
        'negll_empirical': negll_emp,
        'eshd_mixture': eshd_mix,
        'auroc_mixture': auroc_mix,
        'negll_mixture': negll_mix,
        'training_time': training_time
    }
    
    ensemble_results.append(run_result)
    
    # Also store in lists for easy aggregation
    ensemble_metrics['eshd_empirical'].append(eshd_emp)
    ensemble_metrics['auroc_empirical'].append(auroc_emp)
    ensemble_metrics['negll_empirical'].append(negll_emp)
    ensemble_metrics['eshd_mixture'].append(eshd_mix)
    ensemble_metrics['auroc_mixture'].append(auroc_mix)
    ensemble_metrics['negll_mixture'].append(negll_mix)
    ensemble_metrics['training_time'].append(training_time)
    
    # NEW: Store samples for true ensemble
    all_ensemble_gs.append(gs)
    all_ensemble_thetas.append(thetas)
    
    print(f"  Empirical - E-SHD: {eshd_emp:5.2f}, AUROC: {auroc_emp:5.3f}, NegLL: {negll_emp:6.2f}")
    print(f"  Mixture   - E-SHD: {eshd_mix:5.2f}, AUROC: {auroc_mix:5.3f}, NegLL: {negll_mix:6.2f}")
    print(f"  Time: {training_time:.1f}s")

# NEW: Compute TRUE ENSEMBLE by combining ALL samples from all runs
print("\n" + "="*60)
print("TRUE DEEP ENSEMBLE (combining all 20 samples)")
print("="*60)

# Combine all graphs and parameters from all runs into single arrays
combined_gs = np.concatenate(all_ensemble_gs, axis=0)  # [20, d, d] 
combined_thetas = jax.tree_map(lambda *arrays: np.concatenate(arrays, axis=0), *all_ensemble_thetas)

print(f"Combined ensemble contains {combined_gs.shape[0]} total samples")

# Create a single DiBS instance to compute distributions (any will work since we're just using the method)
dibs_for_ensemble = JointDiBS(
    x=data.x, 
    interv_mask=None, 
    graph_model=graph_model, 
    likelihood_model=likelihood_model
)

# Get true ensemble distributions
true_ensemble_empirical = dibs_for_ensemble.get_empirical(combined_gs, combined_thetas)
true_ensemble_mixture = dibs_for_ensemble.get_mixture(combined_gs, combined_thetas)

# Compute metrics on true ensemble
true_eshd_emp = expected_shd(dist=true_ensemble_empirical, g=data.g)
true_auroc_emp = threshold_metrics(dist=true_ensemble_empirical, g=data.g)['roc_auc']
true_negll_emp = neg_ave_log_likelihood(
    dist=true_ensemble_empirical, 
    eltwise_log_likelihood=dibs_for_ensemble.eltwise_log_likelihood_observ, 
    x=data.x_ho
)

true_eshd_mix = expected_shd(dist=true_ensemble_mixture, g=data.g)
true_auroc_mix = threshold_metrics(dist=true_ensemble_mixture, g=data.g)['roc_auc']
true_negll_mix = neg_ave_log_likelihood(
    dist=true_ensemble_mixture, 
    eltwise_log_likelihood=dibs_for_ensemble.eltwise_log_likelihood_observ, 
    x=data.x_ho
)

print(f"TRUE ENSEMBLE Results:")
print(f"  Empirical - E-SHD: {true_eshd_emp:5.2f}, AUROC: {true_auroc_emp:5.3f}, NegLL: {true_negll_emp:6.2f}")
print(f"  Mixture   - E-SHD: {true_eshd_mix:5.2f}, AUROC: {true_auroc_mix:5.3f}, NegLL: {true_negll_mix:6.2f}")
print("\n" + "="*60)
print("SVGD APPROACH (1 run × 20 particles)")
print("="*60)

# SVGD: 1 run with 20 particles
key, subk = random.split(key)

start_time = time.time()

dibs_svgd = JointDiBS(
    x=data.x, 
    interv_mask=None, 
    graph_model=graph_model, 
    likelihood_model=likelihood_model
)

gs_svgd, thetas_svgd = dibs_svgd.sample(
    key=subk, 
    n_particles=n_particles_svgd, 
    steps=n_steps, 
    callback_every=callback_every
)

svgd_training_time = time.time() - start_time

# Get distributions
svgd_empirical = dibs_svgd.get_empirical(gs_svgd, thetas_svgd)
svgd_mixture = dibs_svgd.get_mixture(gs_svgd, thetas_svgd)

# Compute metrics
# Empirical
svgd_eshd_emp = expected_shd(dist=svgd_empirical, g=data.g)
svgd_auroc_emp = threshold_metrics(dist=svgd_empirical, g=data.g)['roc_auc']
svgd_negll_emp = neg_ave_log_likelihood(
    dist=svgd_empirical, 
    eltwise_log_likelihood=dibs_svgd.eltwise_log_likelihood_observ, 
    x=data.x_ho
)

# Mixture
svgd_eshd_mix = expected_shd(dist=svgd_mixture, g=data.g)
svgd_auroc_mix = threshold_metrics(dist=svgd_mixture, g=data.g)['roc_auc']
svgd_negll_mix = neg_ave_log_likelihood(
    dist=svgd_mixture, 
    eltwise_log_likelihood=dibs_svgd.eltwise_log_likelihood_observ, 
    x=data.x_ho
)

print(f"SVGD Results:")
print(f"  Empirical - E-SHD: {svgd_eshd_emp:5.2f}, AUROC: {svgd_auroc_emp:5.3f}, NegLL: {svgd_negll_emp:6.2f}")
print(f"  Mixture   - E-SHD: {svgd_eshd_mix:5.2f}, AUROC: {svgd_auroc_mix:5.3f}, NegLL: {svgd_negll_mix:6.2f}")
print(f"  Time: {svgd_training_time:.1f}s")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

# Compute statistics for ensemble
def compute_stats(values):
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values)
    }

print("\nDEEP ENSEMBLE STATISTICS - AVERAGE OF INDIVIDUALS (20 runs × 1 particle):")
print("-" * 70)

for metric_name in ['eshd_empirical', 'auroc_empirical', 'negll_empirical', 
                   'eshd_mixture', 'auroc_mixture', 'negll_mixture']:
    stats = compute_stats(ensemble_metrics[metric_name])
    print(f"{metric_name:15s}: {stats['mean']:6.2f} ± {stats['std']:5.2f} "
          f"[{stats['min']:5.2f}, {stats['max']:5.2f}] (median: {stats['median']:5.2f})")

training_stats = compute_stats(ensemble_metrics['training_time'])
print(f"{'training_time':15s}: {training_stats['mean']:6.1f} ± {training_stats['std']:5.1f}s "
      f"(total: {sum(ensemble_metrics['training_time']):.1f}s)")

print(f"\nTRUE DEEP ENSEMBLE STATISTICS (combined 20 samples):")
print("-" * 60)
print(f"{'eshd_empirical':15s}: {true_eshd_emp:6.2f}")
print(f"{'auroc_empirical':15s}: {true_auroc_emp:6.2f}")
print(f"{'negll_empirical':15s}: {true_negll_emp:6.2f}")
print(f"{'eshd_mixture':15s}: {true_eshd_mix:6.2f}")
print(f"{'auroc_mixture':15s}: {true_auroc_mix:6.2f}")
print(f"{'negll_mixture':15s}: {true_negll_mix:6.2f}")

print(f"\nSVGD RESULTS (1 run × 20 particles):")
print("-" * 50)
print(f"{'eshd_empirical':15s}: {svgd_eshd_emp:6.2f}")
print(f"{'auroc_empirical':15s}: {svgd_auroc_emp:6.2f}")
print(f"{'negll_empirical':15s}: {svgd_negll_emp:6.2f}")
print(f"{'eshd_mixture':15s}: {svgd_eshd_mix:6.2f}")
print(f"{'auroc_mixture':15s}: {svgd_auroc_mix:6.2f}")
print(f"{'negll_mixture':15s}: {svgd_negll_mix:6.2f}")
print(f"{'training_time':15s}: {svgd_training_time:6.1f}s")

print("\n" + "="*60)
print("COMPARISON ANALYSIS")
print("="*60)

# Compare approaches
print("\nEMPIRICAL DISTRIBUTION COMPARISON:")
print("-" * 40)
print("(A) AVERAGE-OF-INDIVIDUALS vs SVGD:")
ensemble_mean_eshd_emp = np.mean(ensemble_metrics['eshd_empirical'])
ensemble_mean_auroc_emp = np.mean(ensemble_metrics['auroc_empirical'])
ensemble_mean_negll_emp = np.mean(ensemble_metrics['negll_empirical'])

print(f"Expected SHD:")
print(f"  Deep Ensemble: {ensemble_mean_eshd_emp:5.2f} (± {np.std(ensemble_metrics['eshd_empirical']):.2f})")
print(f"  SVGD:          {svgd_eshd_emp:5.2f}")
print(f"  Difference:    {ensemble_mean_eshd_emp - svgd_eshd_emp:+5.2f} (negative is better for ensemble)")

print(f"\nAUROC:")
print(f"  Deep Ensemble: {ensemble_mean_auroc_emp:5.3f} (± {np.std(ensemble_metrics['auroc_empirical']):.3f})")
print(f"  SVGD:          {svgd_auroc_emp:5.3f}")
print(f"  Difference:    {ensemble_mean_auroc_emp - svgd_auroc_emp:+5.3f} (positive is better for ensemble)")

print(f"\nNegative Log-Likelihood:")
print(f"  Deep Ensemble: {ensemble_mean_negll_emp:6.2f} (± {np.std(ensemble_metrics['negll_empirical']):.2f})")
print(f"  SVGD:          {svgd_negll_emp:6.2f}")
print(f"  Difference:    {ensemble_mean_negll_emp - svgd_negll_emp:+6.2f} (negative is better for ensemble)")

print("\n(B) TRUE-ENSEMBLE vs SVGD:")
print(f"Expected SHD:")
print(f"  True Ensemble: {true_eshd_emp:5.2f}")
print(f"  SVGD:          {svgd_eshd_emp:5.2f}")
print(f"  Difference:    {true_eshd_emp - svgd_eshd_emp:+5.2f} (negative is better for ensemble)")

print(f"\nAUROC:")
print(f"  True Ensemble: {true_auroc_emp:5.3f}")
print(f"  SVGD:          {svgd_auroc_emp:5.3f}")
print(f"  Difference:    {true_auroc_emp - svgd_auroc_emp:+5.3f} (positive is better for ensemble)")

print(f"\nNegative Log-Likelihood:")
print(f"  True Ensemble: {true_negll_emp:6.2f}")
print(f"  SVGD:          {svgd_negll_emp:6.2f}")
print(f"  Difference:    {true_negll_emp - svgd_negll_emp:+6.2f} (negative is better for ensemble)")

print("\nMIXTURE DISTRIBUTION COMPARISON:")
print("-" * 40)
print("(A) AVERAGE-OF-INDIVIDUALS vs SVGD:")
ensemble_mean_eshd_mix = np.mean(ensemble_metrics['eshd_mixture'])
ensemble_mean_auroc_mix = np.mean(ensemble_metrics['auroc_mixture'])
ensemble_mean_negll_mix = np.mean(ensemble_metrics['negll_mixture'])

print(f"Expected SHD:")
print(f"  Deep Ensemble: {ensemble_mean_eshd_mix:5.2f} (± {np.std(ensemble_metrics['eshd_mixture']):.2f})")
print(f"  SVGD:          {svgd_eshd_mix:5.2f}")
print(f"  Difference:    {ensemble_mean_eshd_mix - svgd_eshd_mix:+5.2f} (negative is better for ensemble)")

print(f"\nAUROC:")
print(f"  Deep Ensemble: {ensemble_mean_auroc_mix:5.3f} (± {np.std(ensemble_metrics['auroc_mixture']):.3f})")
print(f"  SVGD:          {svgd_auroc_mix:5.3f}")
print(f"  Difference:    {ensemble_mean_auroc_mix - svgd_auroc_mix:+5.3f} (positive is better for ensemble)")

print(f"\nNegative Log-Likelihood:")
print(f"  Deep Ensemble: {ensemble_mean_negll_mix:6.2f} (± {np.std(ensemble_metrics['negll_mixture']):.2f})")
print(f"  SVGD:          {svgd_negll_mix:6.2f}")
print(f"  Difference:    {ensemble_mean_negll_mix - svgd_negll_mix:+6.2f} (negative is better for ensemble)")

print("\n(B) TRUE-ENSEMBLE vs SVGD:")
print(f"Expected SHD:")
print(f"  True Ensemble: {true_eshd_mix:5.2f}")
print(f"  SVGD:          {svgd_eshd_mix:5.2f}")
print(f"  Difference:    {true_eshd_mix - svgd_eshd_mix:+5.2f} (negative is better for ensemble)")

print(f"\nAUROC:")
print(f"  True Ensemble: {true_auroc_mix:5.3f}")
print(f"  SVGD:          {svgd_auroc_mix:5.3f}")
print(f"  Difference:    {true_auroc_mix - svgd_auroc_mix:+5.3f} (positive is better for ensemble)")

print(f"\nNegative Log-Likelihood:")
print(f"  True Ensemble: {true_negll_mix:6.2f}")
print(f"  SVGD:          {svgd_negll_mix:6.2f}")
print(f"  Difference:    {true_negll_mix - svgd_negll_mix:+6.2f} (negative is better for ensemble)")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

# Average-of-individuals comparison
better_empirical_avg = []
better_mixture_avg = []

if ensemble_mean_eshd_emp < svgd_eshd_emp:
    better_empirical_avg.append("E-SHD")
if ensemble_mean_auroc_emp > svgd_auroc_emp:
    better_empirical_avg.append("AUROC")
if ensemble_mean_negll_emp < svgd_negll_emp:
    better_empirical_avg.append("NegLL")

if ensemble_mean_eshd_mix < svgd_eshd_mix:
    better_mixture_avg.append("E-SHD")
if ensemble_mean_auroc_mix > svgd_auroc_mix:
    better_mixture_avg.append("AUROC")
if ensemble_mean_negll_mix < svgd_negll_mix:
    better_mixture_avg.append("NegLL")

# True ensemble comparison
better_empirical_true = []
better_mixture_true = []

if true_eshd_emp < svgd_eshd_emp:
    better_empirical_true.append("E-SHD")
if true_auroc_emp > svgd_auroc_emp:
    better_empirical_true.append("AUROC")
if true_negll_emp < svgd_negll_emp:
    better_empirical_true.append("NegLL")

if true_eshd_mix < svgd_eshd_mix:
    better_mixture_true.append("E-SHD")
if true_auroc_mix > svgd_auroc_mix:
    better_mixture_true.append("AUROC")
if true_negll_mix < svgd_negll_mix:
    better_mixture_true.append("NegLL")

print(f"AVERAGE-OF-INDIVIDUALS Deep Ensemble outperforms SVGD on:")
print(f"  Empirical distribution: {better_empirical_avg if better_empirical_avg else 'None'}")
print(f"  Mixture distribution:   {better_mixture_avg if better_mixture_avg else 'None'}")

print(f"\nTRUE Deep Ensemble outperforms SVGD on:")
print(f"  Empirical distribution: {better_empirical_true if better_empirical_true else 'None'}")
print(f"  Mixture distribution:   {better_mixture_true if better_mixture_true else 'None'}")

total_ensemble_time = sum(ensemble_metrics['training_time'])
print(f"\nComputational efficiency:")
print(f"  Deep Ensemble total time: {total_ensemble_time:.1f}s")
print(f"  SVGD time:               {svgd_training_time:.1f}s")
print(f"  Time ratio (Ensemble/SVGD): {total_ensemble_time/svgd_training_time:.1f}x")

print(f"\nIMPORTANT: This comparison demonstrates two different ways to use deep ensembles:")
print(f"  - AVERAGE-OF-INDIVIDUALS: Average the performance metrics across runs")
print(f"  - TRUE ENSEMBLE: Combine all samples into one distribution, then evaluate")
print(f"  - SVGD: Particle interaction for Bayesian inference")
print(f"\nThe TRUE ENSEMBLE approach is the proper way to evaluate ensemble methods!")
print(f"Averaging individual performances != Performance of the ensemble!")

# Save results for further analysis
results_dict = {
    'ensemble_results': ensemble_results,
    'ensemble_metrics': ensemble_metrics,
    'true_ensemble_results': {
        'eshd_empirical': true_eshd_emp,
        'auroc_empirical': true_auroc_emp,
        'negll_empirical': true_negll_emp,
        'eshd_mixture': true_eshd_mix,
        'auroc_mixture': true_auroc_mix,
        'negll_mixture': true_negll_mix,
        'combined_samples': combined_gs.shape[0]
    },
    'svgd_results': {
        'eshd_empirical': svgd_eshd_emp,
        'auroc_empirical': svgd_auroc_emp,
        'negll_empirical': svgd_negll_emp,
        'eshd_mixture': svgd_eshd_mix,
        'auroc_mixture': svgd_auroc_mix,
        'negll_mixture': svgd_negll_mix,
        'training_time': svgd_training_time
    },
    'ground_truth_edges': np.sum(data.g),
    'experiment_params': {
        'n_ensemble_runs': n_ensemble_runs,
        'n_particles_svgd': n_particles_svgd,
        'n_steps': n_steps,
        'n_vars': data.x.shape[1],
        'n_train_samples': data.x.shape[0],
        'n_test_samples': data.x_ho.shape[0]
    }
}

print(f"\nResults stored in 'results_dict' variable for further analysis.")
print(f"Individual ensemble runs available in 'ensemble_results' list.")
print(f"Ensemble aggregated metrics available in 'ensemble_metrics' dict.")

print("\n" + "="*60)
print("EXPERIMENT COMPLETED")
print("="*60) 
