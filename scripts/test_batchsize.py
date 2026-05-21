#!/usr/bin/env python3
"""
Test script to find optimal batch size for single GPU training.
Tests memory usage and throughput with different batch sizes.
"""

import os
import sys
import torch
import argparse
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from r2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian
from r2_gaussian.dataset import Scene
from r2_gaussian.utils.loss_utils import l1_loss, ssim
from r2_gaussian.utils.general_utils import safe_state


def test_batch_size(dataset_path, batch_size, num_iterations=50):
    """Test training with given batch size."""
    
    print(f"\n{'='*60}")
    print(f"Testing batch_size={batch_size}")
    print(f"{'='*60}")
    
    # Clean GPU cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Setup dataset with all required attributes
        class DummyDataset:
            def __init__(self, source_path):
                self.source_path = source_path
                self.model_path = None
                self.scale_min = -1.0
                self.scale_max = -1.0
                self.sh_degree = 3
                self.white_background = False
                self.random_background = False
                self.eval = False
                self.data_device = "cuda"
                self.images_per_gpu_ema = 0
        
        dataset = DummyDataset(dataset_path)
        scene = Scene(dataset, shuffle=False)
        
        # Setup Gaussians
        gaussians = GaussianModel(None)
        initialize_gaussian(gaussians, dataset, None)
        
        # Setup optimizer (simple version)
        opt_params = OptimizationParams(argparse.Namespace())
        gaussians.training_setup(opt_params)
        
        # Setup rendering params
        pipe_params = PipelineParams(argparse.Namespace())
        
        # Get query function
        scanner_cfg = scene.scanner_cfg
        queryfunc = lambda x: query(
            x,
            scanner_cfg["offOrigin"],
            scanner_cfg["nVoxel"],
            scanner_cfg["sVoxel"],
            pipe_params,
        )
        
        # Get training cameras
        train_cameras = scene.getTrainCameras()
        if len(train_cameras) == 0:
            print(f"  ERROR: No training cameras found")
            return None
        
        # Test iterations
        times = []
        peak_memory = 0
        
        print(f"  Dataset: {len(train_cameras)} cameras")
        print(f"  Batch size: {batch_size}")
        print(f"  Running {num_iterations} iterations...")
        
        for iter_idx in range(num_iterations):
            iter_start = time.time()
            
            # Sample batch of cameras
            batch_cameras = np.random.choice(train_cameras, size=min(batch_size, len(train_cameras)), replace=False)
            
            # Render and compute loss
            total_loss = 0.0
            for viewpoint in batch_cameras:
                render_pkg = render(viewpoint, gaussians, pipe_params)
                image = render_pkg["render"]
                
                gt_image = viewpoint.original_image.cuda()
                loss = l1_loss(image, gt_image)
                total_loss += loss / batch_size
            
            # Backward
            total_loss.backward()
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            
            iter_time = time.time() - iter_start
            times.append(iter_time)
            
            # Track memory
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            peak_memory = max(peak_memory, current_memory)
            
            if (iter_idx + 1) % 10 == 0:
                avg_time = np.mean(times[-10:])
                print(f"    Iter {iter_idx+1:3d}/{num_iterations}: "
                      f"Time={iter_time:.3f}s (avg={avg_time:.3f}s), "
                      f"Memory={current_memory:.2f}GB (peak={peak_memory:.2f}GB)")
        
        # Clean up
        del scene, gaussians
        torch.cuda.empty_cache()
        
        # Results
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = 1.0 / avg_time  # iterations per second
        
        results = {
            'batch_size': batch_size,
            'avg_time': avg_time,
            'std_time': std_time,
            'throughput': throughput,
            'peak_memory': peak_memory,
        }
        
        print(f"\n  RESULTS:")
        print(f"    Average Time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"    Throughput: {throughput:.2f} iter/s")
        print(f"    Peak Memory: {peak_memory:.2f}GB")
        
        return results
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test optimal batch size")
    parser.add_argument("--data", type=str, required=True, help="Path to CT case")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--num_iterations", type=int, default=50,
                        help="Number of iterations per test")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Error: Data path not found: {args.data}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Batch Size Testing Script")
    print(f"{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Data: {args.data}")
    
    results_list = []
    for batch_size in args.batch_sizes:
        results = test_batch_size(args.data, batch_size, args.num_iterations)
        if results:
            results_list.append(results)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Batch Size':<15} {'Avg Time (s)':<18} {'Throughput (it/s)':<20} {'Peak Mem (GB)':<15}")
    print(f"{'-'*68}")
    
    for r in results_list:
        print(f"{r['batch_size']:<15} {r['avg_time']:<18.4f} {r['throughput']:<20.2f} {r['peak_memory']:<15.2f}")
    
    # Find best batch size by throughput
    if results_list:
        best = max(results_list, key=lambda x: x['throughput'])
        print(f"\n✓ Best batch size by throughput: {best['batch_size']} "
              f"(throughput={best['throughput']:.2f} it/s, memory={best['peak_memory']:.2f}GB)")
    
    print(f"\nNote: Run with a case that fits in GPU memory. Recommended: batch_size <= 4 for 3090.")


if __name__ == "__main__":
    safe_state(False)
    main()
