#!/usr/bin/env python3
"""
Convert .npy files to compressed .npz format to save storage space.
Usage: python convert_npy_to_npz.py <case_dir>
"""
import argparse
from pathlib import Path
import numpy as np
import os

def convert_case_to_npz(case_dir):
    """Convert volume_gt.npy and vol_pred.npy to compressed .npz."""
    case_dir = Path(case_dir)
    
    if not case_dir.is_dir():
        print(f"Error: {case_dir} is not a directory")
        return False
    
    case_name = case_dir.name
    vg_path = case_dir / 'volume_gt.npy'
    vp_path = case_dir / 'vol_pred.npy'
    
    # Output paths
    npz_path = case_dir / f'{case_name}_volumes.npz'
    vg_compressed_path = case_dir / 'volume_gt.npz'
    vp_compressed_path = case_dir / 'vol_pred.npz'
    
    results = {
        'case': case_name,
        'files_converted': [],
        'size_before': 0,
        'size_after': 0,
    }
    
    # Convert volume_gt.npy
    if vg_path.exists():
        print(f"Loading {vg_path.name}...")
        vg = np.load(vg_path)
        vg_original_size = vg_path.stat().st_size
        results['size_before'] += vg_original_size
        
        print(f"  Shape: {vg.shape}, dtype: {vg.dtype}")
        print(f"  Original size: {vg_original_size:,} bytes = {vg_original_size/1024/1024:.2f} MB")
        
        # Save as compressed .npz
        np.savez_compressed(vg_compressed_path, volume_gt=vg)
        vg_compressed_size = vg_compressed_path.stat().st_size
        results['size_after'] += vg_compressed_size
        results['files_converted'].append('volume_gt.npy')
        
        print(f"  Compressed to {vg_compressed_path.name}")
        print(f"  Compressed size: {vg_compressed_size:,} bytes = {vg_compressed_size/1024/1024:.2f} MB")
        print(f"  Compression ratio: {vg_compressed_size/vg_original_size*100:.1f}%")
        print(f"  Space saved: {vg_original_size - vg_compressed_size:,} bytes = {(vg_original_size - vg_compressed_size)/1024/1024:.2f} MB")
        print()
    
    # Convert vol_pred.npy
    if vp_path.exists():
        print(f"Loading {vp_path.name}...")
        vp = np.load(vp_path)
        vp_original_size = vp_path.stat().st_size
        results['size_before'] += vp_original_size
        
        print(f"  Shape: {vp.shape}, dtype: {vp.dtype}")
        print(f"  Original size: {vp_original_size:,} bytes = {vp_original_size/1024/1024:.2f} MB")
        
        # Save as compressed .npz
        np.savez_compressed(vp_compressed_path, vol_pred=vp)
        vp_compressed_size = vp_compressed_path.stat().st_size
        results['size_after'] += vp_compressed_size
        results['files_converted'].append('vol_pred.npy')
        
        print(f"  Compressed to {vp_compressed_path.name}")
        print(f"  Compressed size: {vp_compressed_size:,} bytes = {vp_compressed_size/1024/1024:.2f} MB")
        print(f"  Compression ratio: {vp_compressed_size/vp_original_size*100:.1f}%")
        print(f"  Space saved: {vp_original_size - vp_compressed_size:,} bytes = {(vp_original_size - vp_compressed_size)/1024/1024:.2f} MB")
        print()
    
    # Combined summary
    if results['size_before'] > 0:
        print("=" * 60)
        print(f"Summary for {case_name}:")
        print(f"  Files converted: {', '.join(results['files_converted'])}")
        print(f"  Total size before: {results['size_before']:,} bytes = {results['size_before']/1024/1024:.2f} MB")
        print(f"  Total size after: {results['size_after']:,} bytes = {results['size_after']/1024/1024:.2f} MB")
        print(f"  Total compression ratio: {results['size_after']/results['size_before']*100:.1f}%")
        print(f"  Total space saved: {results['size_before'] - results['size_after']:,} bytes = {(results['size_before'] - results['size_after'])/1024/1024:.2f} MB")
        print("=" * 60)
        return True
    else:
        print("No .npy files found to convert")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .npy files to compressed .npz format')
    parser.add_argument('case_dir', help='Path to case directory')
    args = parser.parse_args()
    
    convert_case_to_npz(args.case_dir)
