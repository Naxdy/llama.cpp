#!/usr/bin/env python3
"""
Transfer indexer tensors and metadata from an MSA GGUF to a dense GGUF.

Usage:
  python transfer_indexer.py <dense.gguf> <msa.gguf> <output.gguf>
  python transfer_indexer.py <dense-00001-of-00005.gguf> <msa-00001-of-00006.gguf> <output.gguf> --split-max-size 20

This script:
1. Reads all metadata + tensors from the dense GGUF (all splits)
2. Reads indexer metadata + tensors from the MSA GGUF (all splits)
3. Writes a new merged GGUF with combined metadata and tensors

The output is a single merged file by default. Use --split-max-size to split
the output into multiple files of approximately the given size (in GB).
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# Add gguf-py to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gguf import GGUFReader, GGUFWriter, GGUFValueType


# Metadata keys to skip when copying from the reader (handled by the writer or split format)
SKIP_METADATA_KEYS = {
    'general.architecture',  # added by writer constructor
    'split.no',              # split metadata
    'split.count',           # split metadata
    'split.tensors.count',   # split metadata
}


def parse_split_path(path):
    """Parse a GGUF split file path and return (prefix, split_no, split_count)."""
    match = re.match(r'^(.+)-(\d{5})-of-(\d{5})\.gguf$', str(path))
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return str(path), 1, 1


def get_split_files(path):
    """Get all split file paths for a GGUF."""
    prefix, _, split_count = parse_split_path(path)
    if split_count == 1:
        return [str(path)]
    return [f"{prefix}-{i:05d}-of-{split_count:05d}.gguf" for i in range(1, split_count + 1)]


# MSA indexer tensor name suffixes (after "blk.%d.indexer.")
MSA_INDEXER_SUFFIXES = ('.q_proj', '.k_proj', '.q_norm', '.k_norm')
# Old-style MSA tensor names (used by Sciguy429's GGUF)
MSA_OLD_PREFIXES = ('index_q_proj', 'index_k_proj', 'index_q_norm', 'index_k_norm')


def is_indexer_tensor(name):
    """Check if a tensor name is an MSA indexer tensor (not a DSA compressor tensor)."""
    # Strip .weight/.bias suffix to get the base tensor name
    # (GGUF stores tensors as blk.N.indexer.q_proj.weight, etc.)
    base = name
    for suffix in ('.weight', '.bias'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    # Match new-style: blk.N.indexer.{q_proj,k_proj,q_norm,k_norm}
    if '.indexer.' in base:
        for suffix in MSA_INDEXER_SUFFIXES:
            if base.endswith(suffix):
                return True
        # Exclude DSA tensors: indexer.proj, indexer.attn_k, indexer.attn_q_b
        return False
    # Match old-style: blk.N.index_{q_proj,k_proj,q_norm,k_norm}
    for prefix in MSA_OLD_PREFIXES:
        if prefix in base:
            return True
    return False

def rename_indexer_tensor(name):
    """Rename old-style indexer tensor names to new-style (index_q_proj -> indexer.q_proj)."""
    name = name.replace('index_q_proj', 'indexer.q_proj')
    name = name.replace('index_k_proj', 'indexer.k_proj')
    name = name.replace('index_q_norm', 'indexer.q_norm')
    name = name.replace('index_k_norm', 'indexer.k_norm')
    return name


def get_field_value(field) -> Any:
    """Get the value of a metadata field from a GGUFReader."""
    vtype = field.types[0]
    if vtype == GGUFValueType.STRING:
        return str(field.parts[field.data[0]].tobytes(), encoding='utf-8')
    elif vtype == GGUFValueType.ARRAY:
        sub_type = field.types[-1]
        if sub_type == GGUFValueType.STRING:
            return [str(field.parts[idx].tobytes(), encoding='utf-8') for idx in field.data]
        else:
            return [field.parts[idx].tolist()[0] for idx in field.data]
    else:
        return field.parts[field.data[0]].tolist()[0]


def add_field_to_writer(writer, key, field):
    """Add a metadata field from a GGUFReader to a GGUFWriter."""
    if key in SKIP_METADATA_KEYS:
        return
    vtype = field.types[0]
    if vtype == GGUFValueType.STRING:
        writer.add_string(key, get_field_value(field))
    elif vtype == GGUFValueType.ARRAY:
        sub_type = field.types[-1]
        writer.add_key_value(key, get_field_value(field), vtype, sub_type=sub_type)
    elif vtype == GGUFValueType.UINT32:
        writer.add_uint32(key, int(get_field_value(field)))
    elif vtype == GGUFValueType.INT32:
        writer.add_int32(key, int(get_field_value(field)))
    elif vtype == GGUFValueType.FLOAT32:
        writer.add_float32(key, float(get_field_value(field)))
    elif vtype == GGUFValueType.UINT64:
        writer.add_uint64(key, int(get_field_value(field)))
    elif vtype == GGUFValueType.INT64:
        writer.add_int64(key, int(get_field_value(field)))
    elif vtype == GGUFValueType.BOOL:
        writer.add_bool(key, bool(get_field_value(field)))
    elif vtype == GGUFValueType.UINT8:
        writer.add_uint8(key, int(get_field_value(field)))
    elif vtype == GGUFValueType.INT8:
        writer.add_int8(key, int(get_field_value(field)))
    elif vtype == GGUFValueType.UINT16:
        writer.add_uint16(key, int(get_field_value(field)))
    elif vtype == GGUFValueType.INT16:
        writer.add_int16(key, int(get_field_value(field)))
    elif vtype == GGUFValueType.FLOAT64:
        writer.add_float64(key, float(get_field_value(field)))
    else:
        print(f"  WARNING: Skipping unsupported type {vtype} for key {key}")


def main():
    parser = argparse.ArgumentParser(
        description='Transfer indexer tensors from an MSA GGUF to a dense GGUF')
    parser.add_argument('dense', help='Path to the dense GGUF (split 0 or single file)')
    parser.add_argument('msa', help='Path to the MSA GGUF (split 0 or single file)')
    parser.add_argument('output', help='Path to the output GGUF file')
    parser.add_argument('--split-max-size', type=float, default=0,
                        help='Max split size in GB (0 = no split, default: 0)')
    args = parser.parse_args()

    # Read dense GGUF
    print(f"Reading dense GGUF: {args.dense}")
    dense_files = get_split_files(args.dense)
    dense_readers = [GGUFReader(f) for f in dense_files]
    dense_reader0 = dense_readers[0]

    # Read all tensors from dense
    print(f"  Reading tensors from {len(dense_readers)} split files...")
    dense_tensors = []
    for reader in dense_readers:
        dense_tensors.extend(reader.tensors)
    print(f"  Total tensors: {len(dense_tensors)}")

    # Read MSA GGUF
    print(f"Reading MSA GGUF: {args.msa}")
    msa_files = get_split_files(args.msa)
    msa_readers = [GGUFReader(f) for f in msa_files]
    msa_reader0 = msa_readers[0]

    # Read indexer tensors from MSA
    print(f"  Reading tensors from {len(msa_readers)} split files...")
    msa_tensors = []
    for reader in msa_readers:
        msa_tensors.extend(reader.tensors)
    indexer_tensors = [t for t in msa_tensors if is_indexer_tensor(t.name)]
    print(f"  Total tensors: {len(msa_tensors)}, indexer tensors: {len(indexer_tensors)}")

    if not indexer_tensors:
        print("ERROR: No indexer tensors found in MSA GGUF")
        sys.exit(1)

    # Check for duplicate indexer tensors in dense
    dense_indexer = [t for t in dense_tensors if is_indexer_tensor(t.name)]
    if dense_indexer:
        print(f"  WARNING: dense GGUF already has {len(dense_indexer)} indexer tensors")
        print(f"  Removing them from the output to avoid duplicates")
        dense_tensors = [t for t in dense_tensors if not is_indexer_tensor(t.name)]

    # Get the architecture from the dense GGUF
    arch_field = dense_reader0.fields.get('general.architecture')
    if arch_field is None:
        raise ValueError("No general.architecture found in dense GGUF")
    arch_name = str(arch_field.parts[arch_field.data[0]].tobytes(), encoding='utf-8')
    print(f"  Architecture: {arch_name}")

    # Verify MSA GGUF has the same architecture
    msa_arch_field = msa_reader0.fields.get('general.architecture')
    if msa_arch_field is not None:
        msa_arch_name = str(msa_arch_field.parts[msa_arch_field.data[0]].tobytes(), encoding='utf-8')
        if msa_arch_name != arch_name:
            raise ValueError(f"Architecture mismatch: dense={arch_name}, msa={msa_arch_name}")
        print(f"  MSA architecture: {msa_arch_name} (matches)")

    # Create the writer
    print(f"Creating output GGUF: {args.output}")
    split_max_size = int(args.split_max_size * 1024**3) if args.split_max_size > 0 else 0
    writer = GGUFWriter(args.output, arch_name, split_max_size=split_max_size)

    # Copy metadata from dense
    print("  Copying metadata from dense GGUF...")
    for key, field in dense_reader0.fields.items():
        add_field_to_writer(writer, key, field)

    # Add MSA-specific metadata from MSA GGUF (keys not present in dense)
    # This includes: attention.indexer.*, leading_dense_block_count, etc.
    print("  Adding MSA metadata from MSA GGUF...")
    msa_keys_added = 0
    for key, field in msa_reader0.fields.items():
        if key in SKIP_METADATA_KEYS:
            continue
        if key in dense_reader0.fields:
            continue  # dense GGUF already has this key, keep its value
        add_field_to_writer(writer, key, field)
        value = get_field_value(field)
        print(f"    {key} = {value}")
        msa_keys_added += 1
    print(f"  Added {msa_keys_added} MSA-specific metadata keys")
    # Add all tensors from dense
    print("  Adding tensors from dense GGUF...")
    for i, tensor in enumerate(dense_tensors):
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        if (i + 1) % 100 == 0:
            print(f"    Added {i + 1}/{len(dense_tensors)} tensors")

    # Add indexer tensors from MSA (with name renaming if needed)
    print("  Adding indexer tensors from MSA GGUF...")
    for i, tensor in enumerate(indexer_tensors):
        name = rename_indexer_tensor(tensor.name)
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type)
        if (i + 1) % 50 == 0:
            print(f"    Added {i + 1}/{len(indexer_tensors)} indexer tensors")

    # Write the file
    print("  Writing output file...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    total_tensors = len(dense_tensors) + len(indexer_tensors)
    print(f"Done! Output: {args.output}")
    print(f"  Total tensors: {total_tensors}")
    if args.split_max_size == 0:
        print(f"  Single file (no split). To split: gguf-split --split {args.output}")


if __name__ == '__main__':
    main()
