"""Utilities for loading and caching compiled CoreML models."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Iterable

import coremltools as ct


def parse_coremldata_bin(
    mlmodelc_path: Path,
    probe_names: Iterable[bytes] | None = None,
) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Parse ``coremldata.bin`` to recover IO/state shapes for CompiledMLModel."""
    coremldata_path = mlmodelc_path / "coremldata.bin"
    if not coremldata_path.exists():
        return {"inputs": {}, "outputs": {}, "states": {}}

    data = coremldata_path.read_bytes()
    if probe_names is None:
        probe_names = (b"hidden_states", b"inputs_embeds", b"pixel_patches", b"input")

    def decode_varint(buffer: bytes, pos: int) -> tuple[int, int]:
        result = 0
        shift = 0
        while pos < len(buffer):
            byte = buffer[pos]
            result |= (byte & 0x7F) << shift
            pos += 1
            if not (byte & 0x80):
                break
            shift += 7
        return result, pos

    def decode_shape(buffer: bytes, start: int, length: int) -> list[int]:
        shape: list[int] = []
        pos = start
        end = start + length
        while pos < end:
            val, pos = decode_varint(buffer, pos)
            shape.append(val)
        return shape

    def parse_field(buffer: bytes, pos: int):
        if pos >= len(buffer):
            return None

        tag, pos = decode_varint(buffer, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07

        if wire_type == 0:
            value, pos = decode_varint(buffer, pos)
        elif wire_type == 2:
            length, pos = decode_varint(buffer, pos)
            value = buffer[pos : pos + length]
            pos += length
        elif wire_type == 1:
            value = buffer[pos : pos + 8]
            pos += 8
        elif wire_type == 5:
            value = buffer[pos : pos + 4]
            pos += 4
        else:
            return None

        return field_num, wire_type, value, pos

    proto_start = None
    for i in range(len(data) - 10):
        if data[i] == 0x0A and i + 2 < len(data) and data[i + 2] == 0x0A:
            chunk = data[i : i + 64]
            if any(probe_name in chunk for probe_name in probe_names):
                proto_start = i
                break

    if proto_start is None:
        proto_start = 0x53 if len(data) > 0x53 else 0

    proto_data = data[proto_start:]
    result = {"inputs": {}, "outputs": {}, "states": {}}

    pos = 0
    while pos < len(proto_data) - 5:
        field = parse_field(proto_data, pos)
        if field is None:
            break

        field_num, wire_type, value, new_pos = field
        pos = new_pos

        if field_num not in [1, 10, 13] or wire_type != 2:
            continue

        name = "unknown"
        shape: list[int] = []

        inner_pos = 0
        while inner_pos < len(value):
            inner = parse_field(value, inner_pos)
            if inner is None:
                break
            ifnum, iwtype, ival, inner_new_pos = inner
            inner_pos = inner_new_pos

            if ifnum == 1 and iwtype == 2:
                try:
                    name = ival.decode("utf-8")
                except Exception:
                    pass
            elif ifnum == 3 and iwtype == 2:
                type_pos = 0
                while type_pos < len(ival):
                    type_field = parse_field(ival, type_pos)
                    if type_field is None:
                        break
                    tfnum, twtype, tval, type_new_pos = type_field
                    type_pos = type_new_pos

                    if tfnum == 5 and twtype == 2:
                        arr_pos = 0
                        while arr_pos < len(tval):
                            arr_field = parse_field(tval, arr_pos)
                            if arr_field is None:
                                break
                            afnum, awtype, aval, arr_new_pos = arr_field
                            arr_pos = arr_new_pos
                            if afnum == 1 and awtype == 2:
                                shape = decode_shape(
                                    tval,
                                    arr_pos - len(aval),
                                    len(aval),
                                )

                    elif tfnum == 8 and twtype == 2:
                        state_pos = 0
                        while state_pos < len(tval):
                            state_field = parse_field(tval, state_pos)
                            if state_field is None:
                                break
                            sfnum, swtype, sval, state_new_pos = state_field
                            state_pos = state_new_pos
                            if sfnum == 1 and swtype == 2:
                                nested_pos = 0
                                while nested_pos < len(sval):
                                    nf = parse_field(sval, nested_pos)
                                    if nf is None:
                                        break
                                    nfnum, nwtype, nval, nested_new_pos = nf
                                    nested_pos = nested_new_pos
                                    if nfnum == 1 and nwtype == 2:
                                        shape = decode_shape(
                                            sval,
                                            nested_pos - len(nval),
                                            len(nval),
                                        )

        if field_num == 1:
            result["inputs"][name] = {"shape": shape}
        elif field_num == 10:
            result["outputs"][name] = {"shape": shape}
        elif field_num == 13:
            result["states"][name] = {"shape": shape}

    return result


def get_compiled_model_path(mlpackage_path: Path) -> Path:
    """Get the compiled ``.mlmodelc`` path corresponding to an ``.mlpackage``."""
    return mlpackage_path.with_suffix(".mlmodelc")


def compiled_model_exists(mlpackage_path: Path) -> bool:
    """Check if a cached compiled model exists for the given ``.mlpackage``."""
    compiled_path = get_compiled_model_path(mlpackage_path)
    return compiled_path.exists() and compiled_path.is_dir()


def cache_compiled_model(mlmodel: Any, mlpackage_path: Path, verbose: bool = True) -> Path:
    """Cache compiled model directory next to its ``.mlpackage``."""
    compiled_path = get_compiled_model_path(mlpackage_path)
    temp_compiled_path = mlmodel.get_compiled_model_path()

    if verbose:
        print(f"  Caching compiled model to: {compiled_path}")

    shutil.copytree(temp_compiled_path, str(compiled_path), dirs_exist_ok=True)
    return compiled_path


def extract_specs_from_mlmodel(mlmodel: Any) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Extract input/output/state shapes from an MLModel's spec."""
    spec = mlmodel.get_spec()
    specs = {"inputs": {}, "outputs": {}, "states": {}}

    for input_desc in spec.description.input:
        shape = list(input_desc.type.multiArrayType.shape)
        specs["inputs"][input_desc.name] = {"shape": shape}

    for output_desc in spec.description.output:
        shape = list(output_desc.type.multiArrayType.shape)
        specs["outputs"][output_desc.name] = {"shape": shape}

    for state_desc in spec.description.state:
        shape = list(state_desc.type.multiArrayType.shape)
        specs["states"][state_desc.name] = {"shape": shape}

    return specs


def load_model_with_cache(
    mlpackage_path: Path,
    cache_compiled: bool = False,
    verbose: bool = True,
    probe_names: Iterable[bytes] | None = None,
) -> tuple[Any, dict[str, dict[str, dict[str, list[int]]]]]:
    """Load CoreML model using cached ``.mlmodelc`` when available."""
    compiled_path = get_compiled_model_path(mlpackage_path)

    if compiled_model_exists(mlpackage_path):
        if verbose:
            print(f"  Found cached compiled model: {compiled_path}")
        try:
            model = ct.models.CompiledMLModel(str(compiled_path))
            specs = parse_coremldata_bin(compiled_path, probe_names=probe_names)
            return model, specs
        except Exception as exc:
            if verbose:
                print(
                    "  Cached compiled model failed to load; "
                    f"falling back to .mlpackage ({type(exc).__name__}: {exc})"
                )

    if verbose:
        print(f"  Loading from .mlpackage: {mlpackage_path}")
    mlmodel = ct.models.MLModel(str(mlpackage_path))
    specs = extract_specs_from_mlmodel(mlmodel)

    if cache_compiled:
        cache_compiled_model(mlmodel, mlpackage_path, verbose)

    return mlmodel, specs
