import os
import re
from coremltools.models.compute_plan import MLComputePlan


def analyze_compute_plan(mlmodel):
    """
    Analyzes the Compute Plan of a CoreML model and prints a table of operations
    and their selected/supported compute devices.
    """
    print("\n[Compute Device Analysis]")
    try:
        compiled_path = mlmodel.get_compiled_model_path()
        compute_plan = MLComputePlan.load_from_path(compiled_path)

        print(
            f"{'Operation':<20} | {'Identifier':<30} | {'Selected Device':<15} | {'Cost':<10} | {'Supported Devices'}"
        )
        print("-" * 120)
        prog = compute_plan.model_structure.program
        for f in prog.functions.values():
            for op in f.block.operations:
                op_name = str(op.operator_name)
                if op_name != "const":
                    dev = "Unk"
                    supp = ""
                    ident = ""
                    cost_val = ""

                    if len(op.outputs) > 0:
                        ident = op.outputs[0].name

                    try:
                        u = compute_plan.get_compute_device_usage_for_mlprogram_operation(
                            op
                        )
                        dev = u.preferred_compute_device.__class__.__name__.replace(
                            "ML", ""
                        ).replace("ComputeDevice", "")
                        supp = ",".join(
                            [
                                d.__class__.__name__.replace("ML", "")
                                .replace("ComputeDevice", "")
                                .replace("NeuralEngine", "NE")
                                for d in u.supported_compute_devices
                            ]
                        )

                        c = compute_plan.get_estimated_cost_for_mlprogram_operation(op)
                        if c is not None and hasattr(c, "weight"):
                            cost_val = f"{c.weight:.2e}"
                    except Exception:
                        pass

                    print(
                        f"{op_name:<20} | {ident:<30} | {dev:<15} | {cost_val:<10} | {supp}"
                    )
    except Exception as e:
        print(f"Compute Plan Analysis failed: {e}")


def parse_mil_args(args_str):
    """
    Parses a comma-separated argument string, respecting nested parentheses/brackets.
    Returns a list of split strings.
    """
    args = []
    current = []
    level = 0

    for char in args_str:
        if char in "([{":
            level += 1
            current.append(char)
        elif char in ")]}":
            level -= 1
            current.append(char)
        elif char == "," and level == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        args.append("".join(current).strip())

    return args


def inspect_mil_program(mlmodel):
    """
    Parses the compiled MIL structure to provide deep inspection of operations,
    input shapes, DTypes, and constant values.
    """
    try:
        compiled_path = mlmodel.get_compiled_model_path()
        # Find model.mil in the compiled package
        mil_path = os.path.join(compiled_path, "model.mil")

        # In case the directory structure varies (sometimes deeper in hierarchy)
        if not os.path.exists(mil_path):
            for root, _, files in os.walk(compiled_path):
                if "model.mil" in files:
                    mil_path = os.path.join(root, "model.mil")
                    break

        if not os.path.exists(mil_path):
            print(f"MIL file not found in {compiled_path}")
            return

        print(f"\n[Deep Inspection via MIL File ({mil_path})]")
        print("=" * 60)

        with open(mil_path, "r") as f:
            content = f.read()

        lines = content.split("\n")

        # Symbol table: name -> {shape, dtype, value (optional)}
        var_map = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # 1. Parse Function Inputs
            if line.startswith("func "):
                match_func = re.search(r"\((tensor<.*)\)", line)
                if match_func:
                    args_str = match_func.group(1)
                    args = args_str.split(", tensor")

                    for arg in args:
                        if not arg.strip().startswith("tensor"):
                            arg = "tensor" + arg

                        match_arg = re.search(
                            r"tensor<([^,]+),\s*\[(.*?)\]>\s+(\w+)", arg
                        )
                        if match_arg:
                            dtype = match_arg.group(1)
                            shape = f"[{match_arg.group(2)}]"
                            name = match_arg.group(3)
                            var_map[name] = {"shape": shape, "dtype": dtype}

            if " = " not in line:
                continue

            parts = line.split(" = ", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip()

            # 2. Parse LHS (Output Definition)
            match_lhs = re.search(r"tensor<([^,]+),\s*\[(.*?)\]>\s+(\S+)", lhs)
            if not match_lhs:
                match_lhs = re.search(r"tensor<([^,]+),\s*\[\]>\s+(\S+)", lhs)
                if match_lhs:
                    shape = "[]"
                    dtype = match_lhs.group(1)
                    out_name = match_lhs.group(2)
                else:
                    continue
            else:
                dtype = match_lhs.group(1)
                shape = f"[{match_lhs.group(2)}]"
                out_name = match_lhs.group(3)

            # Register output
            var_map[out_name] = {"shape": shape, "dtype": dtype}

            # 3. Parse RHS (Op or Const)
            op_name_match = re.match(r"^([a-zA-Z0-9_]+)", rhs)
            if not op_name_match:
                continue
            op_type = op_name_match.group(1)

            args_raw = rhs[len(op_type) :]  # Capture everything after op name

            if op_type == "const":
                val_str = "Blob"
                if "BLOBFILE" in args_raw:
                    val_str = "Weights"
                else:
                    val_match = re.search(r"val = tensor<.*?>\((.*?)\)", args_raw)
                    if val_match:
                        val_str = val_match.group(1)

                var_map[out_name]["value"] = val_str

            else:
                # Operation Display
                print(f"Operation: {op_type}")
                print(f"  Output: {out_name} {shape} ({dtype})")
                print("  Inputs:")

                # Robustly extract arguments inside the FIRST balanced parenthesis pair
                # args_raw includes the opening parenthesis of the op call
                start_idx = args_raw.find("(")
                end_idx = -1
                if start_idx != -1:
                    depth = 0
                    for i in range(start_idx, len(args_raw)):
                        if args_raw[i] == "(":
                            depth += 1
                        elif args_raw[i] == ")":
                            depth -= 1
                        if depth == 0:
                            end_idx = i
                            break

                content_args = ""
                if start_idx != -1 and end_idx != -1:
                    # Exclude the outer parentheses
                    content_args = args_raw[start_idx + 1 : end_idx]

                # Parse arguments by splitting on top-level commas
                pairs = parse_mil_args(content_args)

                for p in pairs:
                    if " = " in p:
                        items = p.split(" = ", 1)
                        if len(items) == 2:
                            k, v = items
                            k = k.strip()
                            v = v.strip()

                            # Check if value is a tuple (e.g., "(x, y)")
                            if v.startswith("(") and v.endswith(")"):
                                print(f"    - {k}:")
                                inner_content = v[1:-1]
                                inner_vals = parse_mil_args(inner_content)
                                for idx, iv in enumerate(inner_vals):
                                    iv = iv.strip()
                                    elem_info = iv
                                    if iv in var_map:
                                        info = var_map[iv]
                                        s = info["shape"]
                                        d = info["dtype"]
                                        if "value" in info:
                                            val = info["value"]
                                            if val == "Weights":
                                                elem_info = f"{iv} {s} ({d}) [Weights]"
                                            elif val == "Blob":
                                                elem_info = f"{iv} {s} ({d}) [Blob]"
                                            else:
                                                elem_info = f"{iv} {s} ({d}): {val}"
                                        else:
                                            elem_info = f"{iv} {s} ({d})"
                                    print(f"      - {elem_info}")
                                continue

                            info_str = v  # Default to just name

                            if v in var_map:
                                info = var_map[v]
                                s = info["shape"]
                                d = info["dtype"]

                                if "value" in info:
                                    val = info["value"]
                                    if val == "Weights":
                                        info_str = f"Weights {s} ({d})"
                                    elif val == "Blob":
                                        info_str = f"Blob {s} ({d})"
                                    else:
                                        info_str = f"{val} {s} ({d})"
                                else:
                                    info_str = f"{v} {s} ({d})"

                            print(f"    - {k}: {info_str}")
                print("-" * 40)
    except Exception as e:
        print(f"Deep MIL Inspection failed: {e}")
