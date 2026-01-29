"""Modified CoreMLTools graph passes for extended LayerNorm fusion.

This module provides modified versions of CoreMLTools graph passes that support
additional patterns not covered by the default passes.

WHY THIS IS NEEDED:
-------------------
CoreMLTools' default `fuse_layernorm_or_instancenorm` pass only recognizes LayerNorm
patterns when normalization happens over axis=1 (pattern 5: BC1S format). However,
in transformer models with QK-norm (e.g., Qwen3), we need to normalize query states
over axis=2 (the head_dim axis in BCHW format: batch, heads, head_dim, seq).

Without this extension, the normalization operations remain unfused:
    reduce_mean -> sub -> mul -> reduce_mean -> add -> rsqrt -> mul -> mul(gamma)

With this extension, they fuse into a single optimized operation:
    layer_norm (with gamma)

MODIFICATIONS FROM ORIGINAL:
----------------------------
1. `_try_match_and_transform_pattern_5`: Extended to accept ANY single axis,
   not just axis=1. Also validates that both reduce_mean ops use the same axis.

2. `_try_apply_transform`: Extended to handle gamma-only patterns (RMSNorm style)
   where there's no beta term, just gamma scaling after normalization.
   Original required both gamma AND beta for single-axis patterns.

3. Added support for detecting mul(gamma) directly after normalization,
   not just the add(beta) -> mul(gamma) pattern.

DESIGN PRINCIPLE:
-----------------
By supporting any normalization axis, we avoid the need for transpose/permute
operations to move the target axis to position 1 before normalization. This keeps
the computation graph cleaner and avoids potential graph cuts on the Neural Engine.

Usage:
    from coremlmodels.graph_passes import register_extended_passes

    # Call before ct.convert() to override default passes
    register_extended_passes()
"""

from typing import List

import numpy as np

from coremltools.converters.mil.mil import Block
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Var
from coremltools.converters.mil.mil.passes.defs.optimize_normalization import (
    fuse_layernorm_or_instancenorm as _OriginalFuseLayernormOrInstancenorm,
)
from coremltools.converters.mil.mil.passes.helper import _check_no_output_connection
from coremltools.converters.mil.mil.passes.pass_registry import register_pass


@register_pass(namespace="common", override=True)
class fuse_layernorm_or_instancenorm(_OriginalFuseLayernormOrInstancenorm):
    """Extended LayerNorm/InstanceNorm fusion pass supporting any single axis.

    Inherits from CoreMLTools' original pass and overrides two methods:
    - _try_apply_transform: Handle gamma-only patterns (no beta) for RMSNorm
    - _try_match_and_transform_pattern_5: Accept any single axis, not just axis=1

    All other patterns (1-4) and helper methods are inherited unchanged.

    Example use cases enabled by this extension:
    - QK-norm in attention: normalize query states over head_dim (axis=2 in BCHW)
    - RMSNorm with gamma but no beta (standard RMSNorm pattern)
    """

    @staticmethod
    def _try_apply_transform(
        reduce_op: Operation,
        block: Block,
        gamma_var: Var,
        beta_var: Var,
        epsilon_var: Var,
        end_op: Operation,
        ops_to_remove: List[Operation],
    ) -> bool:
        """Insert instance_norm / layer_norm and delete all ops.

        Extended to support any single axis for layer_norm (not just axis=1/-3).
        """
        if not _check_no_output_connection(block, ops_to_remove):
            return False

        axes = reduce_op.axes.val
        rank = len(reduce_op.x.shape)

        is_layernorm = False
        is_instancenorm = False
        is_require_rank4_transpose = False

        negative_axes = [a - rank if a >= 0 else a for a in axes]
        negative_axes.sort()

        gamma_rank = gamma_var.rank if gamma_var is not None else -1
        beta_rank = beta_var.rank if beta_var is not None else -1

        if gamma_rank == len(axes) and beta_rank == len(axes):
            if negative_axes == list(range(-len(negative_axes), 0)):
                is_layernorm = True

        # =======================================================================
        # MODIFICATION 1: Support any single axis for layer_norm
        # =======================================================================
        # Original: Only axis=1 (or -3 for rank 4) was supported
        # Extended: Any single axis works, enabling QK-norm (axis=2) fusion
        #
        # Also handle gamma-only patterns (beta is None) for RMSNorm:
        # Original: Required both gamma AND beta (gamma_rank == 1 and beta_rank == 1)
        # Extended: Allow gamma without beta (common in RMSNorm implementations)
        if len(negative_axes) == 1:
            if (gamma_var is None and beta_var is None):
                # No gamma or beta - plain normalization
                is_layernorm = True
            elif gamma_rank == 1 and (beta_var is None or beta_rank == 1):
                # Gamma only (RMSNorm style) OR both gamma and beta
                is_layernorm = True

            if gamma_var is not None and gamma_var.val is not None:
                ops_to_remove.append(gamma_var.op)
                gamma_var = gamma_var.val
            else:
                gamma_var = None

            if beta_var is not None and beta_var.val is not None:
                ops_to_remove.append(beta_var.op)
                beta_var = beta_var.val
            else:
                beta_var = None

        if rank == 4 and (negative_axes == [-2, -1] or negative_axes == [-3, -2]):
            if (
                gamma_var is not None
                and beta_var is not None
                and len(np.squeeze(gamma_var).shape) == 1
                and len(np.squeeze(beta_var).shape) == 1
            ):
                is_instancenorm = True
            if negative_axes == [-3, -2]:
                is_require_rank4_transpose = True

        if not (is_instancenorm or is_layernorm):
            return False

        out_name = end_op.outputs[0].name

        if is_require_rank4_transpose:
            x = mb.transpose(
                x=reduce_op.x,
                perm=[0, 3, 1, 2],
                name=out_name + "_transpose_nhwc_nchw",
                before_op=end_op,
            )
        if is_instancenorm:
            x = mb.instance_norm(
                x=x if is_require_rank4_transpose else reduce_op.x,
                gamma=np.squeeze(gamma_var),
                beta=np.squeeze(beta_var),
                epsilon=epsilon_var,
                name=out_name + "_instancenorm"
                if is_require_rank4_transpose
                else out_name,
                before_op=end_op,
            )
            ops_to_remove.extend([gamma_var.op, beta_var.op])
        else:
            x = mb.layer_norm(
                x=x if is_require_rank4_transpose else reduce_op.x,
                axes=axes,
                gamma=gamma_var,
                beta=beta_var,
                epsilon=epsilon_var,
                name=out_name + "_layernorm"
                if is_require_rank4_transpose
                else out_name,
                before_op=end_op,
            )
        if is_require_rank4_transpose:
            x = mb.transpose(
                x=x,
                perm=[0, 2, 3, 1],
                name=out_name + "_transpose_nchw_nhwc",
                before_op=end_op,
            )

        end_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=end_op, old_var=end_op.outputs[0], new_var=x
        )
        block.remove_ops(ops_to_remove)
        return True

    def _try_match_and_transform_pattern_5(self, reduce_op, block) -> bool:
        """Pattern 5: BC1S LayerNorm - EXTENDED to support any single axis.

        =======================================================================
        MODIFICATION 2: Accept any single normalization axis
        =======================================================================
        Original: Only matched when axes=[1] (the "C" in BC1S format)
        Extended: Matches any single axis, enabling:
          - axis=1: Standard RMSNorm (channels in BCHW)
          - axis=2: QK-norm (head_dim in B,heads,head_dim,seq)
          - axis=-1: Last dimension normalization

        Pattern matched (same as original):
            reduce_mean -> sub -> mul(square) -> reduce_mean -> add(eps) -> rsqrt -> mul

        Also extended to support gamma-only patterns (RMSNorm style) where there's
        no beta term, just gamma scaling after normalization:
          - Original: add(beta) -> mul(gamma)
          - Extended: Also matches mul(gamma) directly (no beta)
        """
        ops_to_remove = []
        root_var = reduce_op.x

        if len(list(root_var.child_ops)) < 2:
            return False

        if not self._check_reduce_op(reduce_op):
            return False
        # MODIFIED: Original checked `reduce_op.axes.val != [1]` here
        # We only require a single axis (any value) with keep_dims=True
        if len(reduce_op.axes.val) != 1 or not reduce_op.keep_dims.val:
            return False
        target_axis = reduce_op.axes.val[0]  # Store to verify second reduce_mean matches
        ops_to_remove.append(reduce_op)

        if not self._check_child_op_types(reduce_op, ["sub"], check_order=False):
            return False
        child_ops_reduce_mean = list(reduce_op.outputs[0].child_ops)
        sub_op1 = child_ops_reduce_mean[0]

        if sub_op1 is None or not self._check_child_op_types(
            sub_op1, child_op_types=["mul", "mul", "mul"]
        ):
            return False
        if not (sub_op1.x == root_var and sub_op1.y == reduce_op.outputs[0]):
            return False
        ops_to_remove.append(sub_op1)

        square_op = self._try_get_child_op_type(sub_op1, "mul")
        if square_op is None or not self._check_child_op_types(
            square_op, child_op_types=["reduce_mean"]
        ):
            return False
        if square_op.x != square_op.y:
            return False
        ops_to_remove.append(square_op)

        reduce_op2 = self._try_get_child_op_type(square_op, "reduce_mean")
        if not self._check_reduce_op(reduce_op2) or not self._check_child_op_types(
            reduce_op2, child_op_types=["add"]
        ):
            return False
        # MODIFIED: Check that reduce_op2 uses the same axis as reduce_op
        if (
            len(reduce_op2.axes.val) != 1
            or reduce_op2.axes.val[0] != target_axis
            or not reduce_op2.keep_dims.val
        ):
            return False
        ops_to_remove.append(reduce_op2)

        add_op1 = self._try_get_child_op_type(reduce_op2, "add")
        if add_op1 is None or not self._check_child_op_types(
            add_op1, child_op_types=["rsqrt"]
        ):
            return False
        epsilon_var = add_op1.y if add_op1.x == reduce_op2.outputs[0] else add_op1.x
        if epsilon_var.val is None or len(epsilon_var.val.shape) != 0:
            return False
        ops_to_remove.append(add_op1)

        rsqrt_op = self._try_get_child_op_type(add_op1, "rsqrt")
        if rsqrt_op is None or not self._check_child_op_types(
            rsqrt_op, child_op_types=["mul"]
        ):
            return False
        ops_to_remove.append(rsqrt_op)

        mul_op = self._try_get_child_op_type(rsqrt_op, "mul")
        if mul_op is None:
            return False
        if mul_op.y != sub_op1.outputs[0] and mul_op.x != sub_op1.outputs[0]:
            return False
        ops_to_remove.append(mul_op)

        end_op = mul_op
        gamma_var = None
        beta_var = None

        # =======================================================================
        # MODIFICATION 3: Support gamma-only patterns (RMSNorm style)
        # =======================================================================
        # Original only matched: add(beta) -> mul(gamma)
        # Extended to also match: mul(gamma) directly after normalization
        #
        # Pattern detection order:
        # 1. Check for add(beta) -> mul(gamma): both beta and gamma (original)
        # 2. Check for mul(gamma) directly: gamma only, no beta (NEW)
        # 3. Neither: no gamma/beta, plain normalization

        add_beta_op = self._try_get_child_op_type(mul_op, "add")
        mul_gamma_direct = self._try_get_child_op_type(mul_op, "mul")

        if add_beta_op is not None:
            # Pattern: add(beta) -> mul(gamma)
            mul_gamma_op = self._try_get_child_op_type(add_beta_op, "mul")

            if mul_gamma_op is not None:
                # Both beta and gamma
                if not self._check_child_op_types(mul_op, child_op_types=["add"]):
                    return False
                if not self._check_child_op_types(add_beta_op, child_op_types=["mul"]):
                    return False

                beta_var = (
                    add_beta_op.y
                    if add_beta_op.x == mul_op.outputs[0]
                    else add_beta_op.x
                )
                gamma_var = (
                    mul_gamma_op.y
                    if mul_gamma_op.x == add_beta_op.outputs[0]
                    else mul_gamma_op.x
                )

                if beta_var.val is None or gamma_var.val is None:
                    return False

                gamma_var = mb.const(
                    val=np.squeeze(gamma_var.val),
                    name="_fuse_layernorm_gamma",
                )

                beta_var = mb.const(
                    val=np.squeeze(beta_var.val) * gamma_var.val,
                    name="_fuse_layernorm_beta",
                )

                ops_to_remove.extend([add_beta_op, mul_gamma_op])
                end_op = mul_gamma_op

        elif mul_gamma_direct is not None:
            # Pattern: mul(gamma) directly after normalization (gamma only, no beta)
            # This is the RMSNorm pattern
            gamma_candidate = (
                mul_gamma_direct.y
                if mul_gamma_direct.x == mul_op.outputs[0]
                else mul_gamma_direct.x
            )

            # Verify it's a const (weight) and not another computed value
            if gamma_candidate.val is not None:
                gamma_var = mb.const(
                    val=np.squeeze(gamma_candidate.val),
                    name="_fuse_layernorm_gamma",
                )
                beta_var = None

                ops_to_remove.append(mul_gamma_direct)
                end_op = mul_gamma_direct

        return self._try_apply_transform(
            reduce_op, block, gamma_var, beta_var, epsilon_var, end_op, ops_to_remove
        )


def register_extended_passes():
    """Register the extended graph passes to override defaults.

    This function should be called before ct.convert() to ensure the extended
    passes are used instead of the default CoreMLTools passes.

    The extended passes support:
    - LayerNorm fusion for any single axis (not just axis=1)
    - QK-norm patterns in attention (axis=2 for query states)

    Example:
        from coremlmodels.graph_passes import register_extended_passes

        # Register before conversion
        register_extended_passes()

        # Now convert your model
        mlmodel = ct.convert(traced_model, ...)
    """
    # The @register_pass decorator already registered our pass when this module
    # was imported. Since we use the same name "fuse_layernorm_or_instancenorm"
    # in the "common" namespace, it overwrites the default pass.
    pass
