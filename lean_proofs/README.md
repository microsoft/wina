# Lean 4 formalization of WINA's mathematical guarantees

Machine-checked Lean 4 and Mathlib proofs for the mathematical statements in *WINA: Weight Informed Neuron Activation for Accelerating Large Language Model Inference*. The package covers the gating objectives and top-$K$ rules in Eqs. (1)–(4), Lemma 3.1 and Eq. (6), the base and inductive upper-bound route of Theorem 3.2, and the SVD Gram identity in Eq. (5).

The project builds without `sorry` or `admit`. Every declaration mapped by `catalog.json` is included in the generated axiom audit; `#print axioms` reports no axioms beyond `propext`, `Classical.choice`, and `Quot.sound`.

## Build

Install Lean through `elan`, then run in this directory:

```text
lake build
```

The Lean and Mathlib versions are pinned to v4.31.0. The catalog-driven axiom audit is reproducible with:

```text
python generate_catalog_axioms.py --module WINA --run
```

## Files

- `Basic.lean` defines binary gates as retained-coordinate sets, masking, the dense and gated objectives from Eqs. (1)–(2), and the literal magnitude/WINA scores from Eqs. (3)–(4). It proves that the squared algebraic WINA score used in later proofs has exactly the same top-$K$ selections as $|x_i|\|W_{:,i}\|_2$.
- `TopK.lean` proves the tie-safe finite exchange theorem: a size-$K$ set containing the largest scores maximizes the retained sum and minimizes the omitted sum. It also proves that a WINA top-$K$ gate exists whenever $K$ is feasible.
- `SingleLayer.lean` follows Appendix A.7 and proves Lemma 3.1. It rewrites the squared output error as a sum over omitted coordinates, uses the diagonal Gram matrix to cancel cross terms, and proves the paper's norm-valued optimality conclusion on the fixed-density gate domain.
- `SVDGram.lean` follows Appendix A.2 and proves $(WV)^\top(WV)=\Sigma^\top\Sigma$ from $W=U\Sigma V^\top$ and the orthogonality identities. It derives orthogonal columns from diagonal $\Sigma$ in the paper's square convention and states the dimensionally precise rectangular variant.
- `DeepUpperBound.lean` follows Appendix A.8. It proves the exact two-layer and $N+1$ error decompositions, proves the scaled vector Young inequality internally, derives the valid omitted-score identities under their exact Gram conditions, and proves the joint top-$K$ minimizer theorem from an explicit gate-independent separable-score representation. That conclusion is stated both for an abstract score family and directly in the paper's objects, where the bound is built from the column-norm-weighted scores $x_i^2\|A_{:,i}\|_2^2$ and the gates are the ones Eq. (4) selects, including its literal $|x_i|\|A_{:,i}\|_2$ form.
- `Examples.lean` supplies exact two-coordinate identity-network witnesses showing that the concrete two-layer repair in `DeepUpperBound.lean` is non-vacuous and proves its minimizer inequality against every admissible rival gate pair. It also instantiates the paper-notation joint-minimizer theorem on a single linear layer, where Appendix A.7's identity discharges the upper-bound premise with equality.
- `WINA.lean` is the umbrella import.
- `catalog.json` maps every paper item to its source/PDF location, Lean declarations, dependencies, and any additional hypothesis. `generate_catalog_axioms.py`, `CatalogAxioms.lean`, and `catalog-axiom-audit.txt` make the declaration-resolution and axiom audit reproducible from that map.

## Mathematical clarifications exposed by the formalization

Lemma 3.1 is valid on the fixed-density domain described by the surrounding text: all competing gates retain exactly $K$ coordinates. Without that restriction, the all-ones gate has zero error. The Lean theorem states this domain explicitly, treats ties set-valuedly, and proves the literal Euclidean-norm inequality rather than only a squared surrogate.

For Theorem 3.2, the exact decomposition and Young upper bound in Appendix A.8 hold as stated. The newest-layer residual has the printed omitted-score form when the newest weight matrix has diagonal Gram matrix. The earlier two-layer residual has the analogous form when the effective matrix $A=W^{(2)}W^{(1)}$ has diagonal Gram matrix.

The joint minimization step additionally needs the layer scores in the upper bound to be independent of all gate choices. `deep_upper_bound_optimality` makes this meaning of “separable” explicit: on the intended fixed-cardinality domains, an upper bound of the form

$$
\mathcal U(x;\mathcal G)=\sum_{\ell}\gamma_\ell\sum_{i:g_i^{(\ell)}=0}s_{\ell i}(x),
\qquad \gamma_\ell\ge0,
$$

is jointly minimized by layerwise top-$K$ gates when the scores $s_{\ell i}(x)$ are gate-independent. Appendix A.8's newest-layer scores contain the incoming gated activation and therefore depend on earlier gates; the Lean theorem exposes the gate-independent representation as an additional premise rather than deriving it from the per-layer diagonal-Gram assumptions. `deep_wina_upper_bound_optimality` then specializes that bound to $s_{\ell i}(x)=x_i^2\|A_{:,i}^{(\ell)}\|_2^2$ and the gates of Eq. (4), which is the theorem's own "minimizing $\mathcal U$ reduces to selecting, at each layer, the $k$ largest coordinates weighted by squared column norms".

For the particular two-layer Young objective, `twoLayerYoungBound_minimal_of_effectiveTopK_of_newestResidual_eq_zero` gives a concrete sufficient condition: the effective product has diagonal Gram matrix, the first gate is top-$K$ for the effective scores, and the second gate omits no nonzero incoming contribution. `Examples.lean` verifies these conditions simultaneously on an exact identity-network instance.
