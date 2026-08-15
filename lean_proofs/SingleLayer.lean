import Basic
import TopK

/-!
# WINA Lemma 3.1 / Eq. (6)

The proof follows Appendix A.7 in the paper's order. The output error is rewritten as
the image of the omitted coordinates; the diagonal Gram hypothesis removes every
cross term; and the finite top-K exchange theorem proves optimality. The theorem uses
the fixed-cardinality gate domain stated in the prose surrounding Lemma 3.1, which is
missing from the displayed `argmin`.
-/

open scoped BigOperators

namespace WINA

variable {ι κ : Type*} [Fintype ι] [Fintype κ] [DecidableEq ι]

omit [DecidableEq ι] in
/-- Under the paper's diagonal-Gram hypothesis, squared output norm is the sum of
the squared column-weighted coordinates. This is the cross-term cancellation step
in Appendix A.7. -/
theorem sqNorm_mulVec_eq_sum_winaScoreSq_of_gram_isDiag
    (W : Matrix κ ι ℝ) (z : ι → ℝ) (hGram : (W.transpose * W).IsDiag) :
    sqNorm (W.mulVec z) = ∑ i, winaScoreSq W z i := by
  classical
  simp only [sqNorm, pow_two, winaScoreSq]
  change W.mulVec z ⬝ᵥ W.mulVec z = _
  rw [Matrix.dotProduct_mulVec, ← Matrix.vecMul_transpose W z, Matrix.vecMul_vecMul]
  simp only [dotProduct, Matrix.vecMul]
  apply Finset.sum_congr rfl
  intro i _
  rw [Finset.sum_eq_single i]
  · simp only [Matrix.mul_apply, Matrix.transpose_apply]
    ring
  · intro j _ hji
    rw [hGram hji]
    simp
  · simp

/-- The squared single-layer error is exactly the sum of omitted WINA scores. -/
theorem singleLayer_error_eq_omittedScoreSum
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (kept : Finset ι)
    (hGram : (W.transpose * W).IsDiag) :
    singleLayerGatingObjective W x kept =
      ∑ i ∈ Finset.univ \ kept, winaScoreSq W x i := by
  classical
  rw [singleLayerGatingObjective, outputDeviationSq, linearOutput, gatedLinearOutput,
    linearOutput, ← Matrix.mulVec_sub]
  change sqNorm (W.mulVec (omittedPart kept x)) = _
  rw [sqNorm_mulVec_eq_sum_winaScoreSq_of_gram_isDiag W (omittedPart kept x) hGram]
  calc
    (∑ i, winaScoreSq W (omittedPart kept x) i) =
        ∑ i ∈ Finset.univ \ kept, winaScoreSq W (omittedPart kept x) i := by
      symm
      apply Finset.sum_subset (Finset.subset_univ _)
      intro i _ hi
      have hmem : i ∈ kept := by
        by_contra hnot
        exact hi (Finset.mem_sdiff.mpr ⟨Finset.mem_univ i, hnot⟩)
      simp [winaScoreSq, omittedPart, mask, gateIndicator, hmem]
    _ = ∑ i ∈ Finset.univ \ kept, winaScoreSq W x i := by
      apply Finset.sum_congr rfl
      intro i hi
      have hnot : i ∉ kept := (Finset.mem_sdiff.mp hi).2
      simp [winaScoreSq, omittedPart_apply_of_not_mem hnot]

/-- Lemma 3.1: every size-K WINA gate minimizes the fixed-density output error. -/
theorem singleLayer_topK_optimality
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (k : ℕ) (wina other : Finset ι)
    (hGram : (W.transpose * W).IsDiag)
    (hwina : IsWINATopK W x k wina) (hother : other.card = k) :
    singleLayerGatingObjective W x wina ≤ singleLayerGatingObjective W x other := by
  rw [singleLayer_error_eq_omittedScoreSum W x wina hGram,
    singleLayer_error_eq_omittedScoreSum W x other hGram]
  exact omitted_sum_minimal_of_isTopK hwina hother

/-- The literal norm-valued conclusion of Lemma 3.1. -/
theorem singleLayer_topK_optimality_norm
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (k : ℕ) (wina other : Finset ι)
    (hGram : (W.transpose * W).IsDiag)
    (hwina : IsWINATopK W x k wina) (hother : other.card = k) :
    singleLayerGatingObjectiveNorm W x wina ≤ singleLayerGatingObjectiveNorm W x other := by
  apply Real.sqrt_le_sqrt
  exact singleLayer_topK_optimality W x k wina other hGram hwina hother

omit [Fintype ι] [DecidableEq ι] in
/-- The paper's transpose-product formulation is exactly Mathlib's orthogonal-column
predicate, providing a notation bridge for Lemma 3.1. -/
theorem gram_isDiag_iff_hasOrthogonalCols (W : Matrix κ ι ℝ) :
    (W.transpose * W).IsDiag ↔ W.HasOrthogonalCols :=
  Matrix.transpose_mul_self_isDiag_iff_hasOrthogonalCols

end WINA
