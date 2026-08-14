import SingleLayer

/-!
# WINA Theorem 3.2: exact route and separability bridge

This file follows Appendix A.8 in source order. It first formalizes the exact two-layer
gated-error decomposition, then proves the scaled vector Young inequality used for the
upper bound, and then derives the two omitted-score identities under precisely the
diagonal-Gram hypotheses needed for each effective matrix.

The final theorem makes the paper's word "separable" explicit: the bound is a sum of
gate-independent nonnegative omitted-coordinate scores, with one fixed-cardinality
domain per layer. Under that bridge, the paper's layerwise WINA choices jointly minimize
the bound. The bridge is an explicit additional hypothesis; it is not silently inferred
from the paper's per-layer diagonal-Gram assumptions.
-/

open scoped BigOperators

namespace WINA

variable {ι h o : Type*} [Fintype ι] [Fintype h] [Fintype o]
  [DecidableEq ι] [DecidableEq h]

/-- Dense output of the two-layer base case in Appendix A.8. -/
def twoLayerDense (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ) : o → ℝ :=
  W2.mulVec (W1.mulVec x)

/-- Gated output of the two-layer base case in Appendix A.8. -/
def twoLayerGated (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ)
    (kept1 : Finset ι) (kept2 : Finset h) : o → ℝ :=
  W2.mulVec (mask kept2 (W1.mulVec (mask kept1 x)))

/-- The newest-layer residual `v` in Appendix A.8. -/
def newestLayerResidual (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ)
    (kept1 : Finset ι) (kept2 : Finset h) : o → ℝ :=
  W2.mulVec (mask kept2 (W1.mulVec (mask kept1 x)) - W1.mulVec (mask kept1 x))

/-- The earlier-layer residual `u` in Appendix A.8. -/
def earlierLayerResidual (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ)
    (kept1 : Finset ι) : o → ℝ :=
  W2.mulVec (W1.mulVec (mask kept1 x - x))

omit [Fintype o] in
/-- Appendix A.8's exact identity `e_g^(2) = v + u`. -/
theorem twoLayer_error_decomposition
    (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ)
    (kept1 : Finset ι) (kept2 : Finset h) :
    twoLayerGated W1 W2 x kept1 kept2 - twoLayerDense W1 W2 x =
      newestLayerResidual W1 W2 x kept1 kept2 + earlierLayerResidual W1 W2 x kept1 := by
  simp only [twoLayerGated, twoLayerDense, newestLayerResidual, earlierLayerResidual,
    ← Matrix.mulVec_sub, ← Matrix.mulVec_add]
  apply congrArg W2.mulVec
  rw [Matrix.mulVec_sub]
  abel

omit [Fintype o] [Fintype ι] [DecidableEq ι] in
/-- The same exact decomposition at the paper's `N+1` induction step. Here `gatedPrev`
and `densePrev` are $y_g^{(N)}$ and $y^{(N)}$. -/
theorem layer_error_decomposition (W : Matrix o h ℝ) (densePrev gatedPrev : h → ℝ)
    (kept : Finset h) :
    W.mulVec (mask kept gatedPrev) - W.mulVec densePrev =
      W.mulVec (mask kept gatedPrev - gatedPrev) + W.mulVec (gatedPrev - densePrev) := by
  simp only [← Matrix.mulVec_sub, ← Matrix.mulVec_add]
  apply congrArg W.mulVec
  abel

/-- The exact squared-norm expansion used immediately after the decomposition. -/
theorem sqNorm_add (u v : o → ℝ) :
    sqNorm (u + v) = sqNorm u + sqNorm v + 2 * (u ⬝ᵥ v) := by
  have hcross : (∑ i, 2 * u i * v i) = 2 * ∑ i, u i * v i := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    ring
  simp only [sqNorm, dotProduct, Pi.add_apply, add_sq]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, hcross]
  ring

/-- Squared Euclidean norm is insensitive to the orientation of a difference. -/
theorem sqNorm_sub_comm (u v : o → ℝ) : sqNorm (u - v) = sqNorm (v - u) := by
  simp only [sqNorm, Pi.sub_apply]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Scalar Young inequality in exactly the scaled form used by Appendix A.8. -/
theorem scalar_young_scaled {α a b : ℝ} (hα : 0 < α) :
    2 * a * b ≤ α * a ^ 2 + α⁻¹ * b ^ 2 := by
  have hαne : α ≠ 0 := ne_of_gt hα
  have hsquare : 0 ≤ (α * a - b) ^ 2 := sq_nonneg (α * a - b)
  have hid : α * (α * a ^ 2 + α⁻¹ * b ^ 2 - 2 * a * b) = (α * a - b) ^ 2 := by
    field_simp [hαne]
    ring
  nlinarith

/-- Vector Young inequality, proved coordinatewise rather than left as an external premise. -/
theorem dotProduct_young_scaled (u v : o → ℝ) {α : ℝ} (hα : 0 < α) :
    2 * (u ⬝ᵥ v) ≤ α * sqNorm u + α⁻¹ * sqNorm v := by
  simp only [dotProduct, sqNorm]
  calc
    2 * ∑ i, u i * v i = ∑ i, 2 * u i * v i := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      ring
    _ ≤ ∑ i, (α * u i ^ 2 + α⁻¹ * v i ^ 2) :=
      Finset.sum_le_sum fun i _ => scalar_young_scaled hα
    _ = α * ∑ i, u i ^ 2 + α⁻¹ * ∑ i, v i ^ 2 := by
      rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]

/-- The concrete two-layer Young objective constructed in Appendix A.8. -/
noncomputable def twoLayerYoungBound (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ)
    (kept1 : Finset ι) (kept2 : Finset h) (α : ℝ) : ℝ :=
  (1 + α) * sqNorm (earlierLayerResidual W1 W2 x kept1) +
    (1 + α⁻¹) * sqNorm (newestLayerResidual W1 W2 x kept1 kept2)

/-- Appendix A.8's provable two-layer upper bound. -/
theorem twoLayer_error_le_youngBound
    (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ)
    (kept1 : Finset ι) (kept2 : Finset h) {α : ℝ} (hα : 0 < α) :
    sqNorm (twoLayerGated W1 W2 x kept1 kept2 - twoLayerDense W1 W2 x) ≤
      twoLayerYoungBound W1 W2 x kept1 kept2 α := by
  rw [twoLayer_error_decomposition, sqNorm_add]
  have hy := dotProduct_young_scaled
    (earlierLayerResidual W1 W2 x kept1)
    (newestLayerResidual W1 W2 x kept1 kept2) hα
  have hcomm :
      newestLayerResidual W1 W2 x kept1 kept2 ⬝ᵥ earlierLayerResidual W1 W2 x kept1 =
        earlierLayerResidual W1 W2 x kept1 ⬝ᵥ newestLayerResidual W1 W2 x kept1 kept2 := by
    exact dotProduct_comm _ _
  simp only [twoLayerYoungBound]
  nlinarith

/-- The Young upper-bound step at the paper's general `N+1` induction stage. -/
theorem layer_error_le_youngBound
    (W : Matrix o h ℝ) (densePrev gatedPrev : h → ℝ) (kept : Finset h)
    {α : ℝ} (hα : 0 < α) :
    sqNorm (W.mulVec (mask kept gatedPrev) - W.mulVec densePrev) ≤
      (1 + α) * sqNorm (W.mulVec (gatedPrev - densePrev)) +
        (1 + α⁻¹) * sqNorm (W.mulVec (mask kept gatedPrev - gatedPrev)) := by
  rw [layer_error_decomposition, sqNorm_add]
  have hy := dotProduct_young_scaled
    (W.mulVec (gatedPrev - densePrev))
    (W.mulVec (mask kept gatedPrev - gatedPrev)) hα
  have hcomm :
      W.mulVec (mask kept gatedPrev - gatedPrev) ⬝ᵥ W.mulVec (gatedPrev - densePrev) =
        W.mulVec (gatedPrev - densePrev) ⬝ᵥ W.mulVec (mask kept gatedPrev - gatedPrev) := by
    exact dotProduct_comm _ _
  nlinarith

/-- The newest-layer norm has the conditional omitted-score form claimed in Appendix A.8. -/
theorem newestLayerResidual_sq_eq_omittedScoreSum
    (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ)
    (kept1 : Finset ι) (kept2 : Finset h)
    (hGram2 : (W2.transpose * W2).IsDiag) :
    sqNorm (newestLayerResidual W1 W2 x kept1 kept2) =
      ∑ j ∈ Finset.univ \ kept2, winaScoreSq W2 (W1.mulVec (mask kept1 x)) j := by
  let y := W1.mulVec (mask kept1 x)
  have herr := singleLayer_error_eq_omittedScoreSum W2 y kept2 hGram2
  rw [singleLayerGatingObjective, outputDeviationSq, linearOutput, gatedLinearOutput,
    linearOutput] at herr
  calc
    sqNorm (newestLayerResidual W1 W2 x kept1 kept2) =
        sqNorm (W2.mulVec (mask kept2 y) - W2.mulVec y) := by
      rw [newestLayerResidual, ← Matrix.mulVec_sub]
    _ = sqNorm (W2.mulVec y - W2.mulVec (mask kept2 y)) :=
      sqNorm_sub_comm _ _
    _ = ∑ j ∈ Finset.univ \ kept2, winaScoreSq W2 y j := herr

omit [DecidableEq h] in
/-- The first-layer norm has the paper's omitted-score form when the effective product
`A = W2 W1` itself has diagonal Gram matrix. This localizes the missing hypothesis in
the corresponding Appendix A.8 line. -/
theorem earlierLayerResidual_sq_eq_omittedScoreSum
    (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ) (kept1 : Finset ι)
    (hEffective : ((W2 * W1).transpose * (W2 * W1)).IsDiag) :
    sqNorm (earlierLayerResidual W1 W2 x kept1) =
      ∑ i ∈ Finset.univ \ kept1, winaScoreSq (W2 * W1) x i := by
  have herr := singleLayer_error_eq_omittedScoreSum (W2 * W1) x kept1 hEffective
  rw [singleLayerGatingObjective, outputDeviationSq, linearOutput, gatedLinearOutput,
    linearOutput] at herr
  calc
    sqNorm (earlierLayerResidual W1 W2 x kept1) =
        sqNorm ((W2 * W1).mulVec (mask kept1 x) - (W2 * W1).mulVec x) := by
      rw [earlierLayerResidual, Matrix.mulVec_sub W1, Matrix.mulVec_sub W2,
        Matrix.mulVec_mulVec,
        Matrix.mulVec_mulVec]
    _ = sqNorm ((W2 * W1).mulVec x - (W2 * W1).mulVec (mask kept1 x)) :=
      sqNorm_sub_comm _ _
    _ = ∑ i ∈ Finset.univ \ kept1, winaScoreSq (W2 * W1) x i := herr

/-- A concrete source-route repair for the two-layer base case. If WINA's first gate
is also top-K for the effective matrix `W2 * W1`, and its second gate omits no nonzero
incoming contribution (so the newest residual vanishes), then WINA minimizes the
particular Young bound constructed in Appendix A.8. -/
theorem twoLayerYoungBound_minimal_of_effectiveTopK_of_newestResidual_eq_zero
    (W1 : Matrix h ι ℝ) (W2 : Matrix o h ℝ) (x : ι → ℝ)
    (k1 k2 : ℕ) (wina1 other1 : Finset ι) (wina2 other2 : Finset h)
    {α : ℝ} (hα : 0 < α)
    (hEffective : ((W2 * W1).transpose * (W2 * W1)).IsDiag)
    (hwina1 : IsWINATopK (W2 * W1) x k1 wina1)
    (_hwina2 : wina2.card = k2) (hother1 : other1.card = k1)
    (_hother2 : other2.card = k2)
    (hwinaNewest : newestLayerResidual W1 W2 x wina1 wina2 = 0) :
    twoLayerYoungBound W1 W2 x wina1 wina2 α ≤
      twoLayerYoungBound W1 W2 x other1 other2 α := by
  have hearlier := omitted_sum_minimal_of_isTopK hwina1 hother1
  rw [← earlierLayerResidual_sq_eq_omittedScoreSum W1 W2 x wina1 hEffective,
    ← earlierLayerResidual_sq_eq_omittedScoreSum W1 W2 x other1 hEffective] at hearlier
  have hnewest : 0 ≤ sqNorm (newestLayerResidual W1 W2 x other1 other2) := by
    exact Finset.sum_nonneg fun _ _ => sq_nonneg _
  have hαinv : 0 ≤ α⁻¹ := (inv_pos.mpr hα).le
  have hwinaZero : sqNorm (newestLayerResidual W1 W2 x wina1 wina2) = 0 := by
    rw [hwinaNewest]
    simp [sqNorm]
  have hfirst := mul_le_mul_of_nonneg_left hearlier (show 0 ≤ 1 + α by linarith)
  have hsecond :
      0 ≤ (1 + α⁻¹) * sqNorm (newestLayerResidual W1 W2 x other1 other2) :=
    mul_nonneg (by linarith) hnewest
  simp only [twoLayerYoungBound, hwinaZero, mul_zero, add_zero]
  linarith

section Separable

variable {Layer : Type*} [Fintype Layer] [DecidableEq Layer]
  {Coord : Layer → Type*} [∀ l, Fintype (Coord l)] [∀ l, DecidableEq (Coord l)]

/-- A family of retained-coordinate sets, one for each layer. -/
abbrev GateFamily (Coord : Layer → Type*) := ∀ l, Finset (Coord l)

/-- The explicit gate-independent meaning of the paper's separable upper bound. -/
def separableUpperBound (coeff : Layer → ℝ) (score : ∀ l, Coord l → ℝ)
    (gates : GateFamily Coord) : ℝ :=
  ∑ l, coeff l * ∑ i ∈ Finset.univ \ gates l, score l i

/-- A WINA gate family satisfies the top-K predicate independently at every layer. -/
def IsLayerwiseTopK (score : ∀ l, Coord l → ℝ) (k : Layer → ℕ)
    (gates : GateFamily Coord) : Prop :=
  ∀ l, IsTopK (score l) (k l) (gates l)

/-- The fixed-density joint gate domain intended by the paper's top-k rule. -/
def IsAdmissibleGateFamily (k : Layer → ℕ) (gates : GateFamily Coord) : Prop :=
  ∀ l, (gates l).card = k l

omit [DecidableEq Layer] in
/-- The layerwise top-K choices jointly minimize every gate-independent separable bound. -/
theorem separableUpperBound_minimal
    (coeff : Layer → ℝ) (score : ∀ l, Coord l → ℝ) (k : Layer → ℕ)
    (wina other : GateFamily Coord) (hcoeff : ∀ l, 0 ≤ coeff l)
    (hwina : IsLayerwiseTopK score k wina) (hother : ∀ l, (other l).card = k l) :
    separableUpperBound coeff score wina ≤ separableUpperBound coeff score other := by
  apply Finset.sum_le_sum
  intro l _
  exact mul_le_mul_of_nonneg_left
    (omitted_sum_minimal_of_isTopK (hwina l) (hother l)) (hcoeff l)

omit [DecidableEq Layer] in
/-- Paper-facing form of Theorem 3.2: once the bound constructed by the source route
has a gate-independent separable-score representation, WINA is in its joint argmin. -/
theorem deep_upper_bound_optimality
    (error : GateFamily Coord → ℝ) (coeff : Layer → ℝ)
    (score : ∀ l, Coord l → ℝ) (k : Layer → ℕ) (wina : GateFamily Coord)
    (hcoeff : ∀ l, 0 ≤ coeff l) (hwina : IsLayerwiseTopK score k wina)
    (hupper : ∀ gates, IsAdmissibleGateFamily k gates →
      error gates ≤ separableUpperBound coeff score gates) :
    (∀ gates, IsAdmissibleGateFamily k gates →
      error gates ≤ separableUpperBound coeff score gates) ∧
      ∀ other, IsAdmissibleGateFamily k other →
        separableUpperBound coeff score wina ≤ separableUpperBound coeff score other := by
  refine ⟨hupper, ?_⟩
  intro other hother
  exact separableUpperBound_minimal coeff score k wina other hcoeff hwina hother

end Separable

end WINA
