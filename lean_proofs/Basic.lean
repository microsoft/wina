import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.IsDiag
import Mathlib.LinearAlgebra.Matrix.Orthogonal
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# WINA notation

This file formalizes the objects used in Problem / Eq. (1), Refined Problem / Eq. (2),
the magnitude gate in Eq. (3), and the weight-informed gate in Eq. (4). A gate is
represented by the finite set of retained coordinates; its indicator is the paper's
binary vector, and `mask` is the Hadamard product with that vector.
-/

open scoped BigOperators RealInnerProductSpace

namespace WINA

variable {ι κ : Type*}

/-- The squared Euclidean norm of a finite real coordinate vector. -/
def sqNorm [Fintype ι] (x : ι → ℝ) : ℝ := ∑ i, x i ^ 2

/-- The Euclidean norm corresponding to `sqNorm`. -/
noncomputable def euclideanNorm [Fintype ι] (x : ι → ℝ) : ℝ := Real.sqrt (sqNorm x)

/-- The paper's binary gate, represented by its retained-coordinate set. -/
def gateIndicator [DecidableEq ι] (kept : Finset ι) (i : ι) : ℝ :=
  if i ∈ kept then 1 else 0

/-- Coordinatewise masking by the paper's binary gate. -/
def mask [DecidableEq ι] (kept : Finset ι) (x : ι → ℝ) (i : ι) : ℝ :=
  gateIndicator kept i * x i

/-- The coordinates removed by a gate. -/
def omittedPart [DecidableEq ι] (kept : Finset ι) (x : ι → ℝ) : ι → ℝ :=
  x - mask kept x

/-- Dense output of the paper's single linear layer. -/
def linearOutput [Fintype ι] (W : Matrix κ ι ℝ) (x : ι → ℝ) : κ → ℝ :=
  W.mulVec x

/-- Gated output of the paper's single linear layer. -/
def gatedLinearOutput [Fintype ι] [DecidableEq ι]
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (kept : Finset ι) : κ → ℝ :=
  linearOutput W (mask kept x)

/-- Squared output deviation, used for both Eq. (1) and Eq. (2). -/
def outputDeviationSq [Fintype κ] (dense gated : κ → ℝ) : ℝ :=
  sqNorm (dense - gated)

/-- The single-layer optimization objective in Eq. (2), squared without changing its minimizers. -/
def singleLayerGatingObjective [Fintype ι] [Fintype κ] [DecidableEq ι]
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (kept : Finset ι) : ℝ :=
  outputDeviationSq (linearOutput W x) (gatedLinearOutput W x kept)

/-- The literal Euclidean objective displayed in Eq. (2). -/
noncomputable def singleLayerGatingObjectiveNorm
    [Fintype ι] [Fintype κ] [DecidableEq ι]
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (kept : Finset ι) : ℝ :=
  euclideanNorm (linearOutput W x - gatedLinearOutput W x kept)

/-- The deep-network optimization objective in Eq. (1), abstracted over a gated evaluator. -/
def deepGatingObjective {GateFamily : Type*} [Fintype κ]
    (denseOutput : κ → ℝ) (gatedOutput : GateFamily → κ → ℝ) (gates : GateFamily) : ℝ :=
  outputDeviationSq denseOutput (gatedOutput gates)

/-- The squared WINA score $x_i^2\|W_{:,i}\|_2^2$, order-equivalent to Eq. (4). -/
def winaScoreSq [Fintype κ] (W : Matrix κ ι ℝ) (x : ι → ℝ) (i : ι) : ℝ :=
  x i ^ 2 * ∑ r, W r i ^ 2

/-- The paper's literal WINA score $|x_i|\|W_{:,i}\|_2$. -/
noncomputable def winaScore [Fintype κ]
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (i : ι) : ℝ :=
  |x i| * Real.sqrt (∑ r, W r i ^ 2)

theorem winaScore_nonneg [Fintype κ] (W : Matrix κ ι ℝ) (x : ι → ℝ) (i : ι) :
    0 ≤ winaScore W x i :=
  mul_nonneg (abs_nonneg _) (Real.sqrt_nonneg _)

/-- Squaring the literal score gives the algebraic score used in the proofs. -/
theorem winaScore_sq_eq [Fintype κ] (W : Matrix κ ι ℝ) (x : ι → ℝ) (i : ι) :
    winaScore W x i ^ 2 = winaScoreSq W x i := by
  have hsum : 0 ≤ ∑ r, W r i ^ 2 := Finset.sum_nonneg fun _ _ => sq_nonneg _
  rw [winaScore, mul_pow, sq_abs, Real.sq_sqrt hsum]
  rfl

/-- A retained set consists of `k` coordinates with largest values of `score`.
This set-valued formulation handles ties without imposing an arbitrary tie-breaker. -/
def IsTopK [DecidableEq ι] (score : ι → ℝ) (k : ℕ) (kept : Finset ι) : Prop :=
  kept.card = k ∧ ∀ ⦃i j : ι⦄, i ∈ kept → j ∉ kept → score j ≤ score i

/-- The magnitude-only top-$K$ gate in Eq. (3). -/
def IsMagnitudeTopK [DecidableEq ι] (x : ι → ℝ) (k : ℕ) (kept : Finset ι) : Prop :=
  IsTopK (fun i => |x i|) k kept

/-- The WINA top-$K$ gate in Eq. (4), using the squared nonnegative score. -/
def IsWINATopK [Fintype κ] [DecidableEq ι]
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (k : ℕ) (kept : Finset ι) : Prop :=
  IsTopK (winaScoreSq W x) k kept

/-- The squared-score formalization is exactly order-equivalent to the literal Eq. (4) score. -/
theorem isTopK_winaScoreSq_iff_winaScore [Fintype κ] [DecidableEq ι]
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (k : ℕ) (kept : Finset ι) :
    IsWINATopK W x k kept ↔ IsTopK (winaScore W x) k kept := by
  constructor
  · rintro ⟨hcard, htop⟩
    refine ⟨hcard, ?_⟩
    intro i j hi hj
    have hsq := htop hi hj
    rw [← winaScore_sq_eq W x j, ← winaScore_sq_eq W x i] at hsq
    exact (sq_le_sq₀ (winaScore_nonneg W x j) (winaScore_nonneg W x i)).mp hsq
  · rintro ⟨hcard, htop⟩
    refine ⟨hcard, ?_⟩
    intro i j hi hj
    have hraw := htop hi hj
    rw [← winaScore_sq_eq W x j, ← winaScore_sq_eq W x i]
    exact (sq_le_sq₀ (winaScore_nonneg W x j) (winaScore_nonneg W x i)).mpr hraw

@[simp] theorem mask_apply_of_mem [DecidableEq ι] {kept : Finset ι} {x : ι → ℝ}
    {i : ι} (hi : i ∈ kept) : mask kept x i = x i := by
  simp [mask, gateIndicator, hi]

@[simp] theorem mask_apply_of_not_mem [DecidableEq ι] {kept : Finset ι} {x : ι → ℝ}
    {i : ι} (hi : i ∉ kept) : mask kept x i = 0 := by
  simp [mask, gateIndicator, hi]

@[simp] theorem omittedPart_apply_of_mem [DecidableEq ι] {kept : Finset ι} {x : ι → ℝ}
    {i : ι} (hi : i ∈ kept) : omittedPart kept x i = 0 := by
  simp [omittedPart, hi]

@[simp] theorem omittedPart_apply_of_not_mem [DecidableEq ι]
    {kept : Finset ι} {x : ι → ℝ} {i : ι} (hi : i ∉ kept) :
    omittedPart kept x i = x i := by
  simp [omittedPart, hi]

end WINA
