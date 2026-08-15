import DeepUpperBound

/-!
# Non-vacuity witnesses

These small exact instances certify that the repaired hypotheses used by the paper-facing
theorems can hold simultaneously. They are regression witnesses, not empirical examples.
-/

namespace WINA

/-- On the two-coordinate identity network, retaining coordinate `0` at both layers
satisfies the concrete two-layer repair for input `(2,1)` and density one. -/
theorem twoLayer_repair_nonvacuous :
    let x : Fin 2 → ℝ := ![2, 1]
    let kept : Finset (Fin 2) := {0}
    (((1 : Matrix (Fin 2) (Fin 2) ℝ) * (1 : Matrix (Fin 2) (Fin 2) ℝ)).transpose *
        ((1 : Matrix (Fin 2) (Fin 2) ℝ) * (1 : Matrix (Fin 2) (Fin 2) ℝ))).IsDiag ∧
      IsWINATopK ((1 : Matrix (Fin 2) (Fin 2) ℝ) * (1 : Matrix (Fin 2) (Fin 2) ℝ)) x 1 kept ∧
      newestLayerResidual (1 : Matrix (Fin 2) (Fin 2) ℝ)
        (1 : Matrix (Fin 2) (Fin 2) ℝ) x kept kept = 0 := by
  dsimp
  constructor
  · simp
  constructor
  · constructor
    · decide
    intro i j hi hj
    fin_cases i <;> fin_cases j <;>
      simp_all [winaScoreSq, Matrix.one_apply]
  · funext i
    fin_cases i <;>
      simp [newestLayerResidual, mask, gateIndicator, Matrix.one_mulVec]

/-- The complete concrete minimizer theorem is inhabited on the same exact instance. -/
theorem identity_twoLayerYoungBound_minimal
    (other1 other2 : Finset (Fin 2)) (hother1 : other1.card = 1)
    (hother2 : other2.card = 1) :
    let x : Fin 2 → ℝ := ![2, 1]
    let kept : Finset (Fin 2) := {0}
    twoLayerYoungBound (1 : Matrix (Fin 2) (Fin 2) ℝ)
        (1 : Matrix (Fin 2) (Fin 2) ℝ) x kept kept 1 ≤
      twoLayerYoungBound (1 : Matrix (Fin 2) (Fin 2) ℝ)
        (1 : Matrix (Fin 2) (Fin 2) ℝ) x other1 other2 1 := by
  dsimp
  have hrepair := twoLayer_repair_nonvacuous
  exact twoLayerYoungBound_minimal_of_effectiveTopK_of_newestResidual_eq_zero
    (1 : Matrix (Fin 2) (Fin 2) ℝ) (1 : Matrix (Fin 2) (Fin 2) ℝ) ![2, 1]
    1 1 {0} other1 {0} other2 (by norm_num) hrepair.1 hrepair.2.1
    (by decide) hother1 hother2 hrepair.2.2

/-- The deep separable-bound theorem is non-vacuous on a genuine network rather than only on
an abstract score family: for one linear layer with the paper's diagonal Gram hypothesis, the
Appendix A.7 identity discharges the bound hypothesis with equality, so the Eq. (4) gate is in
the argmin. -/
theorem deep_wina_optimality_nonvacuous_singleLayer
    {κ ι : Type*} [Fintype κ] [Fintype ι] [DecidableEq ι]
    (W : Matrix κ ι ℝ) (x : ι → ℝ) (k : ℕ) (wina : Finset ι)
    (hGram : (W.transpose * W).IsDiag) (hwina : IsWINATopK W x k wina) :
    (∀ gates : GateFamily (fun _ : Fin 1 => ι),
        IsAdmissibleGateFamily (fun _ => k) gates →
        singleLayerGatingObjective W x (gates 0) ≤
          winaSeparableUpperBound (Layer := Fin 1) (Coord := fun _ => ι) (Out := fun _ => κ)
            (fun _ => (1 : ℝ)) (fun _ => W) (fun _ => x) gates) ∧
      ∀ other : GateFamily (fun _ : Fin 1 => ι),
        IsAdmissibleGateFamily (fun _ => k) other →
        winaSeparableUpperBound (Layer := Fin 1) (Coord := fun _ => ι) (Out := fun _ => κ)
            (fun _ => (1 : ℝ)) (fun _ => W) (fun _ => x) (fun _ => wina) ≤
          winaSeparableUpperBound (Layer := Fin 1) (Coord := fun _ => ι) (Out := fun _ => κ)
            (fun _ => (1 : ℝ)) (fun _ => W) (fun _ => x) other := by
  have hupper : ∀ gates : GateFamily (fun _ : Fin 1 => ι),
      IsAdmissibleGateFamily (fun _ => k) gates →
      singleLayerGatingObjective W x (gates 0) ≤
        winaSeparableUpperBound (Layer := Fin 1) (Coord := fun _ => ι) (Out := fun _ => κ)
          (fun _ => (1 : ℝ)) (fun _ => W) (fun _ => x) gates := by
    intro gates _
    rw [winaSeparableUpperBound, separableUpperBound,
      singleLayer_error_eq_omittedScoreSum W x (gates 0) hGram]
    simp
  exact deep_wina_upper_bound_optimality (Layer := Fin 1) (Coord := fun _ => ι)
    (Out := fun _ => κ) (error := fun gates => singleLayerGatingObjective W x (gates 0))
    (coeff := fun _ => (1 : ℝ)) (A := fun _ => W) (act := fun _ => x) (k := fun _ => k)
    (wina := fun _ => wina) (hcoeff := fun _ => zero_le_one) (hwina := fun _ => hwina)
    (hupper := hupper)

end WINA
