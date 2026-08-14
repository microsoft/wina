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

end WINA
