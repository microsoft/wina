import Basic

/-!
# Appendix A.2 / Eq. (5): SVD Gram bridge

The declarations use the paper's symbols `W`, `U`, `Sigma`, and `V`. Starting from
the stated SVD factorization and the two orthogonality identities, right multiplication
by `V` gives `U * Sigma`, whose Gram matrix is `Sigma.transpose * Sigma`. This is the
dimensionally precise rectangular version of the paper's `Sigma^2` notation.
-/

namespace WINA

variable {m n p : Type*} [Fintype m] [Fintype n] [Fintype p]
  [DecidableEq m] [DecidableEq n] [DecidableEq p]

omit [Fintype m] [DecidableEq m] [DecidableEq p] in
/-- The first substitution in Appendix A.2: `W V = U Sigma`. -/
theorem svd_right_transform_eq
    (W : Matrix m n ℝ) (U : Matrix m p ℝ) (Sigma : Matrix p n ℝ) (V : Matrix n n ℝ)
    (hW : W = U * Sigma * V.transpose) (hV : V.transpose * V = 1) :
    W * V = U * Sigma := by
  rw [hW, Matrix.mul_assoc, hV, Matrix.mul_one]

omit [DecidableEq m] in
/-- Eq. (5), with `Sigma.transpose * Sigma` in the rectangular case. -/
theorem svd_transformed_gram
    (W : Matrix m n ℝ) (U : Matrix m p ℝ) (Sigma : Matrix p n ℝ) (V : Matrix n n ℝ)
    (hW : W = U * Sigma * V.transpose)
    (hU : U.transpose * U = 1) (hV : V.transpose * V = 1) :
    (W * V).transpose * (W * V) = Sigma.transpose * Sigma := by
  rw [svd_right_transform_eq W U Sigma V hW hV, Matrix.transpose_mul]
  calc
    (Sigma.transpose * U.transpose) * (U * Sigma) =
        Sigma.transpose * (U.transpose * U) * Sigma := by
      simp only [Matrix.mul_assoc]
    _ = Sigma.transpose * Sigma := by rw [hU, Matrix.mul_one]

omit [DecidableEq m] in
/-- The paper's column-orthogonality conclusion from Eq. (5). -/
theorem svd_right_transform_hasOrthogonalCols
    (W : Matrix m n ℝ) (U : Matrix m p ℝ) (Sigma : Matrix p n ℝ) (V : Matrix n n ℝ)
    (hW : W = U * Sigma * V.transpose)
    (hU : U.transpose * U = 1) (hV : V.transpose * V = 1)
    (hSigma : (Sigma.transpose * Sigma).IsDiag) :
    (W * V).HasOrthogonalCols := by
  rw [← Matrix.transpose_mul_self_isDiag_iff_hasOrthogonalCols]
  rw [svd_transformed_gram W U Sigma V hW hU hV]
  exact hSigma

omit [DecidableEq n] in
/-- A square diagonal singular-value matrix has diagonal Gram matrix. -/
theorem isDiag_transpose_mul_self {Sigma : Matrix n n ℝ} (hSigma : Sigma.IsDiag) :
    (Sigma.transpose * Sigma).IsDiag := by
  classical
  intro i j hij
  simp only [Matrix.mul_apply, Matrix.transpose_apply]
  apply Finset.sum_eq_zero
  intro k _
  by_cases hki : k = i
  · subst k
    rw [hSigma hij]
    simp
  · rw [hSigma hki]
    simp

omit [DecidableEq m] in
/-- Eq. (5) in the paper's literal square-diagonal convention, deriving rather than
assuming the final diagonal-Gram fact. -/
theorem svd_right_transform_hasOrthogonalCols_square
    (W : Matrix m n ℝ) (U : Matrix m n ℝ) (Sigma V : Matrix n n ℝ)
    (hW : W = U * Sigma * V.transpose)
    (hU : U.transpose * U = 1) (hV : V.transpose * V = 1)
    (hSigma : Sigma.IsDiag) :
    (W * V).HasOrthogonalCols :=
  svd_right_transform_hasOrthogonalCols W U Sigma V hW hU hV
    (isDiag_transpose_mul_self hSigma)

end WINA
