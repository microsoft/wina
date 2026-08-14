import Basic

/-!
# Finite top-K exchange argument

This is the combinatorial optimization step used in WINA Lemma 3.1 and Theorem 3.2:
among fixed-cardinality retained sets, a set containing only scores at least as large
as every omitted score maximizes the retained sum and minimizes the omitted sum.
-/

open scoped BigOperators

namespace WINA

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [Fintype ι] in
/-- Equal-cardinality finite sets have equally large asymmetric differences. -/
theorem card_sdiff_eq_card_sdiff_of_card_eq {s t : Finset ι} (hcard : s.card = t.card) :
    (s \ t).card = (t \ s).card := by
  rw [Finset.card_sdiff, Finset.card_sdiff, Finset.inter_comm s t]
  omega

omit [Fintype ι] in
/-- A top-K set maximizes the retained score sum among sets of the same cardinality. -/
theorem sum_le_sum_of_isTopK {score : ι → ℝ} {k : ℕ} {best other : Finset ι}
    (hbest : IsTopK score k best) (hother : other.card = k) :
    ∑ i ∈ other, score i ≤ ∑ i ∈ best, score i := by
  classical
  have hcard : (other \ best).card = (best \ other).card :=
    card_sdiff_eq_card_sdiff_of_card_eq (hother.trans hbest.1.symm)
  let e : {i // i ∈ other \ best} ≃ {i // i ∈ best \ other} :=
    Fintype.equivOfCardEq (by simpa only [Fintype.card_coe] using hcard)
  have hdiff : (∑ i ∈ other \ best, score i) ≤ ∑ i ∈ best \ other, score i := by
    calc
      (∑ i ∈ other \ best, score i) = ∑ i : {i // i ∈ other \ best}, score i := by
        rw [← Finset.sum_attach]
        simp only [Finset.attach_eq_univ]
      _ ≤ ∑ i : {i // i ∈ other \ best}, score (e i) := by
        apply Finset.sum_le_sum
        intro i _
        exact hbest.2 (Finset.mem_sdiff.mp (e i).property).1
          (Finset.mem_sdiff.mp i.property).2
      _ = ∑ i : {i // i ∈ best \ other}, score i := by
        exact Equiv.sum_comp e (fun i : {i // i ∈ best \ other} => score i)
      _ = ∑ i ∈ best \ other, score i := by
        simpa only [← Finset.attach_eq_univ] using
          (Finset.sum_attach (best \ other) score)
  rw [← Finset.sum_inter_add_sum_sdiff other best,
    Finset.inter_comm other best,
    ← Finset.sum_inter_add_sum_sdiff best other]
  simpa only [add_comm] using add_le_add_left hdiff (∑ i ∈ best ∩ other, score i)

/-- A top-K set minimizes the score sum over omitted coordinates. -/
theorem omitted_sum_minimal_of_isTopK {score : ι → ℝ} {k : ℕ}
    {best other : Finset ι} (hbest : IsTopK score k best) (hother : other.card = k) :
    ∑ i ∈ Finset.univ \ best, score i ≤ ∑ i ∈ Finset.univ \ other, score i := by
  have hselected := sum_le_sum_of_isTopK hbest hother
  have hbest_total := Finset.sum_sdiff (Finset.subset_univ best) (f := score)
  have hother_total := Finset.sum_sdiff (Finset.subset_univ other) (f := score)
  linarith

end WINA
