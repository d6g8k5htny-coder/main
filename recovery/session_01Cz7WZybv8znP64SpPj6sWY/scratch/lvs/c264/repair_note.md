C264 `OrdinaryH0.lean` — owner repair applied; successor elaborates at the pin (record of a local computation, not a receipt).

Anthropic / Claude (companion owner, actual repair author per the handoff [6113372506](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6113372506); same project account, organizational-independence credit 0). Scientific effect NONE. The original V1 bytes (18,469 B, sha256 `261dd0526d42890d8a4ac1225fcf47228eb3c9799af4106cb643a528bcbf968e`), their source review and the failed computation [6113300903](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6113300903) stay as history; nothing of V1 is rewritten.

**Successor:** 19,803 B, sha256 `3d6b0d6b53413c484b1d010bccf0d53ae7f75becb2d747868f22dbd673fe0a88`. `lean -DwarningAsError=true --root=<dir>` at Lean v4.34.1 / Mathlib `d13f23b7` against `ClosedSuperlevel` (`90d1481b…`): exit 0, empty log (2026-10-11T20:30:53Z to 20:31:14Z). `#print axioms` for all 27 declarations (20 theorems, 7 defs): each exactly `[propext, Classical.choice, Quot.sound]`.

What changed, against the five handoff items (no statement changed; every change is recorded in the module docstring):
1. The homology instances `[Preadditive C] [CategoryWithHomology C]` enter the `variable` scope only after the three declarations that do not use them (`ordinaryH0_componentCoproductMap_iota`, `ordinaryH0_componentFiltration`, `ordinaryH0_componentFiltration_map_iota`, the latter two moved up for that reason; `omit … in` was not used because the scanner refuses `omit` at column 0, and no linter was disabled).
2. `ordinaryH0_iso_naturality`: your `calc` body elaborated with an unresolved `Trans` instance at the pin (the step composing through `(ordinaryH0_componentCoproductFunctor R).obj X`, which is `∐ fun _ => R` only after unfolding), so the proof is the same five steps as explicit rewrites: `rw [← Category.assoc, ordinaryH0_pointClass_map, ordinaryH0_pointClass_iso]` on the left, `erw [← Category.assoc, ordinaryH0_pointClass_iso, ordinaryH0_componentCoproductMap_iota]` on the right (default transparency for exactly that unfolding), then `rfl` (`superlevel_componentMap f.hom (mk x)` is `mk (f x)` definitionally). No assumed naturality or compatibility field.
3. `Functor.isoWhiskerLeft` (qualified; `isoWhiskerLeft` is unknown at the pin without the `Functor` prefix).
4. The three names: `ordinaryH0_componentCoproductMap_iota`, `ordinaryH0_componentIota_isoInv`, `ordinaryH0_componentFiltration_map_iota`, all references updated.
5. `census_auxiliary`: the elaborated census of this module runs next (together with C263's) and every compiler-generated constant it lists goes into `SOURCE_FILES.json` at registration.

Full diff V1 → successor:

```diff
--- OrdinaryH0.handoff.lean	2026-10-11 20:27:15.032879128 +0000
+++ OrdinaryH0.lean	2026-10-11 20:30:53.180542708 +0000
@@ -17,6 +17,20 @@
 filtration. It supplies no Morse chart, critical-level attainment theorem,
 interval decomposition, elder correspondence, Field adapter, count identity,
 stochastic law, or scientific-status promotion.
+
+Owner repairs at the pin (Anthropic/Claude lane, companion owner, recorded as the
+actual repair author; handoff bytes SHA-256
+261dd0526d42890d8a4ac1225fcf47228eb3c9799af4106cb643a528bcbf968e did not elaborate
+with `-DwarningAsError=true`, Math- #428 comments 6113300903 and 6113372506): the
+homology instances `[Preadditive C] [CategoryWithHomology C]` enter the `variable`
+scope only after `ordinaryH0_componentCoproductMap_iota`, `ordinaryH0_componentFiltration`
+and `ordinaryH0_componentFiltration_map_iota`, which do not use them (the two filtration
+declarations therefore precede the homology ones; no statement changed);
+`ordinaryH0_iso_naturality` is proved by explicit `rw`/`erw` reassociation of both sides
+(the source author's five-step `calc` elaborated with an unresolved `Trans` instance
+at the pin) instead of a `simp only` that left the right side unreassociated; `isoWhiskerLeft` is referred to as `Functor.isoWhiskerLeft`; the
+three declaration names ending in `ι` are spelled `…_iota` / `…Iota…` for the
+companion's ASCII identifier rule. No statement changed.
 -/
 
 noncomputable section
@@ -44,7 +58,6 @@
     exact congrFun (superlevel_componentMap_comp f.hom g.hom) q
 
 variable {C : Type u} [Category.{v} C] [HasCoproducts.{w} C]
-  [Preadditive C] [CategoryWithHomology C]
 
 /-- Proves: the component-indexed coproduct construction is a functor.
 Method: compose the actual component functor with the categorical coproduct functor.
@@ -55,13 +68,35 @@
 /-- Proves: a coproduct generator maps to the component of its continuous image.
 Method: the universal coproduct map sends each inclusion to its indexed image.
 Does not establish: that distinct source components stay distinct. -/
-theorem ordinaryH0_componentCoproductMap_ι (R : C) {X Y : TopCat.{w}}
+theorem ordinaryH0_componentCoproductMap_iota (R : C) {X Y : TopCat.{w}}
     (f : X ⟶ Y) (q : ZerothHomotopy X) :
     Sigma.ι (fun _ : ZerothHomotopy X => R) q ≫
         (ordinaryH0_componentCoproductFunctor R).map f =
       Sigma.ι (fun _ : ZerothHomotopy Y => R) (superlevel_componentMap f.hom q) := by
   simp [ordinaryH0_componentCoproductFunctor, ordinaryH0_componentFunctor, sigmaConst]
 
+/-- Proves: the existing closed-superlevel family has a component-indexed coproduct filtration.
+Method: compose its actual continuous inclusions with the proved component coproduct functor.
+Does not establish: regularity, finite dimensionality, or Morse hypotheses for the height function. -/
+def ordinaryH0_componentFiltration {X : Type w} [TopologicalSpace X]
+    (g : X → ℝ) (R : C) : ℝ ⥤ C :=
+  superlevel_filtration g ⋙ ordinaryH0_componentCoproductFunctor R
+
+/-- Proves: a filtration component generator maps by the existing actual component inclusion.
+Method: specialize the coproduct generator map formula to a closed-superlevel inclusion.
+Does not establish: a reverse inclusion or endpoint deletion. -/
+theorem ordinaryH0_componentFiltration_map_iota {X : Type w} [TopologicalSpace X]
+    (g : X → ℝ) (R : C) {s t : ℝ} (hst : s ≤ t)
+    (q : ZerothHomotopy (superlevel_set g s)) :
+    Sigma.ι (fun _ : ZerothHomotopy (superlevel_set g s) => R) q ≫
+        (ordinaryH0_componentFiltration g R).map (homOfLE hst) =
+      Sigma.ι (fun _ : ZerothHomotopy (superlevel_set g t) => R)
+        (superlevel_componentInclusion g hst q) := by
+  exact ordinaryH0_componentCoproductMap_iota R
+    (TopCat.ofHom (superlevel_inclusion g hst)) q
+
+variable [Preadditive C] [CategoryWithHomology C]
+
 /-- Proves: a zero-simplex defines a morphism into actual simplicial homology.
 Method: include its chain generator, lift through the zero differential, then take its class.
 Does not establish: a replacement definition of homology or any naturality premise. -/
@@ -137,7 +172,7 @@
 /-- Proves: inverse comparison takes each component generator to any representing point class.
 Method: cancel the comparison isomorphism and use the point-class formula.
 Does not establish: a choice of distinguished geometric representative. -/
-theorem ordinaryH0_componentι_isoInv (X : TopCat.{w}) (R : C) (x : X) :
+theorem ordinaryH0_componentIota_isoInv (X : TopCat.{w}) (R : C) (x : X) :
     Sigma.ι (fun _ : ZerothHomotopy X => R) (ZerothHomotopy.mk x) ≫
         (TopCat.singularHomology₀Iso X R).inv = ordinaryH0_pointClass X R x := by
   apply (cancel_mono (TopCat.singularHomology₀Iso X R).hom).1
@@ -154,7 +189,7 @@
   apply Sigma.hom_ext
   intro q
   obtain ⟨x, rfl⟩ := ZerothHomotopy.mk_surjective q
-  simpa only [← Category.assoc, ordinaryH0_componentι_isoInv] using h x
+  simpa only [← Category.assoc, ordinaryH0_componentIota_isoInv] using h x
 
 /-- Proves: the existing actual singular H0 comparison commutes with every continuous map.
 Method: test both composites on actual point classes, transport them, and compare component images.
@@ -166,8 +201,11 @@
         (ordinaryH0_componentCoproductFunctor R).map f := by
   apply ordinaryH0_hom_ext X R
   intro x
-  simp only [← Category.assoc, ordinaryH0_pointClass_map, ordinaryH0_pointClass_iso,
-    ordinaryH0_componentCoproductMap_ι, superlevel_componentMap_mk]
+  rw [← Category.assoc, ordinaryH0_pointClass_map, ordinaryH0_pointClass_iso]
+  -- the right side composes through `(ordinaryH0_componentCoproductFunctor R).obj X`, which is
+  -- `∐ fun _ => R` only after unfolding, so its reassociation is rewritten at default transparency
+  erw [← Category.assoc, ordinaryH0_pointClass_iso, ordinaryH0_componentCoproductMap_iota]
+  rfl
 
 /-- Proves: actual ordinary singular H0 is naturally isomorphic to component-indexed coproducts.
 Method: assemble the existing objectwise isomorphisms using the commuting square proved above.
@@ -263,33 +301,13 @@
       ZerothHomotopy.mk (f x) = ZerothHomotopy.mk (f y) := by
   rw [ordinaryH0_difference_map, sub_eq_zero, ordinaryH0_pointClass_eq_iff Y R hR]
 
-/-- Proves: the existing closed-superlevel family has a component-indexed coproduct filtration.
-Method: compose its actual continuous inclusions with the proved component coproduct functor.
-Does not establish: regularity, finite dimensionality, or Morse hypotheses for the height function. -/
-def ordinaryH0_componentFiltration {X : Type w} [TopologicalSpace X]
-    (g : X → ℝ) (R : C) : ℝ ⥤ C :=
-  superlevel_filtration g ⋙ ordinaryH0_componentCoproductFunctor R
-
-/-- Proves: a filtration component generator maps by the existing actual component inclusion.
-Method: specialize the coproduct generator map formula to a closed-superlevel inclusion.
-Does not establish: a reverse inclusion or endpoint deletion. -/
-theorem ordinaryH0_componentFiltration_map_ι {X : Type w} [TopologicalSpace X]
-    (g : X → ℝ) (R : C) {s t : ℝ} (hst : s ≤ t)
-    (q : ZerothHomotopy (superlevel_set g s)) :
-    Sigma.ι (fun _ : ZerothHomotopy (superlevel_set g s) => R) q ≫
-        (ordinaryH0_componentFiltration g R).map (homOfLE hst) =
-      Sigma.ι (fun _ : ZerothHomotopy (superlevel_set g t) => R)
-        (superlevel_componentInclusion g hst q) := by
-  exact ordinaryH0_componentCoproductMap_ι R
-    (TopCat.ofHom (superlevel_inclusion g hst)) q
-
 /-- Proves: the existing actual singular H0 filtration is naturally isomorphic to its component coproduct filtration.
 Method: restrict the proved natural comparison along the existing topological filtration.
 Does not establish: the later interval, elder, maximum/saddle, count, Field, or stochastic bridges. -/
 def ordinaryH0_filtrationIso {X : Type w} [TopologicalSpace X]
     (g : X → ℝ) (R : C) :
     superlevel_h0Filtration g R ≅ ordinaryH0_componentFiltration g R :=
-  isoWhiskerLeft (superlevel_filtration g) (ordinaryH0_naturalIso R)
+  Functor.isoWhiskerLeft (superlevel_filtration g) (ordinaryH0_naturalIso R)
 
 /-- Proves: all ordered closed-superlevel inclusion maps satisfy the actual H0/component square.
 Method: read the naturality of the proved filtration isomorphism at the preorder morphism.
```

Still to come (owner work, separate successor commit after the C262 AMEND push and its readback): gate registration of C263 and C264, the replay, the hosted run and its receipt, then the changed-input nonauthor review of the compiled successor, which this comment does not replace.

---
_Generated by [Claude Code](https://claude.ai/code)_
