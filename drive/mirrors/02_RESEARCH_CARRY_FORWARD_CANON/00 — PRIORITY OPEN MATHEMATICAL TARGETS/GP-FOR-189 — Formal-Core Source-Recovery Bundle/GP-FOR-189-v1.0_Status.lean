namespace ResearchFormalCoreR1

set_option autoImplicit false

inductive ProofState where
  | exactDraftCiPending
  | polynomialCompanionCiPending
  | scalarCompanionCiPending
  | partialDefinitions
  | logicalInterfaceOnly
  | numericalMetadataOnly
  | conditionalWrapperOnly
  | exactJacobianOnly
  | open
  | retractedUnconditionalStatus
  deriving Repr, DecidableEq

structure ClaimRecord where
  claimId : String
  state : ProofState
  scope : String
  excluded : String
  deriving Repr

/-- Machine-readable status records. These are metadata, not theorem axioms. -/
def formalizationMatrix : List ClaimRecord := [
  { claimId := "EC-005",
    state := .exactDraftCiPending,
    scope := "Canonical A₂ normalized fold-gap algebra",
    excluded := "No universal physical-field coefficient" },
  { claimId := "EC-008",
    state := .polynomialCompanionCiPending,
    scope := "Scalar factor expansion for the declared planar collision-Gram determinant",
    excluded := "Matrix construction and exact-torus transfer are not encoded" },
  { claimId := "EC-010",
    state := .exactDraftCiPending,
    scope := "Two declared polynomial identities",
    excluded := "No surrounding fold, positivity, Palm, or rate theorem" },
  { claimId := "EC-011",
    state := .scalarCompanionCiPending,
    scope := "Scalar cancellation companion",
    excluded := "ODE, Hessian symmetry, and section-normalization theorem are not encoded" },
  { claimId := "EC-012",
    state := .partialDefinitions,
    scope := "Strip-threshold data and endpoint cases",
    excluded := "Real-power majorant and factor-two theorem remain unformalized" },
  { claimId := "EC-013",
    state := .logicalInterfaceOnly,
    scope := "Conditional elder-rule defect inclusion interface",
    excluded := "Morse–Smale geometry, event definitions, and probability bounds are parameters" },
  { claimId := "EC-014",
    state := .scalarCompanionCiPending,
    scope := "Contact-power scalar identity",
    excluded := "Six-pin matrix determinant and density-transform theorem are not encoded" },
  { claimId := "EC-018",
    state := .numericalMetadataOnly,
    scope := "Interval-evidence metadata",
    excluded := "No interval kernel or replay proof in Lean" },
  { claimId := "EC-020",
    state := .numericalMetadataOnly,
    scope := "Interval-evidence metadata",
    excluded := "No interval kernel or replay proof in Lean" },
  { claimId := "P0.1",
    state := .conditionalWrapperOnly,
    scope := "Status wrapper for candidate uniform exact-field positivity",
    excluded := "No unconditional theorem; independent G/D/T review remains open" },
  { claimId := "EC-019",
    state := .conditionalWrapperOnly,
    scope := "T1 pass and T2 pass with package-integrity hold",
    excluded := "No terminal closure without authoritative raw-carrier replay and operator adjudication" },
  { claimId := "P0.2",
    state := .conditionalWrapperOnly,
    scope := "Conditional exact-field cubic-rate reduction",
    excluded := "No exact-field law without all named hypotheses" },
  { claimId := "Theorem-B-Jacobian",
    state := .exactJacobianOnly,
    scope := "Exact ell = kappa r^3 / 6 change-of-variables factor",
    excluded := "No contact intensity, selection, tails, multiplicity, or off-fold theorem" },
  { claimId := "Theorem-B-Unconditional",
    state := .retractedUnconditionalStatus,
    scope := "Historical full-kappa positive-coefficient claim",
    excluded := "Current PROVEN-HERE status retracted by GP-AUD-187" }
]

end ResearchFormalCoreR1
