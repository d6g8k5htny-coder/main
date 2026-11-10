import Lean
import Component

/-!
Task A contract source only. Component is intentionally absent at the test-first
checkpoint. A missing import/compile is implementation absence, never execution of
these controls. No mock Component, parallel inventory grader, or root main lives here.

Future shared Component boundary:
  requireIdentity : Name → Except String Name
  validateParts : Array (String × ModuleData) →
    Except AuditError (Std.HashMap Name ConstantInfo)
  AuditError.code : AuditError → String
  LoadedModule.{parts, owned, env}
  observeConstant : LoadedModule → ConstantInfo → IO DeclarationObservation
  DeclarationObservation.{name, owner, isPrivate, isProp, kind}
  readRequest, loadExact, encodeName, decodeName as in the accepted design.

IO rejection messages must start exactly "SDGA_ERROR <CODE>:"; assertions accept
only their specified cause. Setup failure and semantic assertion failure are
different records. The parent launcher retains original stdout/stderr, requests,
hashes and process exits, invokes inventory separately against the same baseline,
and reconciles that actual transcript against C02's independent raw/core rows.

Environment supplied by the fixed parent launcher:
  SDGA_CONTROL_CASE                  C01/C02/C03/C12/C13/C15
  SDGA_CONTROL_REQUEST               valid absolute baseline request path
  SDGA_CONTROL_SUBCASE               required for C13; wrong-resolved-path for C15
  SDGA_CONTROL_ALTERNATE_REQUEST     one concrete adverse request for C13/C15

C15 unknown-mode/unknown-field execute the actual native Component CLI in the parent.
Its remaining graph/transcript cases have executor Python_controller.
-/

namespace SDGA
namespace Control

open Lean

private def failure (message : String) : IO α :=
  throw <| IO.userError s!"CONTROL_ASSERT: {message}"

private def setupFailure (message : String) : IO α :=
  throw <| IO.userError s!"CONTROL_SETUP: {message}"

private def check (condition : Bool) (message : String) : IO Unit := do
  unless condition do failure message

private def envRequired (key : String) : IO String := do
  let some value ← IO.getEnv key | setupFailure s!"missing environment {key}"
  if value.isEmpty then setupFailure s!"empty environment {key}"
  return value

private def emit (fields : List (String × Json)) : IO Unit := do
  let stdout ← IO.getStdout
  stdout.putStrLn (Json.mkObj fields).compress
  stdout.flush

private def result (caseId subcaseId status : String) (detail : Json)
    (errorCode : String := "") : IO Unit :=
  emit <| [
    ("record", .str "subcase"), ("case_id", .str caseId),
    ("subcase_id", .str subcaseId), ("executor", .str "Lean"),
    ("status", .str status), ("detail", detail)
  ] ++ (if errorCode.isEmpty then [] else [("error_code", .str errorCode)])

private def run (caseId subcaseId : String) (body : IO Json) : IO Unit := do
  try
    let detail ← body
    result caseId subcaseId "pass" detail
  catch err =>
    result caseId subcaseId "fail" (.str err.toString) "CONTROL_ASSERT"
    throw <| IO.userError s!"CONTROL_ASSERT: {caseId}/{subcaseId}: {err}"

private def success : Json := .mkObj []

private def wires (names : Array Name) : Json :=
  .arr <| names.map (Json.str ∘ SDGA.encodeName)

private def expectIOCode (code : String) (body : IO α) : IO Json := do
  let observed ← try
    let _ ← body
    pure none
  catch err =>
    pure (some err)
  match observed with
  | none => failure s!"expected {code}; operation returned successfully"
  | some (.userError message) =>
    check (message.startsWith s!"SDGA_ERROR {code}:")
      s!"expected exact SDGA_ERROR {code}: prefix; observed {message}"
    return .mkObj [("expected_code", .str code), ("native_error", .str message)]
  | some err =>
    failure s!"expected typed SDGA_ERROR {code}: user error; observed {err}"

private def expectPartCode (parts : Array (String × ModuleData)) (code : String) :
    IO Json := do
  match SDGA.validateParts parts with
  | .ok _ => failure s!"altered actual ModuleData unexpectedly accepted; expected {code}"
  | .error err =>
    check (err.code == code) s!"wrong structural rejection: expected {code}, got {err.code}"
    return .mkObj [("expected_code", .str code), ("alteration", .str "actual-loaded-ModuleData")]

private def getCI (loaded : LoadedModule) (name : Name) : IO ConstantInfo := do
  let some ci := loaded.owned[name]? |
    setupFailure s!"actual compiled fixture CI absent: {SDGA.encodeName name}"
  return ci

private def rawOwner (loaded : LoadedModule) (name : Name) : IO Name := do
  let some idx := loaded.env.getModuleIdxFor? name |
    setupFailure s!"actual declaration owner absent: {SDGA.encodeName name}"
  let some owner := loaded.env.header.moduleNames[idx]? |
    setupFailure "actual owner index outside header.moduleNames"
  let some reverse := loaded.env.getModuleIdx? owner |
    setupFailure "actual owner absent from reverse module index"
  check (idx == reverse) "owner forward/reverse index disagreement"
  return owner

private def fixtureModule : Name := `Fixture.Safe

private def exportedNames (loaded : LoadedModule) : Array Name :=
  loaded.parts[0]!.2.constNames

private def finalData (loaded : LoadedModule) : ModuleData :=
  loaded.parts[2]!.2

private def baseline : IO LoadedModule := do
  let path ← envRequired "SDGA_CONTROL_REQUEST"
  let request ← SDGA.readRequest (System.FilePath.mk path)
  let loaded ← SDGA.loadExact request
  if loaded.parts.map (·.1) != #["exported", "server", "private"] then
    setupFailure "Safe module must load all three ordered genuine OLean parts"
  if loaded.owned.isEmpty then setupFailure "compiled Safe final population is empty"
  return loaded

private def roundTrip (name : Name) (expected : String) : IO Unit := do
  check (SDGA.encodeName name == expected) s!"wire bytes differ: expected {expected}"
  check (SDGA.decodeName expected == .ok name) s!"wire did not preserve exact Name constructor: {expected}"

private def malformedWire (wire : String) : IO Unit := do
  match SDGA.decodeName wire with
  | .error _ => pure ()
  | .ok _ => failure s!"malformed/noncanonical wire accepted: {wire}"

private def c01 : IO Unit := do
  run "C01" "anon-roundtrip" do
    roundTrip .anonymous ""
    return success
  run "C01" "anon-identity-reject" do
    match SDGA.requireIdentity .anonymous with
    | .error _ => return success
    | .ok _ => failure "anonymous Name accepted as module/declaration identity"
  run "C01" "numeric-vs-string" do
    roundTrip (.num .anonymous 12) "n12"
    roundTrip (.str .anonymous "12") "s3132"
    check (SDGA.encodeName (.num .anonymous 12) != SDGA.encodeName (.str .anonymous "12"))
      "numeric and string components coalesced"
    return success
  run "C01" "empty-component" do
    roundTrip (.str .anonymous "") "s"
    roundTrip (.str (.str .anonymous "") "") "s/s"
    return success
  run "C01" "dot-component" do
    roundTrip (.str .anonymous "a.b") "s612e62"
    check (SDGA.encodeName (.str .anonymous "a.b") !=
      SDGA.encodeName (.str (.str .anonymous "a") "b")) "literal dot became namespace separator"
    return success
  run "C01" "slash-component" do
    roundTrip (.str .anonymous "a/b") "s612f62"
    return success
  run "C01" "unicode-normalization-distinct" do
    roundTrip (.str .anonymous "é") "sc3a9"
    roundTrip (.str .anonymous "é") "s65cc81"
    check (SDGA.encodeName (.str .anonymous "é") != SDGA.encodeName (.str .anonymous "é"))
      "distinct Unicode code-point sequences normalized"
    return success
  run "C01" "nonbmp" do
    roundTrip (.str .anonymous "🦉") "sf09fa689"
    return success
  run "C01" "large-nat" do
    roundTrip (.num .anonymous 18446744073709551617) "n18446744073709551617"
    return success
  run "C01" "uppercase-hex-reject" do
    malformedWire "s4A"
    return success
  run "C01" "leading-zero-reject" do
    malformedWire "n012"
    malformedWire "n00"
    return success
  run "C01" "invalid-utf8-reject" do
    malformedWire "sff"
    malformedWire "sc080"
    return success

/-- Direct pinned-core judgment is the oracle for the shared observation helper. -/
private def nativeIsProp (loaded : LoadedModule) (ci : ConstantInfo) : IO Bool := do
  let (observed, _, _) ← (Lean.Meta.isProp ci.type).toIO
    { fileName := "<sdga-task-a-control>", fileMap := default }
    { env := loaded.env }
  return observed

/-- Exhaustive raw CI constructor oracle; does not call the production observation. -/
private def nativeKind : ConstantInfo → String
  | .thmInfo _ => "theorem"
  | .defnInfo _ => "definition"
  | .opaqueInfo _ => "opaque"
  | .axiomInfo _ => "axiom"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"
  | .quotInfo _ => "quotient"

/-- Independently observed fields for reconciliation with the actual CLI transcript. -/
private def nativeRow (loaded : LoadedModule) (ci : ConstantInfo) : IO Json := do
  let owner ← rawOwner loaded ci.name
  let isProp ← nativeIsProp loaded ci
  return .mkObj [
    ("name", .str (SDGA.encodeName ci.name)),
    ("module", .str (SDGA.encodeName owner)),
    ("kind", .str (nativeKind ci)),
    ("is_prop", .bool isProp),
    ("is_private", .bool (isPrivateName ci.name || !(exportedNames loaded).contains ci.name)),
    ("unsafe", .bool ci.isUnsafe), ("partial", .bool ci.isPartial)
  ]

private def rawPartRows (loaded : LoadedModule) : Json :=
  .arr <| loaded.parts.map fun (level, part) =>
    .mkObj [
      ("level", .str level), ("const_names", wires part.constNames),
      ("constant_names", wires (part.constants.map (·.name))),
      ("extra_const_names", wires part.extraConstNames)
    ]

private def c02 (loaded : LoadedModule) : IO Unit := do
  run "C02" "raw-tables" do
    for (_, part) in loaded.parts do
      check (part.constNames == part.constants.map (·.name))
        "raw ModuleData parallel tables differ in length, order or Name identity"
      for name in part.constNames do
        check ((finalData loaded).constNames.contains name)
          "earlier raw population is absent from final private"
    match SDGA.validateParts loaded.parts with
    | .error err => failure s!"genuine raw parts refused with {err.code}"
    | .ok actual =>
      check (actual.size == loaded.owned.size) "validated final map census differs"
      for (name, _) in loaded.owned.toArray do
        check (actual.contains name) "loaded final map differs from validated actual parts"
    return .mkObj [("part_count", toJson loaded.parts.size)]
  run "C02" "foreign-owner" do
    let ci ← getCI loaded `ForeignNamespace.visible
    let owner ← rawOwner loaded ci.name
    let row ← SDGA.observeConstant loaded ci
    check (owner == fixtureModule && row.owner == owner && row.name == ci.name)
      "foreign-namespace declaration lost actual module ownership"
    return .mkObj [("name", .str (SDGA.encodeName ci.name)), ("owner", .str (SDGA.encodeName owner))]
  run "C02" "explicit-private" do
    let candidates := loaded.owned.toArray.filter fun (name, _) =>
      privateToUserName? name == some `Fixture.Safe.hidden
    check (candidates.size == 1) "explicit private source theorem must have exactly one actual CI"
    let (name, ci) := candidates[0]!
    let row ← SDGA.observeConstant loaded ci
    check (isPrivateName name && row.isPrivate && row.name == name)
      "explicit private spelling or is_private observation lost"
    check ((← rawOwner loaded name) == fixtureModule && row.owner == fixtureModule)
      "explicit private theorem owner differs"
    return .mkObj [("name", .str (SDGA.encodeName name))]
  run "C02" "module-private" do
    let ci ← getCI loaded `Fixture.Safe.unexported
    check (!isPrivateName ci.name && !(exportedNames loaded).contains ci.name)
      "unexported fixture must preserve raw public-style Name but be absent exported"
    let row ← SDGA.observeConstant loaded ci
    check (row.name == ci.name && row.isPrivate && row.owner == fixtureModule)
      "exported-population is_private rule or module-private owner lost"
    check ((← rawOwner loaded ci.name) == fixtureModule) "raw module-private owner differs"
    return .mkObj [("name", .str (SDGA.encodeName ci.name))]
  run "C02" "generated-present" do
    let ind ← getCI loaded `Fixture.Safe.Switch
    match ind with
    | .inductInfo val => check (val.ctors.length == 2) "actual inductive does not have two constructors"
    | _ => failure "actual Switch CI is not inductive"
    let generated := (finalData loaded).constants.filter fun ci =>
      match ci with
      | .recInfo val => val.all.contains `Fixture.Safe.Switch
      | _ => false
    check (!generated.isEmpty) "no actual source-generated recursor CI in raw final ModuleData"
    for ci in generated do
      let row ← SDGA.observeConstant loaded ci
      check (loaded.owned.contains ci.name && row.name == ci.name &&
        row.kind == "recursor" && row.owner == fixtureModule)
        "actual generated recursor absent or misclassified"
      check ((← rawOwner loaded ci.name) == fixtureModule) "generated raw owner differs"
    return .mkObj [("generated_names", wires (generated.map (·.name)))]
  run "C02" "dependency-excluded" do
    for name in #[`True, `True.intro, `Nat] do
      check ((loaded.env.find? name).isSome) "dependency setup constant is absent"
      check (!loaded.owned.contains name && !(finalData loaded).constNames.contains name)
        "dependency-owned constant entered owned census"
      check ((← rawOwner loaded name) != fixtureModule) "dependency incorrectly attributed to Safe"
    return success
  run "C02" "final-rows-owner-census" do
    let final := finalData loaded
    check (final.constants.size == loaded.owned.size)
      "owned map does not contain exactly the raw final CI population"
    for ci in final.constants do
      check (loaded.owned.contains ci.name) "raw final CI omitted from loaded owned map"
      let row ← SDGA.observeConstant loaded ci
      check (row.name == ci.name && row.owner == fixtureModule)
        "shared observation omitted or renamed actual owned declaration"
      check ((← rawOwner loaded ci.name) == fixtureModule) "actual final owner differs"
      check (row.isPrivate == (isPrivateName ci.name || !(exportedNames loaded).contains ci.name))
        "is_private disagrees with raw exported population/private spelling"
    let mut oracleRows : Array Json := #[]
    for ci in final.constants do
      oracleRows := oracleRows.push (← nativeRow loaded ci)
    return .mkObj [
      ("final_names", wires final.constNames),
      ("oracle_rows", .arr oracleRows), ("raw_parts", rawPartRows loaded),
      ("oracle_source", .str "raw-ConstantInfo/core-Meta.isProp/actual-owner/raw-exported")
    ]

private def c03 (loaded : LoadedModule) : IO Unit := do
  run "C03" "proof-def-prop" do
    let ci ← getCI loaded `Fixture.Safe.proof
    match ci with
    | .defnInfo _ => pure ()
    | _ => failure "proof must be an actual definition CI, not a theorem-kind shortcut"
    check (ci.type == .const `True []) "raw compiled proof.type is not actual True"
    let row ← SDGA.observeConstant loaded ci
    let judged ← nativeIsProp loaded ci
    check (judged && row.kind == "definition" && row.isProp == judged && row.name == ci.name)
      "shared Meta.isProp observation did not classify True-valued definition as proof"
    return .mkObj [("raw_type", .str "True"), ("is_prop", .bool row.isProp)]
  run "C03" "prop-valued-data" do
    let ci ← getCI loaded `Fixture.Safe.propValue
    match ci with
    | .defnInfo _ => pure ()
    | _ => failure "propValue must be an actual definition CI"
    check (ci.type == .sort .zero) "raw compiled propValue.type is not Prop"
    let row ← SDGA.observeConstant loaded ci
    let judged ← nativeIsProp loaded ci
    check (!judged && row.kind == "definition" && row.isProp == judged && row.name == ci.name)
      "shared Meta.isProp observation misclassified Prop-valued data"
    return .mkObj [("raw_type", .str "Prop"), ("is_prop", .bool row.isProp)]
  run "C03" "nat-data" do
    let ci ← getCI loaded `Fixture.Safe.natValue
    check (ci.type == .const `Nat []) "raw compiled natValue.type is not Nat"
    let row ← SDGA.observeConstant loaded ci
    check (row.kind == "definition" && !row.isProp && row.name == ci.name)
      "actual Nat definition was not observed as data"
    return .mkObj [("raw_type", .str "Nat"), ("is_prop", .bool row.isProp)]
  run "C03" "malformed-type-block" do
    let ci ← getCI loaded `Fixture.Safe.proof
    let .defnInfo info := ci | setupFailure "malformed-type control needs actual proof definition CI"
    -- Alter only an actual compiled CI type, preserving its Name/value/levels/safety.
    let openTypes : Array Expr := #[
      .bvar 0,
      .fvar ⟨`SDGAControlAbsentFVar⟩,
      .mvar ⟨`SDGAControlAbsentMVar⟩
    ]
    let mut errors : Array Json := #[]
    for badType in openTypes do
      let altered := ConstantInfo.defnInfo { info with type := badType }
      check (altered.name == ci.name) "alteration changed actual CI identity"
      errors := errors.push (← expectIOCode "MALFORMED_CONSTANT"
        (SDGA.observeConstant loaded altered))
    return .mkObj [("alteration", .str "actual-loaded-ConstantInfo-type"),
      ("native_errors", .arr errors)]

private def c12 (loaded : LoadedModule) : IO Unit := do
  -- All five negatives retain the actual loaded module regions through loaded.
  -- No standalone synthetic ModuleData or parallel structural validator is used.
  let part := finalData loaded
  if part.constants.isEmpty then setupFailure "no actual CI available for ModuleData alterations"
  let good ← getCI loaded `Fixture.Safe.good
  let .thmInfo theoremInfo := good | setupFailure "good source theorem CI is not thmInfo"
  run "C12" "table-length" do
    let altered := { part with constants := part.constants.pop }
    expectPartCode (loaded.parts.set! 2 ("private", altered)) "PART_TABLE"
  run "C12" "table-identity" do
    let altered := { part with constNames := part.constNames.set! 0 `SDGAControlChangedIdentity }
    expectPartCode (loaded.parts.set! 2 ("private", altered)) "PART_TABLE"
  run "C12" "duplicate-name" do
    let duplicate := part.constants[0]!
    let altered := { part with
      constNames := part.constNames.push duplicate.name
      constants := part.constants.push duplicate
    }
    expectPartCode (loaded.parts.set! 2 ("private", altered)) "PART_DUPLICATE"
  run "C12" "missing-final-row" do
    check ((exportedNames loaded).contains good.name)
      "missing-final-row setup requires actual exported good theorem"
    let altered := { part with
      constNames := part.constNames.filter (· != good.name)
      constants := part.constants.filter (·.name != good.name)
    }
    check (altered.constants.size + 1 == part.constants.size)
      "missing-final-row must remove exactly one actual source CI"
    expectPartCode (loaded.parts.set! 2 ("private", altered)) "PART_POPULATION"
  run "C12" "earlier-subset" do
    let earlier := loaded.parts[0]!.2
    let introducedName : Name := `SDGAControlEarlierOnly
    check (!loaded.owned.contains introducedName) "earlier-only alteration name already exists"
    let renamed := ConstantInfo.thmInfo { theoremInfo with name := introducedName }
    let altered := { earlier with
      constNames := earlier.constNames.push renamed.name
      constants := earlier.constants.push renamed
    }
    expectPartCode (loaded.parts.set! 0 ("exported", altered)) "PART_POPULATION"

private def c13 : IO Unit := do
  let subcase ← envRequired "SDGA_CONTROL_SUBCASE"
  let code ← match subcase with
    | "missing-server" => pure "MISSING_PART"
    | "reordered-parts" => pure "PART_LEVELS"
    | "truncated-private" => pure "OBJECT_READ"
    | _ => setupFailure s!"unsupported C13 subcase {subcase}"
  let alternate ← envRequired "SDGA_CONTROL_ALTERNATE_REQUEST"
  -- The parent first compiles and inventories the genuine baseline successfully.
  -- Missing/truncated controls use a copied output root selected by LEAN_PATH;
  -- reloading the original request here would create an unrelated path failure.
  -- Parse the concrete request outside the expected load-refusal assertion.
  let request ← SDGA.readRequest (System.FilePath.mk alternate)
  run "C13" subcase do
    expectIOCode code (SDGA.loadExact request)

private def c15 : IO Unit := do
  let subcase ← envRequired "SDGA_CONTROL_SUBCASE"
  if subcase != "wrong-resolved-path" then
    setupFailure s!"C15 Controls owns only wrong-resolved-path, received {subcase}"
  let alternate ← envRequired "SDGA_CONTROL_ALTERNATE_REQUEST"
  let request ← SDGA.readRequest (System.FilePath.mk alternate)
  run "C15" "wrong-resolved-path" do
    expectIOCode "RESOLVED_PATH" (SDGA.loadExact request)

end Control

/-- One fixed native control case per compiler invocation; no root main or generic runner. -/
def runControls : IO Unit := do
  try
    let selected ← Control.envRequired "SDGA_CONTROL_CASE"
    match selected with
    | "C01" => Control.c01
    | "C02" => Control.c02 (← Control.baseline)
    | "C03" => Control.c03 (← Control.baseline)
    | "C12" => Control.c12 (← Control.baseline)
    | "C13" => Control.c13
    | "C15" =>
      -- Validate native setup in the original output root before the wrong-path negative.
      let _ ← Control.baseline
      Control.c15
    | _ => Control.setupFailure s!"unsupported fixed Task A case {selected}"
  catch err =>
    -- A subcase assertion already emitted its fail row. A setup rejection has no
    -- subcase execution row, and can never count as an expected semantic refusal.
    match err with
    | .userError message =>
      if message.startsWith "CONTROL_ASSERT:" then
        throw err
      else
        Control.emit [
          ("record", .str "setup_error"), ("executor", .str "Lean"),
          ("status", .str "fail"), ("detail", .str message)
        ]
        throw err
    | _ =>
      Control.emit [
        ("record", .str "setup_error"), ("executor", .str "Lean"),
        ("status", .str "fail"), ("detail", .str err.toString)
      ]
      throw err

end SDGA

#eval SDGA.runControls
