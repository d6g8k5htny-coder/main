import Lean
import Lean.Meta
import Lean.Replay
import Lean.Util.FoldConsts
import Init.Data.String.Legacy

/-!
Task A bridge: actual object loading and inventory only.

The request digests are controller measurements, repeated after exact path
resolution; this program does not independently authenticate SHA-256. Raw OLean
and IR compacted regions remain live with LoadedModule and are never freed here.
Task B policy qualification, Task C kernel replay and Task D IR qualification
remain incomplete. No inventory result establishes SDGA qualification.
-/

namespace SDGA

open Lean

private def failCode (code message : String) : IO α :=
  throw <| IO.userError s!"SDGA_ERROR {code}: {message}"

/-- A lossless value may be anonymous; an inventory identity may not. -/
def requireIdentity (name : Name) : Except String Name :=
  match name with
  | .anonymous => .error "anonymous module/declaration identity"
  | _ => .ok name

private def hexDigit (n : Nat) : Char :=
  Char.ofNat (if n < 10 then 48 + n else 97 + (n - 10))

private def hexBytes (bytes : ByteArray) : String :=
  bytes.foldl (fun out byte =>
    (out.push (hexDigit (byte.toNat / 16))).push (hexDigit (byte.toNat % 16))) ""

private def nameComponents : Name → List String
  | .anonymous => []
  | .str parent component => nameComponents parent ++ ["s" ++ hexBytes component.toUTF8]
  | .num parent component => nameComponents parent ++ ["n" ++ toString component]

/-- Task 1 constructor-preserving wire; never a dotted/display Name encoding. -/
def encodeName (name : Name) : String :=
  String.intercalate "/" (nameComponents name)

private def hexValue (c : Char) : Option Nat :=
  if '0' ≤ c && c ≤ '9' then some (c.toNat - '0'.toNat)
  else if 'a' ≤ c && c ≤ 'f' then some (10 + c.toNat - 'a'.toNat)
  else none

private def decodeHex : List Char → Except String ByteArray
  | [] => .ok ByteArray.empty
  | a :: b :: rest => do
    let some hi := hexValue a | throw "noncanonical lowercase hex"
    let some lo := hexValue b | throw "noncanonical lowercase hex"
    let tail ← decodeHex rest
    return ByteArray.empty.push (UInt8.ofNat (16 * hi + lo)) ++ tail
  | _ => .error "odd hex byte count"

private def decodeComponent (parent : Name) (component : String) : Except String Name := do
  match component.toList with
  | 's' :: digits =>
    let bytes ← decodeHex digits
    let some value := String.fromUTF8? bytes | throw "invalid UTF-8 string component"
    return .str parent value
  | 'n' :: digits =>
    if digits.isEmpty || !(digits.all fun c => '0' ≤ c && c ≤ '9') then
      throw "noncanonical numeric component"
    if digits.length > 1 && digits.head? == some '0' then
      throw "leading zero in numeric component"
    let value := digits.foldl (fun n c => 10 * n + (c.toNat - '0'.toNat)) 0
    return .num parent value
  | _ => throw "invalid Name component"

/-- Reject noncanonical hex, decimal and UTF-8 without Unicode normalization. -/
def decodeName (wire : String) : Except String Name := do
  if wire.isEmpty then return .anonymous
  let mut name := Name.anonymous
  for component in wire.splitOn "/" do
    name ← decodeComponent name component
  if encodeName name != wire then throw "noncanonical Name wire"
  return name

/-- Typed audit causes used by the shared structural and dependency helpers. -/
inductive AuditError where
  | partTable (message : String)
  | partDuplicate (name : Name)
  | partPopulation (name : Name)
  | missingConstant (name : Name)
  | ownedAxiom (name : Name)
  | forbiddenAxiom (name : Name)
  | malformedConstant (name : Name) (message : String)
  | unsafeReplaySkip (name : Name)
  | partialReplaySkip (name : Name)
  | duplicateBase (name : Name)
  | unsupportedExtras (name : Name)
  | unsupportedIRResidue (name : Name)
  deriving Inhabited

def AuditError.code : AuditError → String
  | .partTable _ => "PART_TABLE"
  | .partDuplicate _ => "PART_DUPLICATE"
  | .partPopulation _ => "PART_POPULATION"
  | .missingConstant _ => "MISSING_CONSTANT"
  | .ownedAxiom _ => "OWNED_AXIOM"
  | .forbiddenAxiom _ => "FORBIDDEN_AXIOM"
  | .malformedConstant _ _ => "MALFORMED_CONSTANT"
  | .unsafeReplaySkip _ => "UNSAFE_REPLAY_SKIP"
  | .partialReplaySkip _ => "PARTIAL_REPLAY_SKIP"
  | .duplicateBase _ => "DUPLICATE_BASE"
  | .unsupportedExtras _ => "UNSUPPORTED_EXTRAS"
  | .unsupportedIRResidue _ => "UNSUPPORTED_IR_RESIDUE"

def AuditError.message : AuditError → String
  | .partTable message => message
  | .partDuplicate name => encodeName name
  | .partPopulation name => encodeName name
  | .missingConstant name => encodeName name
  | .ownedAxiom name => encodeName name
  | .forbiddenAxiom name => encodeName name
  | .malformedConstant name message => s!"{encodeName name}: {message}"
  | .unsafeReplaySkip name => encodeName name
  | .partialReplaySkip name => encodeName name
  | .duplicateBase name => encodeName name
  | .unsupportedExtras name => encodeName name
  | .unsupportedIRResidue name => encodeName name

private def fromAudit (result : Except AuditError α) : IO α :=
  match result with
  | .ok value => pure value
  | .error error => failCode error.code error.message

structure PartRequest where
  level : String
  path : System.FilePath
  sha256 : String
  deriving Inhabited

structure ModuleRequest where
  schema : Nat
  scopeId : String
  profileSha256 : String
  module : Name
  oleanParts : Array PartRequest
  irParts : Array PartRequest
  deriving Inhabited

/- Core Json.parse overwrites duplicate fields. This bounded request parser uses
   pinned core token parsers but rejects duplicate fields before constructing an
   object, including duplicates inside part records. -/
namespace RequestJSON

open Std.Internal.Parsec
open Std.Internal.Parsec.String

mutual
  private partial def arrayCore (acc : Array Json) : Parser (Array Json) := do
    let value ← valueCore
    let acc := acc.push value
    let c ← any
    if c == ']' then
      ws
      return acc
    else if c == ',' then
      ws
      arrayCore acc
    else
      fail "unexpected character in request array"

  private partial def objectCore (fields : Std.TreeMap.Raw String Json) :
      Parser (Std.TreeMap.Raw String Json) := do
    Lean.Json.Parser.lookahead (fun c => c == '"') "\""
    skip
    let key ← Lean.Json.Parser.str
    if (fields.get? key).isSome then fail "duplicate request field"
    ws
    Lean.Json.Parser.lookahead (fun c => c == ':') ":"
    skip
    ws
    let value ← valueCore
    let fields := fields.insert key value
    let c ← any
    if c == '}' then
      ws
      return fields
    else if c == ',' then
      ws
      objectCore fields
    else
      fail "unexpected character in request object"

  private partial def valueCore : Parser Json := do
    let c ← peek!
    if c == '[' then
      skip
      ws
      if (← peek!) == ']' then
        skip
        ws
        return .arr #[]
      return .arr (← arrayCore #[])
    else if c == '{' then
      skip
      ws
      if (← peek!) == '}' then
        skip
        ws
        return .obj ∅
      return .obj (← objectCore ∅)
    else if c == '"' then
      skip
      let value ← Lean.Json.Parser.str
      ws
      return .str value
    else if c == 'f' then
      skipString "false"
      ws
      return .bool false
    else if c == 't' then
      skipString "true"
      ws
      return .bool true
    else if c == 'n' then
      skipString "null"
      ws
      return .null
    else if c == '-' || ('0' ≤ c && c ≤ '9') then
      let value ← Lean.Json.Parser.num
      ws
      return .num value
    else
      fail "unexpected request input"
end

def parse (input : String) : Except String Json :=
  Parser.run (do
    ws
    let value ← valueCore
    eof
    return value) input

end RequestJSON

private def exactFields (value : Json) (expected : List String) : Except String Unit := do
  let object ← value.getObj?
  let fields := object.foldl (fun fields key _ => key :: fields) ([] : List String)
  if fields.length != expected.length || !(fields.all expected.contains) then
    throw "request record fields differ"

private def jsonString (value : Json) (field : String) : Except String String := do
  (← value.getObjVal? field).getStr?

private def validDigest (value : String) : Bool :=
  value.length == 64 && value.toList.all (fun c => (hexValue c).isSome)

private def canonicalPath (path : System.FilePath) : Bool :=
  path.isAbsolute && path.normalize == path &&
    !(path.toString.toList.contains '\x00') &&
    (((path.toString.splitOn "/").drop 1).all fun component =>
      !component.isEmpty && component != "." && component != "..")

private def modulePathName : Name → Bool
  | .anonymous => true
  | .str parent component =>
    modulePathName parent && !component.isEmpty &&
      !(component.toList.any fun c => c == '/' || c == '\\' || c == '\x00') &&
      component != "." && component != ".."
  | .num _ _ => false

private def parsePart (value : Json) (allowed : List String) : Except String PartRequest := do
  exactFields value ["level", "path", "sha256"]
  let level ← jsonString value "level"
  let path := System.FilePath.mk (← jsonString value "path")
  let sha256 ← jsonString value "sha256"
  if !allowed.contains level then throw "unknown object part level"
  if !canonicalPath path then throw "object path must be absolute and normalized"
  if !validDigest sha256 then throw "invalid measured SHA-256"
  return { level, path, sha256 }

private def parseRequest (value : Json) : Except String ModuleRequest := do
  exactFields value ["schema", "scope_id", "profile_sha256", "module", "olean_parts", "ir_parts"]
  let schema ← (← value.getObjVal? "schema").getNat?
  if schema != 1 then throw "unsupported request schema"
  let scopeId ← jsonString value "scope_id"
  let profileSha256 ← jsonString value "profile_sha256"
  if scopeId.isEmpty || !validDigest profileSha256 then throw "invalid request identity"
  let moduleName ← decodeName (← jsonString value "module") >>= requireIdentity
  if !modulePathName moduleName then throw "module Name is not a safe import path"
  let oleanRows ← (← value.getObjVal? "olean_parts").getArr?
  let irRows ← (← value.getObjVal? "ir_parts").getArr?
  let oleanParts ← oleanRows.mapM (fun row => parsePart row ["exported", "server", "private"])
  let irParts ← irRows.mapM (fun row => parsePart row ["ir.sig", "ir"])
  -- Part order and existence are load-time causes, not JSON parse failures.
  return { schema, scopeId, profileSha256, module := moduleName, oleanParts, irParts }

/-- Parse structure only; missing/reordered objects are refused by loadExact. -/
def readRequest (path : System.FilePath) : IO ModuleRequest := do
  let text ← try IO.FS.readFile path catch error =>
    failCode "REQUEST_SCHEMA" s!"cannot read request: {error}"
  match RequestJSON.parse text >>= parseRequest with
  | .ok request => return request
  | .error message => failCode "REQUEST_SCHEMA" message

/-- Retain every object region for as long as any raw CI/expression is referenced. -/
structure LoadedModule where
  request : ModuleRequest
  parts : Array (String × ModuleData)
  regions : Array CompactedRegion
  irParts : Array (String × ModuleData)
  irRegions : Array CompactedRegion
  owned : Std.HashMap Name ConstantInfo
  env : Environment

/-- Validate raw parallel tables and duplicates before constructing the final map. -/
def validateParts (parts : Array (String × ModuleData)) :
    Except AuditError (Std.HashMap Name ConstantInfo) := do
  if parts.isEmpty then throw (.partTable "no OLean data")
  for (level, part) in parts do
    if part.constNames != part.constants.map (·.name) then
      throw (.partTable s!"parallel table disagreement in {level}")
    let mut seen : Std.HashSet Name := {}
    for name in part.constNames do
      match requireIdentity name with
      | .error message => throw (.malformedConstant name message)
      | .ok _ => pure ()
      if seen.contains name then throw (.partDuplicate name)
      seen := seen.insert name
  let final := parts[parts.size - 1]!.2
  let mut owned : Std.HashMap Name ConstantInfo := {}
  for ci in final.constants do
    owned := owned.insert ci.name ci
  for (_, part) in parts do
    for name in part.constNames do
      if !owned.contains name then throw (.partPopulation name)
  return owned

private def exactPartPaths (base : System.FilePath) (parts : Array PartRequest)
    (ir : Bool) : IO Unit := do
  for part in parts do
    let expected := if ir then base.withExtension part.level else
      match part.level with
      | "exported" => base
      | "server" => base.addExtension "server"
      | "private" => base.addExtension "private"
      | _ => base
    if part.path != expected then
      failCode "RESOLVED_PATH" s!"part {part.level} differs from requested base"
    unless (← part.path.pathExists) do
      failCode "MISSING_PART" s!"{part.level}: {part.path}"
    let actual ← try realPathNormalized part.path catch error =>
      failCode "OBJECT_READ" s!"{part.level}: {error}"
    if actual != part.path then
      failCode "RESOLVED_PATH" s!"noncanonical resolved part {part.level}"

/- The pinned ordinary OLean writer uses the v2 LE64 layout:
   marker[5], version, flags, lean_version[33], githash[40], base_addr[8],
   then a saved root-address word. v2 has no total-length field. Its native
   mmap fast path trusts that root, so check physical root framing before FFI.
   This is a bounded framing check, not a validator for arbitrary object graphs. -/
private structure ObjectFrame where
  path : System.FilePath
  base : Nat
  byteSize : Nat

private def framingError (message : String) : IO α :=
  throw <| IO.userError s!"object framing: {message}"

private def readExactBytes (handle : IO.FS.Handle) (count : Nat) (label : String) :
    IO ByteArray := do
  let mut bytes := ByteArray.empty
  while bytes.size < count do
    let chunk ← handle.read (count - bytes.size).toUSize
    if chunk.size == 0 then framingError s!"{label}: short read of {count} bytes"
    bytes := bytes ++ chunk
  return bytes

private def skipObjectBytes (handle : IO.FS.Handle) (count : Nat) (label : String) :
    IO Unit := do
  let mut remaining := count
  while remaining > 0 do
    -- A streaming memory bound, not an acceptance threshold for an object file.
    let chunk ← handle.read (min remaining 65536).toUSize
    if chunk.size == 0 then framingError s!"{label}: short read before root object"
    remaining := remaining - chunk.size

private def littleNat (bytes : ByteArray) (offset width : Nat) : Nat :=
  (List.range width).foldl
    (fun value i => value + (bytes.get! (offset + i)).toNat * 256 ^ i) 0

private def preflightParts (parts : Array PartRequest) : IO Unit := do
  if System.Platform.numBits != 64 || !System.Platform.isLinux ||
      !System.Platform.target.startsWith "x86_64" then
    framingError "unsupported object ABI; this bridge requires Linux x86_64 LE64"
  let headerSize : Nat := 5 + 1 + 1 + 33 + 40 + 8
  let wordSize : Nat := 8
  let objectHeaderSize : Nat := 8
  let mut frames : Array ObjectFrame := #[]
  for part in parts do
    let byteSize := (← part.path.metadata).byteSize.toNat
    let prefix ← IO.FS.withFile part.path .read fun handle =>
      readExactBytes handle (headerSize + wordSize) part.path.toString
    let marker := "olean".toUTF8
    for i in [:marker.size] do
      if prefix.get! i != marker.get! i then framingError s!"{part.path}: invalid magic"
    if prefix.get! 5 != 2 then
      framingError s!"{part.path}: unsupported compacted format; expected ordinary OLean v2"
    if byteSize % wordSize != 0 then
      framingError s!"{part.path}: partial machine word at physical end"
    let base := littleNat prefix 80 wordSize
    let root := littleNat prefix headerSize wordSize
    let frame : ObjectFrame := { path := part.path, base, byteSize }
    if base % 65536 != 0 || base + byteSize > 2 ^ 64 then
      framingError s!"{part.path}: invalid logical address range"
    for prior in frames do
      if base < prior.base + prior.byteSize && prior.base < base + byteSize then
        framingError s!"{part.path}: overlapping logical object parts"
    frames := frames.push frame
    if root % wordSize != 0 then framingError s!"{part.path}: unaligned saved root"
    -- A later part's root may legitimately be deduplicated into an earlier part.
    let some owner := frames.find? (fun candidate =>
      candidate.base + headerSize + wordSize ≤ root &&
        root + objectHeaderSize ≤ candidate.base + candidate.byteSize) |
      framingError s!"{part.path}: saved root {root} is outside physical data bounds"
    let rootOffset := root - owner.base
    let rootHeader ← IO.FS.withFile owner.path .read fun handle => do
      skipObjectBytes handle rootOffset owner.path.toString
      readExactBytes handle objectHeaderSize owner.path.toString
    let rootSize := littleNat rootHeader 4 2
    -- Small compacted constructors store their byte size in m_cs_sz.
    if littleNat rootHeader 0 4 != 0 || rootHeader.get! 7 != 0 ||
        rootSize < objectHeaderSize then
      framingError s!"{part.path}: invalid compacted root constructor header"
    let framedRootSize := (rootSize + wordSize - 1) / wordSize * wordSize
    if rootOffset + framedRootSize > owner.byteSize then
      framingError s!"{part.path}: saved root object exceeds physical data bounds"

private def readParts (parts : Array PartRequest) :
    IO (Array (ModuleData × CompactedRegion)) := do
  try
    preflightParts parts
    readModuleDataParts (parts.map (·.path))
  catch error =>
    failCode "OBJECT_READ" error.toString

private def ownerOf (env : Environment) (name : Name) : IO Name := do
  let some index := env.getModuleIdxFor? name |
    failCode "OWNER" s!"missing actual owner for {encodeName name}"
  let some owner := env.header.moduleNames[index]? |
    failCode "OWNER" s!"owner index out of range for {encodeName name}"
  let some reverse := env.getModuleIdx? owner |
    failCode "OWNER" s!"owner reverse index missing for {encodeName name}"
  if reverse != index then failCode "OWNER" s!"owner index mismatch for {encodeName name}"
  return owner

/-- Read the exact resolved private object, all raw parts and the actual import env. -/
def loadExact (request : ModuleRequest) : IO LoadedModule := do
  let levels := request.oleanParts.map (·.level)
  if levels != #["exported"] && levels != #["exported", "server", "private"] then
    failCode "PART_LEVELS" "expected exported or exported/server/private in that order"
  let irLevels := request.irParts.map (·.level)
  if !irLevels.isEmpty && irLevels != #["ir.sig", "ir"] then
    failCode "PART_LEVELS" "expected both ordered IR parts"
  initSearchPath (← getBuildDir)
  let base := request.oleanParts[0]!.path
  let resolved ← try findOLean request.module catch error =>
    failCode "RESOLVED_PATH" error.toString
  if resolved.normalize != base then
    failCode "RESOLVED_PATH" s!"{encodeName request.module}: resolved OLean differs from request"
  exactPartPaths base request.oleanParts false
  exactPartPaths base request.irParts true
  let raw ← readParts request.oleanParts
  if raw.size != request.oleanParts.size then
    failCode "OBJECT_READ" "OLean part count differs"
  let some exported := raw[0]? |
    failCode "OBJECT_READ" "missing exported OLean data"
  let isModule := exported.1.isModule
  if isModule && levels != #["exported", "server", "private"] then
    failCode "MISSING_PART" "module-system object requires all three OLean parts"
  if !isModule && levels != #["exported"] then
    failCode "PART_LEVELS" "legacy object must have only exported data"
  if isModule && irLevels != #["ir.sig", "ir"] then
    failCode "MISSING_PART" "module-system object requires both IR parts"
  let parts := raw.mapIdx fun i part => (request.oleanParts[i]!.level, part.1)
  let owned ← fromAudit (validateParts parts)
  let rawIR ← readParts request.irParts
  let irParts := rawIR.mapIdx fun i part => (request.irParts[i]!.level, part.1)
  let env ← try
    importModules #[{ module := request.module }] {}
      (trustLevel := 0) (loadExts := false) (level := .private)
    catch error => failCode "OBJECT_READ" error.toString
  if env.header.trustLevel != 0 then failCode "OBJECT_READ" "import trust level differs"
  for ci in parts[parts.size - 1]!.2.constants do
    let owner ← ownerOf env ci.name
    if owner != request.module then
      failCode "OWNER" s!"owned declaration has another actual owner: {encodeName ci.name}"
    let some imported := env.find? ci.name |
      failCode "MISSING_CONSTANT" (encodeName ci.name)
    if imported.name != ci.name then
      failCode "OWNER" s!"import identity mismatch for {encodeName ci.name}"
  return {
    request, parts, regions := raw.map (·.2),
    irParts, irRegions := rawIR.map (·.2), owned, env
  }

structure DeclarationObservation where
  name : Name
  owner : Name
  isPrivate : Bool
  isProp : Bool
  kind : String
  isUnsafe : Bool
  isPartial : Bool
  typeDisplay : String

private def constantKind : ConstantInfo → String
  | .thmInfo _ => "theorem"
  | .defnInfo _ => "definition"
  | .opaqueInfo _ => "opaque"
  | .axiomInfo _ => "axiom"
  | .inductInfo _ => "inductive"
  | .ctorInfo _ => "constructor"
  | .recInfo _ => "recursor"
  | .quotInfo _ => "quotient"

/-- Actual imported-environment judgment, with open/malformed types refused first. -/
def observeConstant (loaded : LoadedModule) (ci : ConstantInfo) :
    IO DeclarationObservation := do
  match requireIdentity ci.name with
  | .error message => failCode "MALFORMED_CONSTANT" message
  | .ok _ => pure ()
  if ci.type.hasLooseBVars || ci.type.hasFVar || ci.type.hasMVar then
    failCode "MALFORMED_CONSTANT" s!"open type for {encodeName ci.name}"
  let owner ← ownerOf loaded.env ci.name
  if owner != loaded.request.module || !loaded.owned.contains ci.name then
    failCode "OWNER" s!"observation is not owned: {encodeName ci.name}"
  let ((isProp, typeDisplay), _, _) ← try
    (do
      let isProp ← Meta.isProp ci.type
      let display ← PrettyPrinter.ppExpr ci.type
      pure (isProp, display.pretty)).toIO
      { fileName := "<sdga-component>", fileMap := default }
      { env := loaded.env }
    catch error =>
      failCode "MALFORMED_CONSTANT" s!"{encodeName ci.name}: {error}"
  if typeDisplay.isEmpty then
    failCode "MALFORMED_CONSTANT" s!"empty type diagnostic for {encodeName ci.name}"
  let exported := loaded.parts[0]!.2.constNames
  return {
    name := ci.name, owner, isPrivate := isPrivateName ci.name || !exported.contains ci.name,
    isProp, kind := constantKind ci, isUnsafe := ci.isUnsafe, isPartial := ci.isPartial,
    typeDisplay
  }

/- This is a strict observed dependency closure, not Task B policy qualification.
   Start with the pinned CI type/value collector, then include explicit mutual,
   constructor and recursor rule links so a missing link cannot disappear. -/
private def dependencies (ci : ConstantInfo) : Array Name := Id.run do
  let mut result : Array Name := #[]
  for name in ci.getUsedConstantsAsSet do
    result := result.push name
  match ci with
  | .inductInfo value =>
    for name in value.all ++ value.ctors do result := result.push name
  | .ctorInfo value => result := result.push value.induct
  | .recInfo value =>
    for name in value.all do result := result.push name
    for rule in value.rules do
      result := result.push rule.ctor
      for name in rule.rhs.getUsedConstants do result := result.push name
  | _ => pure ()
  return result

private partial def closeAxioms (lookup : Name → Option ConstantInfo)
    (pending : List Name) (seen : Std.HashSet Name) (axioms : Array Name) :
    Except AuditError (Array Name) := do
  match pending with
  | [] => return axioms
  | name :: rest =>
    if seen.contains name then return ← closeAxioms lookup rest seen axioms
    let some ci := lookup name | throw (.missingConstant name)
    if ci.name != name then throw (.malformedConstant name "lookup returned a different identity")
    let seen := seen.insert name
    let axioms := match ci with
      | .axiomInfo _ => axioms.push name
      | _ => axioms
    closeAxioms lookup ((dependencies ci).toList ++ rest) seen axioms

/-- Every reached lookup is mandatory, including roots, values and transitive links. -/
def strictAxioms (lookup : Name → Option ConstantInfo) (roots : Array Name) :
    Except AuditError (Array Name) :=
  closeAxioms lookup roots.toList {} #[]

private def nameArray (names : Array Name) : Json :=
  .arr (names.map (Json.str ∘ encodeName))

private def emit (record : Json) : IO Unit := do
  let stdout ← IO.getStdout
  stdout.putStrLn record.compress
  stdout.flush

/-- Emit exactly Task 1's header/module/declaration/end fields, incrementally. -/
def inventory (loaded : LoadedModule) : IO Unit := do
  emit <| .mkObj [
    ("record", .str "header"), ("schema", toJson (1 : Nat)),
    ("scope_id", .str loaded.request.scopeId)
  ]
  let final := loaded.parts[loaded.parts.size - 1]!.2
  let mut extras : Array Name := #[]
  let mut extraSeen : Std.HashSet Name := {}
  let mut objectParts : Array Json := #[]
  for h : i in [:loaded.parts.size] do
    let (level, part) := loaded.parts[i]
    let requested := loaded.request.oleanParts[i]!
    objectParts := objectParts.push <| .mkObj [
      ("level", .str level), ("sha256", .str requested.sha256),
      ("const_names", nameArray part.constNames),
      ("constant_names", nameArray (part.constants.map (·.name)))
    ]
    for name in part.extraConstNames do
      match requireIdentity name with
      | .error message => failCode "MALFORMED_CONSTANT" message
      | .ok _ => pure ()
      if !extraSeen.contains name then
        extraSeen := extraSeen.insert name
        extras := extras.push name
  emit <| .mkObj [
    ("record", .str "module"), ("name", .str (encodeName loaded.request.module)),
    ("const_count", toJson final.constants.size), ("object_parts", .arr objectParts),
    ("extra_const_names", nameArray extras)
  ]
  let lookup := fun name => loaded.owned[name]? <|> loaded.env.find? name
  let finalDigest := loaded.request.oleanParts[loaded.request.oleanParts.size - 1]!.sha256
  let mut axiomUnion : Array Name := #[]
  let mut axiomSeen : Std.HashSet Name := {}
  for ci in final.constants do
    let row ← observeConstant loaded ci
    let axioms ← fromAudit (strictAxioms lookup #[ci.name])
    for name in axioms do
      if !axiomSeen.contains name then
        axiomSeen := axiomSeen.insert name
        axiomUnion := axiomUnion.push name
    emit <| .mkObj [
      ("record", .str "declaration"), ("module", .str (encodeName row.owner)),
      ("name", .str (encodeName row.name)), ("kind", .str row.kind),
      ("is_prop", .bool row.isProp), ("is_private", .bool row.isPrivate),
      ("unsafe", .bool row.isUnsafe), ("partial", .bool row.isPartial),
      ("type_display", .str row.typeDisplay), ("object_part_sha256", .str finalDigest),
      ("axioms", nameArray axioms)
    ]
  emit <| .mkObj [
    ("record", .str "end"), ("module_count", toJson (1 : Nat)),
    ("declaration_count", toJson final.constants.size), ("axioms", nameArray axiomUnion)
  ]

/-- Task C is intentionally absent; this entry point cannot report replay success. -/
def replayOwned (_base : Kernel.Environment) (_owned : Std.HashMap Name ConstantInfo) :
    IO Kernel.Environment :=
  failCode "NOT_IMPLEMENTED" "Task C kernel replay is not implemented or qualified"

def runReplay (_loaded : LoadedModule) : IO Unit :=
  failCode "NOT_IMPLEMENTED" "Task C kernel replay is not implemented or qualified"

private def diagnostic (error : IO.Error) : IO Unit := do
  let message := match error with
    | .userError message =>
      if message.startsWith "SDGA_ERROR " then message
      else s!"SDGA_ERROR INTERNAL: {message}"
    | _ => s!"SDGA_ERROR INTERNAL: {error}"
  let stderr ← IO.getStderr
  stderr.putStrLn (message.replace "\n" " ")
  stderr.flush

def runCLI (args : List String) : IO UInt32 := do
  try
    match args with
    | ["inventory", path] => inventory (← loadExact (← readRequest (System.FilePath.mk path)))
    | ["replay", path] => runReplay (← loadExact (← readRequest (System.FilePath.mk path)))
    | _ => failCode "CLI_MODE" "expected inventory request.json or replay request.json"
    return 0
  catch error =>
    diagnostic error
    return 1

end SDGA

def main (args : List String) : IO UInt32 :=
  SDGA.runCLI args
