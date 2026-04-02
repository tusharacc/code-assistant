# Requirements: [Feature Name]

<!--
  USAGE:  ca --pipeline --req-file requirements.md
  GUIDE:  Fill in every section. The more precise you are, the less the implementer
          has to guess — and the fewer fix/recheck cycles the pipeline runs.

  RULE 1: Project context is not optional. The implementer cannot read your mind.
           List every file that is touched, why it exists, and what its current
           interface looks like.

  RULE 2: Write acceptance criteria as automated test statements, not wishes.
           Bad:  "The app should be fast."
           Good: "Calling process('hello') returns a string in < 50 ms."

  RULE 3: If you are modifying existing files, paste the relevant function
           signatures and current behaviour. The pipeline reads files — but the
           architect plans before reading, so give it a head-start.

  Delete all HTML comments before running ca.
-->

---

## 1. Project context

<!--
  One paragraph. What is the project? What tech stack? What problem does it solve?
  Include the runtime environment (OS, Python version, language version, etc.).
-->

**Project:** [Name and one-line description]
**Stack:** [e.g. Python 3.12 · asyncio · WebSocket (websockets 12) · SQLite · Electron 30]
**Entry point:** [e.g. `python backend/server.py` / `npm start`]
**Key constraint:** [e.g. macOS only / no network calls except localhost / must stay under 100ms latency]


---

## 2. Current relevant files

<!--
  List every file the implementer is likely to touch or depend on.
  For each: path, one-line purpose, and the public interface (function signatures,
  class names, dataclasses). Paste signatures — do not summarise vaguely.

  If the file must NOT be modified, say so explicitly.
-->

| File | Purpose | Interface (relevant parts) |
|------|---------|---------------------------|
| `backend/server.py` | asyncio WebSocket server | `handle_client(ws)`, `camera_loop(ws, cam, state)`, `analysis_loop(ws, cam, analyzer, state)` |
| `backend/journal.py` | SQLite CRUD | `init_db(path)`, `start_session()→int`, `log_event(sid, exercise, severity, issue, tip)`, `end_session(sid, summary)` |
| `backend/camera.py` | OpenCV + MediaPipe | `Camera.get_frame()→ndarray`, `get_annotated_frame(props)→ndarray`, `get_landmarks(frame)→list\|None`, `camera.cap` (cv2.VideoCapture) |
| *(add rows as needed)* | | |

**Files that MUST NOT be modified:** *(list any — or "none")*


---

## 3. New files to create

<!--
  List each new file that must be created from scratch.
  Give the path, purpose, and full class/function specification.
  Use exact signatures. "Roughly like X" is not enough.
-->

### `[path/to/new_file.py]`

**Purpose:** [one sentence]

```
ClassName(arg1: type, arg2: type = default) -> None
  - attribute1: type — what it holds
  - attribute2: type — what it holds

method_name(self, arg: type) -> return_type
  - Does X.
  - Raises ValueError if Y.
  - Side effect: Z.
```

*(repeat for each new file)*


---

## 4. Changes to existing files

<!--
  For each existing file being modified, list the exact changes needed.
  Do NOT write "refactor file X" — describe the precise diffs:
    - What function/class is added
    - What function/class is removed or renamed
    - What new imports are needed
    - What new state is added to existing objects
  Use before/after code snippets where the change is subtle.
-->

### `[path/to/existing_file.py]`

**Changes:**
1. Add import: `from new_module import NewClass`
2. In `existing_function()`, after line that does X, add:
   ```python
   state["new_key"] = []
   ```
3. Replace call to `old_method()` with `new_method()` — same return type, new
   parameter `extra_arg: bool = False`.
4. Add new function at module level:
   ```python
   def helper(x: int) -> str: ...   # full signature
   ```

*(repeat for each modified file)*


---

## 5. WebSocket / API / protocol changes

<!--
  If the feature changes a message protocol, REST API, or IPC interface, describe
  every new or modified message/endpoint here.
  Paste the exact JSON shape. The implementer copies this verbatim.
-->

**New outbound message (Python → client):**
```json
{
  "type": "session_summary",
  "exercises": ["Push-Up", "Squat"],
  "total_reps": 24
}
```

**New inbound message (client → Python):** *(if any)*
```json
{
  "type": "start_session"
}
```

*(delete this section if no protocol changes)*


---

## 6. Database / storage changes

<!--
  Schema changes, new tables, migrations.
  If modifying an existing schema, include the safe migration pattern required
  (e.g. ALTER TABLE ... ADD COLUMN IF NOT EXISTS vs CREATE TABLE IF NOT EXISTS).
-->

```sql
-- New column on existing table (safe migration required):
ALTER TABLE sessions ADD COLUMN video_path TEXT;

-- New table:
CREATE TABLE IF NOT EXISTS recordings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER REFERENCES sessions(id),
    path        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
```

*(delete this section if no DB changes)*


---

## 7. Frontend / UI changes

<!--
  HTML element additions, CSS changes, JavaScript event handlers.
  Reference element IDs and DOM position exactly:
  "After the closing </table> tag of #journal-table, add..."
-->

**index.html** — add after `</table>`:
```html
<div id="session-summary" style="display:none;">
  <h3>Session Summary</h3>
  <p id="summary-exercises">—</p>
  <p id="summary-reps">—</p>
</div>
```

**app.js** — add case in WebSocket switch:
```javascript
case 'session_summary': {
  // implementation
  break;
}
```

*(delete this section if no frontend changes)*


---

## 8. Implementation order

<!--
  List files in the order the implementer should create/modify them.
  The order must respect import dependencies — if file B imports from file A,
  file A must be written first.
  This prevents the implementer from calling a function that doesn't exist yet.
-->

1. `[file with no new dependencies]` — no imports from new code
2. `[file that imports from step 1]`
3. `[file that imports from steps 1 and 2]`
4. *(continue)*


---

## 9. Acceptance criteria

<!--
  These are the pass/fail gates the tester phase checks.
  Write each criterion as a concrete, automatable statement.
  The tester will attempt to verify each one — if you write vague criteria,
  the tester marks them MANUAL (unverified).

  Format: numbered list, one assertion per line.
  Prefer: "Calling X returns Y" over "X should work correctly."
-->

1. `python -m py_compile [every modified Python file]` exits 0.
2. `[new_file.py]` exists and the class `[ClassName]` can be imported without error.
3. Calling `[function()]` with input `[X]` returns `[Y]` (verify with `python -c "..."`).
4. After [action], [observable state] exists / equals [expected value].
5. If `[error condition]`, no exception is raised — the function returns `[fallback]`.
6. Existing tests in `[test_file.py]` still pass (`pytest [test_file.py] -v`).
7. *(add more — aim for 8–12 criteria)*


---

## 10. What NOT to change

<!--
  Explicit negative scope. List things the implementer must leave alone.
  This prevents "while I'm in here" changes that break other things.
-->

- Do not modify `[file.py]` — it is used by other features not covered here.
- Do not change the `[ClassName]` constructor signature — it has callers outside this scope.
- Do not add new dependencies to `pyproject.toml` unless listed in section 3/4.
- Do not change the existing WebSocket message format for `[type]` messages.


---

## 11. Notes and gotchas

<!--
  Known tricky bits. Paste these proactively to save one Q&A round-trip.
  Examples: version incompatibilities, macOS-only APIs, thread-safety issues,
  async/sync boundaries, specific error handling patterns the codebase uses.
-->

- **[Library X] quirk:** [description]. Correct usage: `[snippet]`.
- **Thread safety:** `[module]` is called from an executor thread — do not call
  asyncio APIs directly inside its methods.
- **macOS only:** `[API]` requires macOS 13+. Do not add a cross-platform fallback.
- **Existing pattern to follow:** All modules use `from logger import get_logger`
  and `log = get_logger(__name__)`. Do not use stdlib `logging.getLogger` directly.
