"""Compact owner digest. Obligations belong to the owner across every session."""


async def render_digest(self, body):
    hands = await self.all("SELECT h.*, t.state AS task_state, t.obligation_id, t.owner_spoke_since "
                          "FROM hands h JOIN tasks t ON t.id=h.task_id "
                          "WHERE t.state IN ('open','waiting') OR h.state IN ('reserved','dispatched','running','checking','integrating') "
                          "OR t.obligation_id IN (SELECT id FROM obligations WHERE state IN ('open','result_ready','reviewed','accepted')) "
                          "ORDER BY h.created_at DESC, h.rowid DESC")
    obligations = await self.all("SELECT * FROM obligations WHERE state IN ('open','result_ready','reviewed','accepted')")
    questions = await self.all("SELECT * FROM questions WHERE state='open'")
    permits = await self.all("SELECT * FROM permits WHERE state IN ('waiting','granted')")
    forgotten = {hand['task_id'] for hand in hands if self.parent_forgotten(hand)}
    hands = [hand for hand in hands if hand['task_id'] not in forgotten]
    obligations = [item for item in obligations if item['task_id'] not in forgotten]
    items, lines, seen = [], [], set()
    active = sum(h["state"] in {"reserved", "dispatched", "running", "checking", "integrating"} for h in hands)
    ready = sum(o["state"] == "result_ready" for o in obligations)
    overdue = set()
    for obligation in obligations:
        if self.obligation_overdue(obligation, await self.get_task(obligation["task_id"])):
            overdue.add(obligation["id"])
    for hand in hands:
        text = f"{hand['name']} ({hand['kind']}, {hand['role'] or 'unrouted'}): {hand['state']}"
        if hand["outcome"] and hand["outcome"] != hand["state"]:
            text += "; " + hand["outcome"]
        evidence = hand["evidence"]
        if evidence:
            text += f"; {evidence['level']} {evidence['scope']['kind']}"
        if hand["owner_spoke_since"]:
            text += "; Owner wrote since this started"
        if hand["obligation_id"] in overdue:
            text += "; answer overdue"
        question = next((q for q in questions if q["hand_id"] == hand["id"]), None)
        if question:
            text += "; asked: " + (question["text"] or "")
        permit = next((p for p in permits if p["hand_id"] == hand["id"]), None)
        if permit:
            text += "; awaiting approval"
        if hand["state"] == "unknown_effect":
            effect = await self.one("SELECT tool, call_hash FROM effects WHERE hand_id=? AND done_at IS NULL ORDER BY intent_at DESC LIMIT 1", (hand["id"],))
            if effect:
                text += f"; unconfirmed {effect['tool']} ({effect['call_hash'][:12]})"
            probe = hand["decisions"].get("probe")
            if probe:
                details = probe["scope"].get("details", {})
                text += "; probe: " + str(details.get("output") or details.get("error") or "no output").replace("\n", " ")
        items.append({"task_id": hand["task_id"], "hand_id": hand["id"], "obligation_id": hand["obligation_id"],
                      "question_id": question["id"] if question else None, "permit_id": permit["id"] if permit else None,
                      "evidence": evidence, "text": text})
        seen.add(hand["task_id"])
        lines.append(text)
    for obligation in obligations:
        if obligation["task_id"] in seen:
            continue
        text = f"{obligation['task_id']}: {obligation['state']}; Owner has not been told"
        if obligation["id"] in overdue:
            text += "; answer overdue"
        items.append({"task_id": obligation["task_id"], "obligation_id": obligation["id"], "text": text})
        lines.append(text)
    if not items:
        return {"text": "", "items": [], "cursor": body.cursor}
    limit = min(900, self.config["digest"]["max_chars"])
    text = f"Your hands: {active} active, {ready} results ready. Owner is owed {len(obligations)} answers."
    if overdue:
        text += f" {len(overdue)} overdue."
    for n, line in enumerate(lines):
        remaining = len(lines) - n
        suffix = f"\n{remaining} more; /hands."
        candidate = text + "\n- " + line
        if len(candidate) + len(suffix) > limit:
            text += suffix
            break
        text = candidate
    return {"text": text[:limit], "items": items, "cursor": self.now()}


async def digest(body):
    from .controller import get_controller
    return await (await get_controller()).digest(body)
