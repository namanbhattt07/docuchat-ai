"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { createTemplate, deleteTemplate, listTemplates, TemplateDraft, TemplateItem, updateTemplate } from "../../lib/api";
import { draftFromTemplate, EMPTY_DRAFT, TEMPLATE_LIMITS, templateSections, validateTemplateDraft } from "../../lib/templates";
import { PencilIcon, PlusIcon, TrashIcon } from "./icons";

// Group 7 prompt templates: pick a template, see what it will produce, and
// apply it to the current document/collection scope. Custom templates are a
// name + description + instruction (+ optional output format) -- no template
// language. Applying goes through /chat (mode "template"), so grounding,
// collections and citations behave exactly as they do in Chat.

type Props = {
  scopeLabel: string;
  // A reason applying is unavailable right now (nothing selected, etc.).
  disabledReason: string | null;
  busy: boolean;
  onApply: (template: TemplateItem, focus: string) => void;
};

type Editing = { id: string | null; draft: TemplateDraft } | null;

export default function TemplatePanel({ scopeLabel, disabledReason, busy, onApply }: Props) {
  const [templates, setTemplates] = useState<TemplateItem[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [focus, setFocus] = useState("");
  const [loadError, setLoadError] = useState("");
  const [managing, setManaging] = useState(false);
  const [editing, setEditing] = useState<Editing>(null);
  const [formError, setFormError] = useState("");
  const [saving, setSaving] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

  const reload = useCallback(async (preferId?: string) => {
    try {
      const items = await listTemplates();
      setTemplates(items);
      setSelectedId(current => (preferId && items.some(item => item.id === preferId) ? preferId : items.some(item => item.id === current) ? current : items[0]?.id ?? ""));
      setLoadError("");
    } catch (err) { setLoadError(err instanceof Error ? err.message : "Could not load templates."); }
  }, []);
  useEffect(() => { void reload(); }, [reload]);

  const selected = templates.find(item => item.id === selectedId) ?? null;
  const builtins = templates.filter(item => item.builtin);
  const custom = templates.filter(item => !item.builtin);
  const sections = useMemo(() => (selected ? templateSections(selected) : []), [selected]);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!editing) return;
    const problem = validateTemplateDraft(editing.draft);
    if (problem) { setFormError(problem); return; }
    setSaving(true); setFormError("");
    try {
      const saved = editing.id ? await updateTemplate(editing.id, editing.draft) : await createTemplate(editing.draft);
      setEditing(null);
      await reload(saved.id);
    } catch (err) { setFormError(err instanceof Error ? err.message : "Could not save the template."); }
    finally { setSaving(false); }
  }
  async function remove(id: string) {
    if (confirmDeleteId !== id) { setConfirmDeleteId(id); return; }
    setConfirmDeleteId(null);
    try { await deleteTemplate(id); await reload(); }
    catch (err) { setFormError(err instanceof Error ? err.message : "Could not delete the template."); }
  }
  function setField(field: keyof TemplateDraft, value: string) {
    setEditing(current => (current ? { ...current, draft: { ...current.draft, [field]: value } } : current));
  }
  const startNew = (draft: TemplateDraft = EMPTY_DRAFT) => { setEditing({ id: null, draft }); setFormError(""); setManaging(true); };

  return (
    <div className="template-panel">
      <div className="template-picker">
        <label className="exam-field is-grow">
          <span className="exam-field-label">Template</span>
          <select value={selectedId} onChange={event => setSelectedId(event.target.value)} aria-label="Template" disabled={templates.length === 0}>
            <optgroup label="Built-in">{builtins.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</optgroup>
            {custom.length > 0 && <optgroup label="My templates">{custom.map(item => <option key={item.id} value={item.id}>{item.name}</option>)}</optgroup>}
          </select>
        </label>
        <button type="button" className="template-manage" aria-expanded={managing} onClick={() => { setManaging(current => !current); setEditing(null); setFormError(""); }}>
          {managing ? "Done" : "Manage"}
        </button>
      </div>

      {loadError && <div className="error">{loadError}</div>}

      {selected && (
        <>
          <p className="template-description">{selected.description || "Custom template."}</p>
          {sections.length > 0 && <ul className="template-sections" aria-label="Sections this template produces">{sections.map(name => <li key={name}>{name}</li>)}</ul>}
          <p className="exam-scope">Applies to: <b>{scopeLabel}</b></p>
          <label className="exam-field">
            <span className="exam-field-label">Focus (optional)</span>
            <input value={focus} onChange={event => setFocus(event.target.value)} placeholder="e.g. focus on the methodology" maxLength={300} disabled={busy} />
          </label>
          <button
            type="button"
            className="exam-primary template-apply"
            disabled={busy || Boolean(disabledReason)}
            onClick={() => { onApply(selected, focus); setFocus(""); }}
          >
            {busy ? "Applying…" : `Apply “${selected.name}”`}
          </button>
          {disabledReason && <p className="exam-hint">{disabledReason}</p>}
        </>
      )}

      {managing && (
        <div className="template-manager">
          <div className="template-manager-head">
            <b>My templates</b>
            <div>
              {selected && selected.builtin && (
                <button type="button" className="exam-link" onClick={() => startNew({ ...draftFromTemplate(selected), name: `${selected.name} (custom)` })}>Customize selected</button>
              )}
              <button type="button" className="exam-secondary template-new" onClick={() => startNew()}><PlusIcon size={12} /> New</button>
            </div>
          </div>
          {custom.length === 0 && !editing && <p className="exam-hint">No custom templates yet. Built-in templates can’t be edited, but you can customize a copy.</p>}
          <ul className="template-custom-list">
            {custom.map(item => (
              <li key={item.id}>
                <span title={item.description}>{item.name}</span>
                <button type="button" onClick={() => { setEditing({ id: item.id, draft: draftFromTemplate(item) }); setFormError(""); }} aria-label={`Edit ${item.name}`}><PencilIcon size={12} /></button>
                <button type="button" className={confirmDeleteId === item.id ? "is-confirming" : ""} onClick={() => void remove(item.id)} aria-label={confirmDeleteId === item.id ? `Confirm delete ${item.name}` : `Delete ${item.name}`}>
                  {confirmDeleteId === item.id ? "Confirm" : <TrashIcon size={12} />}
                </button>
              </li>
            ))}
          </ul>
          {editing && (
            <form className="template-form" onSubmit={save}>
              <label className="exam-field"><span className="exam-field-label">Name</span>
                <input value={editing.draft.name} onChange={event => setField("name", event.target.value)} maxLength={TEMPLATE_LIMITS.name} required />
              </label>
              <label className="exam-field"><span className="exam-field-label">Description</span>
                <input value={editing.draft.description} onChange={event => setField("description", event.target.value)} maxLength={TEMPLATE_LIMITS.description} />
              </label>
              <label className="exam-field"><span className="exam-field-label">Instruction</span>
                <textarea rows={4} value={editing.draft.instruction} onChange={event => setField("instruction", event.target.value)} maxLength={TEMPLATE_LIMITS.instruction} placeholder="What should the model produce from the document?" required />
              </label>
              <label className="exam-field"><span className="exam-field-label">Output format (optional, one “## Heading” per line)</span>
                <textarea rows={3} value={editing.draft.output_format} onChange={event => setField("output_format", event.target.value)} maxLength={TEMPLATE_LIMITS.output_format} placeholder={"## Summary\n## Risks"} />
              </label>
              {formError && <div className="error">{formError}</div>}
              <div className="template-form-actions">
                <button type="submit" className="exam-primary" disabled={saving}>{saving ? "Saving…" : editing.id ? "Save changes" : "Create template"}</button>
                <button type="button" className="exam-secondary" onClick={() => { setEditing(null); setFormError(""); }}>Cancel</button>
              </div>
            </form>
          )}
          {!editing && formError && <div className="error">{formError}</div>}
        </div>
      )}
    </div>
  );
}
