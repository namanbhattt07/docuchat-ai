import type { TemplateDraft, TemplateItem } from "./api";

// Group 7 prompt templates: request building and draft validation. The limits
// mirror backend services/templates.py so the form can explain a problem
// before a round trip; the server remains the authority.

export const TEMPLATE_LIMITS = { name: 80, description: 300, instruction: 2000, output_format: 1000 } as const;

export const EMPTY_DRAFT: TemplateDraft = { name: "", description: "", instruction: "", output_format: "" };

export const draftFromTemplate = (template: Pick<TemplateItem, "name" | "description" | "instruction" | "output_format">): TemplateDraft => ({
  name: template.name,
  description: template.description ?? "",
  instruction: template.instruction,
  output_format: template.output_format ?? "",
});

/** Section headings named by an output format ("## Facts" -> "Facts"). */
export const templateSections = (template: Pick<TemplateItem, "output_format">): string[] =>
  (template.output_format ?? "")
    .split("\n")
    .map(line => line.trim())
    .filter(line => line.startsWith("#"))
    .map(line => line.replace(/^#+\s*/, "").trim())
    .filter(Boolean);

/** The first problem with a draft, or null if it can be saved. */
export function validateTemplateDraft(draft: TemplateDraft): string | null {
  const name = draft.name.trim();
  if (!name) return "Give the template a name.";
  if (name.length > TEMPLATE_LIMITS.name) return `Name must be ${TEMPLATE_LIMITS.name} characters or fewer.`;
  if (draft.description.trim().length > TEMPLATE_LIMITS.description) return `Description must be ${TEMPLATE_LIMITS.description} characters or fewer.`;
  const instruction = draft.instruction.trim();
  if (!instruction) return "Write the instruction the template should follow.";
  if (instruction.length > TEMPLATE_LIMITS.instruction) return `Instruction must be ${TEMPLATE_LIMITS.instruction} characters or fewer.`;
  if (draft.output_format.trim().length > TEMPLATE_LIMITS.output_format) return `Output format must be ${TEMPLATE_LIMITS.output_format} characters or fewer.`;
  return null;
}

/** What "Apply" sends to /chat, and what appears as the user's message in the
 * thread. A focus note is folded in; without one the message says what ran. */
export function buildTemplateRequest(template: Pick<TemplateItem, "id" | "name">, focus: string) {
  const note = focus.trim();
  const question = note ? `${template.name}: ${note}` : `Apply the “${template.name}” template`;
  return { question, templateId: template.id };
}
