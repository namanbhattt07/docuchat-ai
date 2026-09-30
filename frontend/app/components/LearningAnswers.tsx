import { BrainstormMeta, TutorMeta, TutorSectionKind } from "../../lib/api";
import MarkdownContent from "./MarkdownContent";

// Group 6: structured renderings of tutor and brainstorm replies. The labels
// here are the UI half of the grounding contract -- document-derived content
// and assistant-generated content must never look alike.

const SECTION_TAGS: Record<TutorSectionKind, string> = {
  document: "From your document",
  tutor: "Tutor-generated",
  example: "Tutor-generated",
  analogy: "Tutor-generated",
  question: "Your turn",
  notice: "Note",
};

export function TutorAnswer({ tutor }: { tutor: TutorMeta }) {
  return (
    <div className="tutor-answer">
      <div className="learning-answer-head">
        <span className="mode-badge mode-badge-tutor">Tutor{tutor.action_label ? ` · ${tutor.action_label}` : ""}</span>
        {tutor.topic && <span className="learning-topic" title="Current tutor topic">{tutor.topic}</span>}
      </div>
      {tutor.sections.map((section, index) => (
        <section key={`${section.kind}-${index}`} className={`tutor-section kind-${section.kind}`}>
          <header>
            <span className="tutor-section-title">{section.title}</span>
            <span className="tutor-section-tag">{SECTION_TAGS[section.kind]}</span>
          </header>
          {section.note && <p className="tutor-section-note">{section.note}</p>}
          <MarkdownContent text={section.content} />
        </section>
      ))}
    </div>
  );
}

const IDEA_KIND_LABELS: Record<string, string> = {
  project: "Project",
  application: "Application",
  research_question: "Research question",
  hypothesis: "Hypothesis",
  extension: "Extension",
  idea: "Idea",
};

export function BrainstormAnswer({ brainstorm }: { brainstorm: BrainstormMeta }) {
  return (
    <div className="brainstorm-answer">
      <div className="learning-answer-head"><span className="mode-badge mode-badge-brainstorm">Brainstorm</span></div>
      {brainstorm.evidence.length > 0 && (
        <section className="brainstorm-block is-evidence">
          <header>
            <span className="tutor-section-title">Document evidence</span>
            <span className="tutor-section-tag">From your document</span>
          </header>
          <ul>
            {brainstorm.evidence.map((item, index) => (
              <li key={index}>{item.point} <span className="cite-markers">{item.source_ids.map(id => `[${id}]`).join(" ")}</span></li>
            ))}
          </ul>
        </section>
      )}
      {brainstorm.ideas.length > 0 && (
        <section className="brainstorm-block is-ideas">
          <header>
            <span className="tutor-section-title">Generated ideas</span>
            <span className="tutor-section-tag">AI-generated · not from the document</span>
          </header>
          <ul className="idea-list">
            {brainstorm.ideas.map((idea, index) => (
              <li key={index} className="idea-card">
                <div className="idea-card-head">
                  <strong>{idea.title}</strong>
                  <span className="idea-kind">{IDEA_KIND_LABELS[idea.kind] ?? "Idea"}</span>
                </div>
                {idea.description && <p>{idea.description}</p>}
                {idea.builds_on.length > 0 && <p className="idea-builds-on">Builds on evidence {idea.builds_on.map(id => `[${id}]`).join(" ")}</p>}
              </li>
            ))}
          </ul>
        </section>
      )}
      {brainstorm.notice && <p className={`learning-notice ${brainstorm.insufficient_evidence ? "is-warning" : ""}`}>{brainstorm.notice}</p>}
    </div>
  );
}
