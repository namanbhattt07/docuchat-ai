"use client";

import { Dispatch, useState } from "react";
import {
  AnswerEvaluation, Citation, evaluateExamAnswer, evaluateWriting, ExamQuestion, generateQuestions, LearningScope,
  RequestedQuestionType, WritingFeedbackResult,
} from "../../lib/api";
import {
  correctAnswerText, currentQuestion, ExamSession, examSummary, MAX_QUESTIONS, QUESTION_TYPE_LABELS, questionText,
  StudyAction, StudyState, StudyView,
} from "../../lib/exam";
import type { TutorSelection } from "../../lib/tutor";
import { ExternalLinkIcon } from "./icons";
import MarkdownContent from "./MarkdownContent";

type Props = {
  state: StudyState;
  dispatch: Dispatch<StudyAction>;
  // null when there's nothing to study from (e.g. a collection with no
  // documents checked) -- every action is disabled with a reason instead.
  scope: LearningScope | null;
  scopeLabel: string;
  selection: TutorSelection | null;
  onOpenCitation: (citation: Citation) => void;
};

const VIEWS: { id: StudyView; label: string }[] = [
  { id: "exam", label: "Practice exam" },
  { id: "bank", label: "Question bank" },
  { id: "writing", label: "Writing feedback" },
];
const TYPE_OPTIONS: { id: RequestedQuestionType; label: string }[] = [
  { id: "mcq", label: "MCQ" },
  { id: "true_false", label: "True / False" },
  { id: "short_answer", label: "Short answer" },
  { id: "mixed", label: "Mixed" },
];
const VERDICT_LABELS: Record<AnswerEvaluation["verdict"], string> = { correct: "Correct", partial: "Partially correct", incorrect: "Incorrect" };
const RATING_LABELS: Record<string, string> = { strong: "Strong", adequate: "Adequate", needs_work: "Needs work", not_assessed: "Not assessed" };
const LETTERS = "ABCDEF";

function SourceList({ citations, onOpen, label = "Source" }: { citations: Citation[]; onOpen: (citation: Citation) => void; label?: string }) {
  if (citations.length === 0) return null;
  return (
    <div className="exam-sources">
      <p className="evidence-label">{label}{citations.length > 1 ? "s" : ""}</p>
      {citations.map(citation => (
        <details key={`${citation.document_id}-${citation.index}`}>
          <summary><b>[{citation.index}]</b> {citation.filename} <span>Page {citation.page_number}</span></summary>
          <p>{citation.excerpt}</p>
          <button type="button" className="open-in-pdf" onClick={() => onOpen(citation)}><ExternalLinkIcon size={13} /> Open in PDF</button>
        </details>
      ))}
    </div>
  );
}

function GroundingBadge({ question }: { question: ExamQuestion }) {
  return question.grounded
    ? <span className="exam-badge is-grounded">From your document</span>
    : <span className="exam-badge is-general">General knowledge</span>;
}

// ---------------------------------------------------------------------------
// Setup (shared by Practice exam and Question bank)
// ---------------------------------------------------------------------------

function SetupForm({ state, dispatch, selection, busy, disabledReason, scopeLabel, submitLabel, onSubmit }: {
  state: StudyState; dispatch: Dispatch<StudyAction>; selection: TutorSelection | null; busy: boolean; disabledReason: string | null;
  scopeLabel: string; submitLabel: string; onSubmit: () => void;
}) {
  const { config } = state;
  return (
    <form className="exam-setup" onSubmit={event => { event.preventDefault(); onSubmit(); }}>
      <p className="exam-scope">Questions come from: <b>{config.useSelection && selection ? `selected passage (p. ${selection.page})` : scopeLabel}</b></p>
      <div className="exam-field">
        <span className="exam-field-label">Question type</span>
        <div className="segmented" role="radiogroup" aria-label="Question type">
          {TYPE_OPTIONS.map(option => (
            <button
              type="button" key={option.id} role="radio" aria-checked={config.questionType === option.id}
              className={config.questionType === option.id ? "is-active" : ""}
              onClick={() => dispatch({ type: "config", patch: { questionType: option.id } })}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>
      <div className="exam-field-row">
        <label className="exam-field">
          <span className="exam-field-label">Questions</span>
          <input
            type="number" min={1} max={MAX_QUESTIONS} value={config.count}
            onChange={event => dispatch({ type: "config", patch: { count: Number(event.target.value) } })}
          />
        </label>
        <label className="exam-field is-grow">
          <span className="exam-field-label">Focus topic (optional)</span>
          <input type="text" value={config.topic} maxLength={300} placeholder="e.g. ZigBee device types" onChange={event => dispatch({ type: "config", patch: { topic: event.target.value } })} />
        </label>
      </div>
      {selection && (
        <label className="exam-checkbox">
          <input type="checkbox" checked={config.useSelection} onChange={event => dispatch({ type: "config", patch: { useSelection: event.target.checked } })} />
          Only use the selected passage (p. {selection.page})
        </label>
      )}
      {disabledReason && <p className="exam-hint">{disabledReason}</p>}
      <button type="submit" className="exam-primary" disabled={busy || Boolean(disabledReason)}>{busy ? "Generating…" : submitLabel}</button>
      {busy && <p className="exam-hint">Writing {config.count} grounded question{config.count > 1 ? "s" : ""} from your document — this can take a minute or two on a local model.</p>}
    </form>
  );
}

// ---------------------------------------------------------------------------
// Practice exam
// ---------------------------------------------------------------------------

function AnswerControls({ question, draft, evaluation, disabled, onChange }: {
  question: ExamQuestion; draft: string; evaluation?: AnswerEvaluation; disabled: boolean; onChange: (value: string) => void;
}) {
  if (question.type === "short_answer") {
    return (
      <textarea
        className="exam-short-answer" value={draft} onChange={event => onChange(event.target.value)} disabled={disabled || Boolean(evaluation)}
        placeholder="Write your answer in a few sentences…" rows={4} maxLength={4000} aria-label="Your answer"
      />
    );
  }
  const options = question.type === "mcq" ? question.options : ["True", "False"];
  const correct = correctAnswerText(question);
  return (
    <div className="exam-options" role="radiogroup" aria-label="Answer options">
      {options.map((option, index) => {
        const chosen = draft === option;
        const state = evaluation ? (option === correct ? "is-correct" : chosen ? "is-wrong" : "") : chosen ? "is-chosen" : "";
        return (
          <button
            type="button" key={option} role="radio" aria-checked={chosen} className={`exam-option ${state}`}
            onClick={() => onChange(option)} disabled={disabled || Boolean(evaluation)}
          >
            {question.type === "mcq" && <span className="exam-option-letter">{LETTERS[index]}</span>}
            <span>{option}</span>
          </button>
        );
      })}
    </div>
  );
}

function EvaluationPanel({ evaluation, onOpen }: { evaluation: AnswerEvaluation; onOpen: (citation: Citation) => void }) {
  return (
    <div className={`exam-feedback verdict-${evaluation.verdict}`} role="status">
      <p className="exam-verdict">
        {VERDICT_LABELS[evaluation.verdict]}
        <span>{evaluation.points.toFixed(evaluation.points % 1 ? 2 : 0)} / {evaluation.max_points} point</span>
      </p>
      <p>{evaluation.feedback}</p>
      {evaluation.question_type === "short_answer" && (
        <>
          <p className="exam-model-answer"><b>Model answer:</b> {evaluation.correct_answer}</p>
          {evaluation.key_points_covered.length > 0 && <ul className="key-points is-covered">{evaluation.key_points_covered.map(point => <li key={point}>{point}</li>)}</ul>}
          {evaluation.key_points_missing.length > 0 && <ul className="key-points is-missing">{evaluation.key_points_missing.map(point => <li key={point}>{point}</li>)}</ul>}
        </>
      )}
      {evaluation.explanation && <p className="exam-explanation"><b>Why:</b> {evaluation.explanation}</p>}
      <SourceList citations={evaluation.citations} onOpen={onOpen} />
    </div>
  );
}

function ExamRunner({ session, dispatch, busy, onSubmit, onOpen, onWritingFeedback }: {
  session: ExamSession; dispatch: Dispatch<StudyAction>; busy: boolean; onSubmit: () => void;
  onOpen: (citation: Citation) => void; onWritingFeedback: (question: string, answer: string) => void;
}) {
  const question = currentQuestion(session);
  if (!question) return null;
  const record = session.answers[question.id];
  const { progress } = session;
  const isLast = session.currentIndex === session.questions.length - 1;
  const canSubmit = !record && session.draft.trim().length > 0;
  return (
    <div className="exam-runner">
      <div className="exam-progress">
        <div className="exam-progress-meta">
          <span>Question {session.currentIndex + 1} of {session.questions.length}</span>
          <span>{progress.correct_answers} correct{progress.partial_answers ? ` · ${progress.partial_answers} partial` : ""} · {progress.answered_questions} answered</span>
        </div>
        <div className="exam-progress-bar" role="progressbar" aria-valuemin={0} aria-valuemax={progress.total_questions} aria-valuenow={progress.answered_questions}>
          <span style={{ width: `${(100 * progress.answered_questions) / progress.total_questions}%` }} />
        </div>
      </div>
      <article className="exam-card">
        <div className="exam-card-meta">
          <span className="exam-badge">{QUESTION_TYPE_LABELS[question.type]}</span>
          <GroundingBadge question={question} />
        </div>
        <p className="exam-question">{question.type === "true_false" ? `True or false: ${question.statement}` : question.question}</p>
        <AnswerControls question={question} draft={record ? record.selected : session.draft} evaluation={record?.evaluation} disabled={busy} onChange={value => dispatch({ type: "exam/draft", value })} />
        {record && <EvaluationPanel evaluation={record.evaluation} onOpen={onOpen} />}
        <div className="exam-card-actions">
          {!record && <button type="button" className="exam-primary" onClick={onSubmit} disabled={busy || !canSubmit}>{busy ? "Checking…" : "Submit answer"}</button>}
          {record && <button type="button" className="exam-primary" onClick={() => dispatch({ type: "exam/next" })}>{isLast ? "See results" : "Next question"}</button>}
          {record && question.type === "short_answer" && (
            <button type="button" className="exam-secondary" onClick={() => onWritingFeedback(question.question, record.selected)}>Get writing feedback</button>
          )}
          <button type="button" className="exam-link" onClick={() => dispatch({ type: "exam/finish" })}>Finish exam</button>
        </div>
      </article>
    </div>
  );
}

function ExamSummaryView({ session, dispatch, onOpen }: { session: ExamSession; dispatch: Dispatch<StudyAction>; onOpen: (citation: Citation) => void }) {
  const summary = examSummary(session);
  return (
    <div className="exam-summary">
      <div className="exam-score">
        <strong>{summary.score % 1 ? summary.score.toFixed(2) : summary.score} / {summary.total}</strong>
        <span>points · {summary.answered ? `${summary.percentage}% of answered questions` : "no questions answered"}</span>
      </div>
      <div className="exam-summary-counts">
        <span className="verdict-correct">{summary.correct} correct</span>
        <span className="verdict-partial">{summary.partial} partial</span>
        <span className="verdict-incorrect">{summary.incorrect} incorrect</span>
        {summary.unanswered > 0 && <span>{summary.unanswered} unanswered</span>}
      </div>
      <p className="exam-hint">
        Scoring: multiple-choice and true/false are 1 point each, graded exactly. A short answer is worth 1 point split evenly
        across its key points (judged by the local model against your document), so partial credit is possible. This session isn&apos;t saved.
      </p>
      <ol className="exam-review">
        {session.questions.map(question => {
          const record = session.answers[question.id];
          return (
            <li key={question.id} className={record ? `verdict-${record.evaluation.verdict}` : "is-unanswered"}>
              <p className="exam-review-question">{questionText(question)}</p>
              <p className="exam-review-answer">
                {record ? <>Your answer: <b>{record.selected}</b> · {VERDICT_LABELS[record.evaluation.verdict]}</> : "Not answered"}
              </p>
              {(!record || record.evaluation.verdict !== "correct") && <p className="exam-review-answer">Correct answer: <b>{correctAnswerText(question)}</b></p>}
              <SourceList citations={question.citations} onOpen={onOpen} />
            </li>
          );
        })}
      </ol>
      <div className="exam-card-actions">
        <button type="button" className="exam-primary" onClick={() => dispatch({ type: "exam/restart" })}>Restart exam</button>
        <button type="button" className="exam-secondary" onClick={() => dispatch({ type: "exam/discard" })}>New exam</button>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Question bank
// ---------------------------------------------------------------------------

function BankQuestion({ question, index, onOpen }: { question: ExamQuestion; index: number; onOpen: (citation: Citation) => void }) {
  const [revealed, setRevealed] = useState(false);
  return (
    <li className="bank-question">
      <div className="exam-card-meta">
        <span className="exam-badge">{index + 1} · {QUESTION_TYPE_LABELS[question.type]}</span>
        <GroundingBadge question={question} />
      </div>
      <p className="exam-question">{question.type === "true_false" ? `True or false: ${question.statement}` : question.question}</p>
      {question.type === "mcq" && (
        <ol className="bank-options">
          {question.options.map((option, optionIndex) => (
            <li key={option} className={revealed && option === question.answer ? "is-correct" : ""}><span>{LETTERS[optionIndex]}</span> {option}</li>
          ))}
        </ol>
      )}
      <button type="button" className="exam-link" onClick={() => setRevealed(current => !current)} aria-expanded={revealed}>{revealed ? "Hide answer" : "Show answer"}</button>
      {revealed && (
        <div className="bank-answer">
          <p><b>Answer:</b> {correctAnswerText(question)}</p>
          {question.type === "short_answer" && <ul className="key-points">{question.key_points.map(point => <li key={point}>{point}</li>)}</ul>}
          {question.explanation && <p className="exam-explanation"><b>Why:</b> {question.explanation}</p>}
          <SourceList citations={question.citations} onOpen={onOpen} />
        </div>
      )}
    </li>
  );
}

// ---------------------------------------------------------------------------
// Writing feedback
// ---------------------------------------------------------------------------

function WritingResult({ result, onOpen }: { result: WritingFeedbackResult; onOpen: (citation: Citation) => void }) {
  const { feedback } = result;
  return (
    <div className="writing-result">
      <section className="writing-block is-original">
        <p className="writing-block-label">Original answer</p>
        <p className="writing-original">{result.original_answer}</p>
      </section>
      {result.suggested_answer && (
        <section className="writing-block is-suggested">
          <p className="writing-block-label">Suggested / improved version <span>AI-generated · your original is unchanged</span></p>
          <MarkdownContent text={result.suggested_answer} />
        </section>
      )}
      <section className="writing-block is-feedback">
        <p className="writing-block-label">Feedback</p>
        {!feedback.structured && <p className="exam-hint">The model didn&apos;t return structured feedback this time — showing its response as-is.</p>}
        {feedback.summary && <p>{feedback.summary}</p>}
        {feedback.strengths.length > 0 && <><p className="writing-subhead">What works</p><ul>{feedback.strengths.map(item => <li key={item}>{item}</li>)}</ul></>}
        {feedback.improvements.length > 0 && <><p className="writing-subhead">What to improve</p><ul>{feedback.improvements.map(item => <li key={item}>{item}</li>)}</ul></>}
        {feedback.missing_points.length > 0 && (
          <><p className="writing-subhead">Missing points</p><ul>{feedback.missing_points.map(item => <li key={item.point}>{item.point} <span className="cite-markers">{item.source_ids.map(id => `[${id}]`).join(" ")}</span></li>)}</ul></>
        )}
        {feedback.factual_issues.length > 0 && (
          <><p className="writing-subhead">Factual issues</p><ul>{feedback.factual_issues.map(item => <li key={item.issue}>{item.issue} <span className="cite-markers">{item.source_ids.map(id => `[${id}]`).join(" ")}</span></li>)}</ul></>
        )}
        {feedback.structured && (
          <>
            <p className="writing-subhead">Rubric <span>qualitative, model-assessed — not a grade</span></p>
            <table className="rubric-table">
              <tbody>
                {feedback.rubric.map(entry => (
                  <tr key={entry.criterion}>
                    <th scope="row">{entry.label}</th>
                    <td><span className={`rubric-rating rating-${entry.rating}`}>{RATING_LABELS[entry.rating] ?? entry.rating}</span></td>
                    <td>{entry.comment}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {!result.grounded && <p className="exam-hint">No document text was available, so factual accuracy wasn&apos;t checked against your documents.</p>}
      </section>
      <SourceList citations={result.citations} onOpen={onOpen} label="Document source" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

export default function ExamPanel({ state, dispatch, scope, scopeLabel, selection, onOpenCitation }: Props) {
  const [busy, setBusy] = useState<"generate" | "grade" | "writing" | null>(null);
  const [error, setError] = useState("");
  const disabledReason = scope ? null : "Select at least one document to study from.";

  function requestScope(): LearningScope | null {
    if (!scope) return null;
    if (state.config.useSelection && selection) {
      return { ...scope, documentIds: [selection.documentId], collectionId: undefined, selectedText: selection.text, selectionPage: selection.page };
    }
    return scope;
  }

  async function generate(target: "exam" | "bank") {
    const effectiveScope = requestScope();
    if (!effectiveScope || busy) return;
    setBusy("generate"); setError("");
    try {
      const result = await generateQuestions(effectiveScope, {
        questionType: state.config.questionType, count: state.config.count, topic: state.config.topic.trim() || undefined,
        seed: Math.floor(Math.random() * 1_000_000),
      });
      const label = effectiveScope.selectedText ? `selected passage (p. ${effectiveScope.selectionPage})` : scopeLabel;
      dispatch(target === "exam"
        ? { type: "exam/start", questions: result.questions, warnings: result.warnings, scopeLabel: label }
        : { type: "bank/set", questions: result.questions, warnings: result.warnings, scopeLabel: label });
    } catch (err) { setError(err instanceof Error ? err.message : "Could not generate questions."); }
    finally { setBusy(null); }
  }

  async function submitAnswer() {
    const session = state.exam;
    const question = session ? currentQuestion(session) : null;
    if (!session || !question || busy) return;
    const answer = session.draft.trim();
    if (!answer) { setError("Choose or write an answer before submitting."); return; }
    setBusy("grade"); setError("");
    try {
      const result = await evaluateExamAnswer(question, answer, session.progress);
      dispatch({ type: "exam/answered", questionId: question.id, selected: answer, evaluation: result.evaluation, progress: result.progress ?? session.progress });
    } catch (err) { setError(err instanceof Error ? err.message : "Could not check that answer."); }
    finally { setBusy(null); }
  }

  async function requestWritingFeedback() {
    const effectiveScope = scope;
    const answer = state.writing.answer;
    if (!effectiveScope || busy) return;
    if (!answer.trim()) { setError("Write an answer before requesting feedback."); return; }
    setBusy("writing"); setError("");
    try {
      dispatch({ type: "writing/result", result: await evaluateWriting(effectiveScope, answer, state.writing.question.trim() || undefined) });
    } catch (err) { setError(err instanceof Error ? err.message : "Could not review that answer."); }
    finally { setBusy(null); }
  }

  const exam = state.exam;
  return (
    <div className="exam-panel">
      <div className="study-tabs" role="tablist" aria-label="Study tools">
        {VIEWS.map(view => (
          <button
            type="button" role="tab" key={view.id} aria-selected={state.view === view.id}
            className={state.view === view.id ? "is-active" : ""} onClick={() => { setError(""); dispatch({ type: "view", view: view.id }); }}
          >
            {view.label}
          </button>
        ))}
      </div>
      <div className="exam-body">
        {error && <div className="error exam-error" role="alert">{error}</div>}

        {state.view === "exam" && (
          <>
            {exam && exam.warnings.length > 0 && !exam.completed && exam.warnings.map(warning => <p key={warning} className="learning-notice is-warning">{warning}</p>)}
            {!exam && (
              <>
                <p className="exam-intro">Answer one question at a time and get feedback as you go. Progress stays in this browser tab only.</p>
                <SetupForm state={state} dispatch={dispatch} selection={selection} busy={busy === "generate"} disabledReason={disabledReason} scopeLabel={scopeLabel} submitLabel="Start exam" onSubmit={() => void generate("exam")} />
              </>
            )}
            {exam && !exam.completed && (
              <ExamRunner
                session={exam} dispatch={dispatch} busy={busy === "grade"} onSubmit={() => void submitAnswer()} onOpen={onOpenCitation}
                onWritingFeedback={(question, answer) => dispatch({ type: "writing/prefill", question, answer })}
              />
            )}
            {exam && exam.completed && <ExamSummaryView session={exam} dispatch={dispatch} onOpen={onOpenCitation} />}
          </>
        )}

        {state.view === "bank" && (
          <>
            <p className="exam-intro">Generate a set of questions with answers and sources to review, or practise them as an exam.</p>
            <SetupForm state={state} dispatch={dispatch} selection={selection} busy={busy === "generate"} disabledReason={disabledReason} scopeLabel={scopeLabel} submitLabel={state.bank ? "Generate new set" : "Generate questions"} onSubmit={() => void generate("bank")} />
            {state.bank && (
              <div className="bank-results">
                {state.bank.warnings.map(warning => <p key={warning} className="learning-notice is-warning">{warning}</p>)}
                <div className="bank-results-head">
                  <span>{state.bank.questions.length} question{state.bank.questions.length === 1 ? "" : "s"} · {state.bank.scopeLabel}</span>
                  <button type="button" className="exam-secondary" onClick={() => dispatch({ type: "bank/practice" })}>Practise as exam</button>
                </div>
                <ol className="bank-list">
                  {state.bank.questions.map((question, index) => <BankQuestion key={question.id} question={question} index={index} onOpen={onOpenCitation} />)}
                </ol>
              </div>
            )}
          </>
        )}

        {state.view === "writing" && (
          <form className="writing-form" onSubmit={event => { event.preventDefault(); void requestWritingFeedback(); }}>
            <p className="exam-intro">Paste or write an answer to get feedback against your document. Your original text is never changed.</p>
            <label className="exam-field">
              <span className="exam-field-label">Question you were answering (optional)</span>
              <input type="text" value={state.writing.question} maxLength={1000} onChange={event => dispatch({ type: "writing/update", patch: { question: event.target.value } })} placeholder="e.g. What is ZigBee and where is it used?" />
            </label>
            <label className="exam-field">
              <span className="exam-field-label">Your answer</span>
              <textarea value={state.writing.answer} rows={6} maxLength={8000} onChange={event => dispatch({ type: "writing/update", patch: { answer: event.target.value } })} placeholder="Write your answer or paragraph here…" />
            </label>
            {disabledReason && <p className="exam-hint">{disabledReason}</p>}
            <button type="submit" className="exam-primary" disabled={busy === "writing" || Boolean(disabledReason) || !state.writing.answer.trim()}>
              {busy === "writing" ? "Reviewing…" : "Get feedback"}
            </button>
            {state.writing.result && <WritingResult result={state.writing.result} onOpen={onOpenCitation} />}
          </form>
        )}
      </div>
    </div>
  );
}
