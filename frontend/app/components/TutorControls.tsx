import { TutorActionId } from "../../lib/api";
import { isActionAvailable, TUTOR_ACTIONS, TutorSessionState } from "../../lib/tutor";
import { CloseIcon, QuoteIcon } from "./icons";

type Props = {
  state: TutorSessionState;
  typedText: string;
  busy: boolean;
  disabled: boolean;
  onAction: (action: TutorActionId) => void;
  onClearTopic: () => void;
  onClearSelection: () => void;
};

// Group 6 TUTOR ACTIONS: one contextual toolbar (not ten separate flows) --
// every chip sends the same structured tutor request with a different
// tutor_action. Shows the session state the tutor is working from (topic,
// pinned selection, an open question) so "this" is never ambiguous.
export default function TutorControls({ state, typedText, busy, disabled, onAction, onClearTopic, onClearSelection }: Props) {
  const typed = typedText.trim();
  return (
    <div className="tutor-controls">
      <div className="tutor-context" aria-live="polite">
        {state.topic ? (
          <span className="tutor-chip is-topic" title="The tutor's current topic — actions apply to it">
            <b>Topic</b> <span>{state.topic}</span>
            <button type="button" onClick={onClearTopic} aria-label="Clear tutor topic"><CloseIcon size={10} /></button>
          </span>
        ) : (
          <span className="tutor-context-empty">No topic yet — ask about a concept, or pick an action to start from the document.</span>
        )}
        {state.selection && (
          <span className="tutor-chip is-selection" title={state.selection.text}>
            <QuoteIcon size={11} /> <span>Selected passage · p. {state.selection.page}</span>
            <button type="button" onClick={onClearSelection} aria-label="Clear selected passage"><CloseIcon size={10} /></button>
          </span>
        )}
        {state.pendingQuestion && <span className="tutor-chip is-waiting">Waiting for your answer</span>}
      </div>
      {typed && <p className="tutor-typed-hint">Actions will use “{typed.length > 40 ? `${typed.slice(0, 40)}…` : typed}” as the topic.</p>}
      <div className="tutor-actions" role="toolbar" aria-label="Tutor actions">
        {TUTOR_ACTIONS.map(action => {
          const available = isActionAvailable(action, state, typedText);
          return (
            <button
              type="button"
              key={action.id}
              onClick={() => onAction(action.id)}
              disabled={busy || disabled || !available}
              title={available ? action.title : `${action.title} — needs a topic or a selected passage first`}
            >
              {action.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
