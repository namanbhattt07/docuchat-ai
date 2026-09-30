import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Structured answers (Group 3: bullets/tables/sectioned/summary formatting)
// need real Markdown rendering. Shared by chat answers (ChatPanel) and the
// Group 6 tutor/brainstorm/exam views. Only ever used for assistant-generated
// text -- a user's own typed message is rendered as plain text so a literal
// "#" or "|" never gets reinterpreted as Markdown.
export default function MarkdownContent({ text }: { text: string }) {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => <div className="md-table-wrap"><table>{children}</table></div>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
