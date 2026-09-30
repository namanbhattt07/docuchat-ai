type IconProps = { size?: number; className?: string };

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none" as const,
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export function ChevronLeftIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="m15 18-6-6 6-6" /></svg>;
}
export function ChevronRightIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="m9 18 6-6-6-6" /></svg>;
}
export function PanelLeftIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="M9.5 4v16" /></svg>;
}
export function PanelRightIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><rect x="3" y="4" width="18" height="16" rx="2.5" /><path d="M14.5 4v16" /></svg>;
}
export function ZoomInIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><circle cx="10.5" cy="10.5" r="6.5" /><path d="M21 21l-4.3-4.3M10.5 8v5M8 10.5h5" /></svg>;
}
export function ZoomOutIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><circle cx="10.5" cy="10.5" r="6.5" /><path d="M21 21l-4.3-4.3M8 10.5h5" /></svg>;
}
export function FitWidthIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><rect x="4" y="6" width="16" height="12" rx="1.5" /><path d="M8 3v3M16 3v3M8 18v3M16 18v3" /></svg>;
}
export function FullscreenIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M9 4H5a1 1 0 0 0-1 1v4M15 4h4a1 1 0 0 1 1 1v4M9 20H5a1 1 0 0 1-1-1v-4M15 20h4a1 1 0 0 0 1-1v-4" /></svg>;
}
export function FullscreenExitIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M4 9h4a1 1 0 0 0 1-1V4M20 9h-4a1 1 0 0 1-1-1V4M4 15h4a1 1 0 0 1 1 1v4M20 15h-4a1 1 0 0 0-1 1v4" /></svg>;
}
export function DownloadIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M12 3v12m0 0-4-4m4 4 4-4" /><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" /></svg>;
}
export function SearchIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /></svg>;
}
export function CloseIcon({ size = 14, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M18 6 6 18M6 6l12 12" /></svg>;
}
export function PlusIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M12 5v14M5 12h14" /></svg>;
}
export function TrashIcon({ size = 15, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m3 0-.8 12a2 2 0 0 1-2 1.8H8.8a2 2 0 0 1-2-1.8L6 7" /></svg>;
}
export function SendIcon({ size = 17, className }: IconProps) {
  return <svg {...base(size)} className={className} strokeWidth={2}><path d="M5 12h14M13 6l6 6-6 6" /></svg>;
}
export function SparkleIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className} fill="currentColor" stroke="none"><path d="M12 2.5c.5 3.2 1 5.4 2.2 6.6 1.2 1.2 3.4 1.7 6.6 2.2-3.2.5-5.4 1-6.6 2.2-1.2 1.2-1.7 3.4-2.2 6.6-.5-3.2-1-5.4-2.2-6.6-1.2-1.2-3.4-1.7-6.6-2.2 3.2-.5 5.4-1 6.6-2.2 1.2-1.2 1.7-3.4 2.2-6.6Z" /></svg>;
}
export function SunIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className}><circle cx="12" cy="12" r="4.2" /><path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4 6 18M18 6l1.4-1.4" /></svg>;
}
export function MoonIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z" /></svg>;
}
export function FileIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M7 3.5h7l3.5 3.5V19a1.5 1.5 0 0 1-1.5 1.5H7A1.5 1.5 0 0 1 5.5 19V5A1.5 1.5 0 0 1 7 3.5Z" /><path d="M14 3.5V7h3.5" /><path d="M8.5 12h7M8.5 15.5h5" /></svg>;
}
export function ExternalLinkIcon({ size = 14, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M9 6H6a1.5 1.5 0 0 0-1.5 1.5v10A1.5 1.5 0 0 0 6 19h10a1.5 1.5 0 0 0 1.5-1.5V15M14 4h6v6M20 4l-9 9" /></svg>;
}
export function ListIcon({ size = 18, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M9 6h11M9 12h11M9 18h11" /><path d="M4.5 6h.01M4.5 12h.01M4.5 18h.01" strokeWidth={2.4} /></svg>;
}
export function ChevronDownIcon({ size = 14, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="m6 9 6 6 6-6" /></svg>;
}
export function EyeIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" /><circle cx="12" cy="12" r="3" /></svg>;
}
export function EyeOffIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M3 3l18 18" /><path d="M10.6 5.7A10.4 10.4 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17.3 17.3 0 0 1-3.4 4.3M6.4 6.9C4 8.6 2.5 12 2.5 12S6 18.5 12 18.5a9.9 9.9 0 0 0 3.4-.6" /><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" /></svg>;
}
export function BookIcon({ size = 14, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H12v18H5.5A1.5 1.5 0 0 1 4 19.5v-15Z" /><path d="M12 3h6.5A1.5 1.5 0 0 1 20 4.5v15a1.5 1.5 0 0 1-1.5 1.5H12" /></svg>;
}
export function LockIcon({ size = 13, className }: IconProps) {
  return <svg {...base(size)} className={className}><rect x="5" y="10.5" width="14" height="9.5" rx="1.8" /><path d="M8 10.5V7a4 4 0 0 1 8 0v3.5" /></svg>;
}
export function GlobeIcon({ size = 13, className }: IconProps) {
  return <svg {...base(size)} className={className}><circle cx="12" cy="12" r="8.5" /><path d="M3.5 12h17M12 3.5c2.3 2.3 3.5 5.3 3.5 8.5s-1.2 6.2-3.5 8.5c-2.3-2.3-3.5-5.3-3.5-8.5s1.2-6.2 3.5-8.5Z" /></svg>;
}
export function FolderIcon({ size = 16, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M3.5 6.5A1.5 1.5 0 0 1 5 5h4.4l2 2.2H19a1.5 1.5 0 0 1 1.5 1.5v9.3A1.5 1.5 0 0 1 19 19.5H5A1.5 1.5 0 0 1 3.5 18V6.5Z" /></svg>;
}
export function ReplyIcon({ size = 13, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M9 8 4 12.5 9 17" /><path d="M4 12.5h9a6 6 0 0 1 6 6V20" /></svg>;
}
export function QuoteIcon({ size = 13, className }: IconProps) {
  return <svg {...base(size)} className={className} fill="currentColor" stroke="none"><path d="M4.5 7.5A3.5 3.5 0 0 1 8 4v2.2a1.3 1.3 0 0 0-1.3 1.3v.5H8V11H4.5V7.5Zm7.5 0A3.5 3.5 0 0 1 15.5 4v2.2a1.3 1.3 0 0 0-1.3 1.3v.5h1.3V11H12V7.5Z" /></svg>;
}
export function PencilIcon({ size = 13, className }: IconProps) {
  return <svg {...base(size)} className={className}><path d="M14.5 4.5 19.5 9.5M4 20l1-4.2L15.8 5A1.8 1.8 0 0 1 18.3 5l.7.7A1.8 1.8 0 0 1 19 8.2L8.2 19 4 20Z" /></svg>;
}
export function ImageIcon({ size = 15, className }: IconProps) {
  return <svg {...base(size)} className={className}><rect x="3" y="4" width="18" height="16" rx="2.5" /><circle cx="9" cy="10" r="1.6" /><path d="m21 16-5-5-8 8" /></svg>;
}
export function TemplateIcon({ size = 14, className }: IconProps) {
  return <svg {...base(size)} className={className}><rect x="4" y="3" width="16" height="18" rx="2.5" /><path d="M8 8h8M8 12h8M8 16h5" /></svg>;
}
