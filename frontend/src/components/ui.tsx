"use client";

import Link from "next/link";
import { ChevronRight, type LucideIcon } from "lucide-react";
import { useId, useState, type ReactNode } from "react";
import { TONE_ICON, toneClass, type Tone } from "@/lib/status";

/* ------------------------------------------------------------------------------------------------ buttons */

type Variant = "primary" | "secondary" | "ghost" | "danger" | "danger-solid";
type Size = "sm" | "md" | "lg";

function btnClass(variant: Variant, size: Size, block?: boolean, iconOnly?: boolean) {
  return [
    "btn",
    variant === "secondary" ? "" : `btn-${variant}`,
    size === "md" ? "" : `btn-${size}`,
    block ? "btn-block" : "",
    iconOnly ? "btn-icon" : "",
  ]
    .filter(Boolean)
    .join(" ");
}

export function Button({
  variant = "secondary",
  size = "md",
  icon: Icon,
  loading,
  block,
  children,
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
  icon?: LucideIcon;
  loading?: boolean;
  block?: boolean;
}) {
  return (
    <button
      type="button"
      {...rest}
      disabled={rest.disabled || loading}
      aria-busy={loading || undefined}
      className={`${btnClass(variant, size, block, !children)} ${className ?? ""}`}
    >
      {loading ? <span className="spinner" aria-hidden="true" /> : Icon ? <Icon size={size === "sm" ? 14 : 16} aria-hidden="true" /> : null}
      {children}
    </button>
  );
}

/** A link styled as a button. External or download links use a plain anchor. */
export function ButtonLink({
  href,
  variant = "secondary",
  size = "md",
  icon: Icon,
  children,
  download,
  disabled,
  ...rest
}: {
  href: string;
  variant?: Variant;
  size?: Size;
  icon?: LucideIcon;
  children?: ReactNode;
  download?: boolean;
  disabled?: boolean;
  title?: string;
  "aria-label"?: string;
}) {
  const cls = btnClass(variant, size, false, !children);
  const inner = (
    <>
      {Icon && <Icon size={size === "sm" ? 14 : 16} aria-hidden="true" />}
      {children}
    </>
  );
  if (disabled) {
    return (
      <span className={cls} aria-disabled="true" {...rest}>
        {inner}
      </span>
    );
  }
  if (download || /^https?:/.test(href)) {
    return (
      <a className={cls} href={href} {...rest}>
        {inner}
      </a>
    );
  }
  return (
    <Link className={cls} href={href} {...rest}>
      {inner}
    </Link>
  );
}

/* ------------------------------------------------------------------------------------------------ status */

export function Pill({ tone = "neutral", icon, dot, children }: { tone?: Tone; icon?: boolean | LucideIcon; dot?: boolean; children: ReactNode }) {
  const I = icon === true ? TONE_ICON[tone] : icon || null;
  return (
    <span className={`pill ${toneClass(tone)}${dot ? " pill-dot" : ""}`}>
      {I && <I aria-hidden="true" />}
      {children}
    </span>
  );
}

export function StatusIcon({ tone, icon }: { tone: Tone; icon?: LucideIcon }) {
  const I = icon ?? TONE_ICON[tone];
  return (
    <span className={`status-icon ${toneClass(tone)}`} aria-hidden="true">
      <I />
    </span>
  );
}

/** The one verdict at the top of a page. */
export function Hero({ tone, title, detail, aside, icon }: { tone: Tone; title: string; detail?: ReactNode; aside?: ReactNode; icon?: LucideIcon }) {
  const I = icon ?? TONE_ICON[tone];
  return (
    <section className={`hero ${toneClass(tone)}`} role="status" aria-live="polite">
      <span className="hero-icon" aria-hidden="true">
        <I />
      </span>
      <div>
        <div className="hero-title">{title}</div>
        {detail && <div className="hero-detail">{detail}</div>}
      </div>
      {aside && <div className="hero-aside">{aside}</div>}
    </section>
  );
}

export function Alert({ tone = "info", title, children }: { tone?: Tone; title?: string; children?: ReactNode }) {
  const I = TONE_ICON[tone];
  return (
    <div className={`alert ${toneClass(tone)}`} role={tone === "warn" ? "alert" : "status"}>
      <I aria-hidden="true" />
      <div>
        {title && <strong>{title} </strong>}
        {children}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------------------------------------ layout */

export function PageHeader({ eyebrow, title, lede, actions }: { eyebrow?: string; title: ReactNode; lede?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-header">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {lede && <p className="lede">{lede}</p>}
      </div>
      {actions && <div className="row no-print">{actions}</div>}
    </header>
  );
}

export function Card({ title, icon: Icon, actions, children, pad = true, className, ...rest }: {
  title?: ReactNode;
  icon?: LucideIcon;
  actions?: ReactNode;
  children: ReactNode;
  pad?: boolean;
  className?: string;
  style?: React.CSSProperties;
  id?: string;
}) {
  return (
    <section className={`card ${className ?? ""}`} {...rest}>
      {title && (
        <div className="card-head">
          <h2>
            {Icon && <Icon size={17} aria-hidden="true" className="muted" />}
            {title}
          </h2>
          {actions}
        </div>
      )}
      {pad ? <div className="card-body">{children}</div> : children}
    </section>
  );
}

export function Empty({ icon: Icon, title, children, action }: { icon: LucideIcon; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty rise">
      <span className="empty-icon" aria-hidden="true">
        <Icon size={24} />
      </span>
      <h2>{title}</h2>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function Skeleton({ h = 16, w = "100%", r }: { h?: number; w?: number | string; r?: number }) {
  return <span className="skeleton" style={{ display: "block", height: h, width: w, borderRadius: r }} aria-hidden="true" />;
}

/* ------------------------------------------------------------------------------------------------ controls */

export function Segmented<T extends string>({ value, options, onChange, label }: {
  value: T;
  options: { value: T; label: string; icon?: LucideIcon }[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map(({ value: v, label: l, icon: I }) => (
        <button key={v} type="button" aria-pressed={value === v} onClick={() => onChange(v)}>
          {I && <I size={14} aria-hidden="true" />}
          {l}
        </button>
      ))}
    </div>
  );
}

export function Switch({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      className="switch"
      disabled={disabled}
      onClick={() => onChange(!checked)}
    />
  );
}

/** Animated open / close section. Controlled when `open` is given. */
export function Disclosure({ summary, children, open: controlled, onToggle, className, buttonClassName }: {
  summary: ReactNode;
  children: ReactNode;
  open?: boolean;
  onToggle?: (open: boolean) => void;
  className?: string;
  buttonClassName?: string;
}) {
  const [inner, setInner] = useState(false);
  const open = controlled ?? inner;
  const id = useId();
  return (
    <div className={className}>
      <button
        type="button"
        className={buttonClassName ?? "link-btn"}
        aria-expanded={open}
        aria-controls={id}
        onClick={() => (onToggle ? onToggle(!open) : setInner(!open))}
      >
        <ChevronRight className="chev" size={16} aria-hidden="true" />
        {summary}
      </button>
      <div id={id} className={`disclosure-body${open ? " is-open" : ""}`} inert={!open}>
        <div>{children}</div>
      </div>
    </div>
  );
}

export function Tabs<T extends string>({ tabs, value, onChange, label }: {
  tabs: { id: T; label: string; count?: number; icon?: LucideIcon }[];
  value: T;
  onChange: (t: T) => void;
  label: string;
}) {
  return (
    <div className="tabs" role="tablist" aria-label={label}>
      {tabs.map((t) => (
        <button key={t.id} role="tab" type="button" aria-selected={value === t.id} id={`tab-${t.id}`} onClick={() => onChange(t.id)}>
          {t.icon && <t.icon size={15} aria-hidden="true" />}
          {t.label}
          {t.count != null && <span className="tab-count">{t.count}</span>}
        </button>
      ))}
    </div>
  );
}
