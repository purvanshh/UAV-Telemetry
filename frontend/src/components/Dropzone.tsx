"use client";

import { useId, useRef, useState } from "react";

/**
 * One obvious place to put a file. Works by click, keyboard or drag-and-drop and
 * always shows which file is chosen. The native input stays in the DOM so forms
 * and screen readers keep working.
 */
export function Dropzone({
  label,
  hint,
  accept,
  file,
  onFile,
  name,
  disabled,
}: {
  label: string;
  hint?: string;
  accept: string;
  file: File | null;
  onFile: (f: File | null) => void;
  name?: string;
  disabled?: boolean;
}) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  function pick(f: File | null | undefined) {
    onFile(f ?? null);
  }

  return (
    <div className="dz-wrap">
      <label className="dz-label" htmlFor={id}>
        {label}
      </label>
      <div
        className={`dropzone${over ? " is-over" : ""}${file ? " has-file" : ""}${disabled ? " is-disabled" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          if (!disabled) pick(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={input}
          id={id}
          name={name}
          type="file"
          accept={accept}
          disabled={disabled}
          className="dz-input"
          onChange={(e) => pick(e.target.files?.[0])}
        />
        {file ? (
          <div className="dz-file">
            <span className="dz-file-name">{file.name}</span>
            <span className="muted small">{formatSize(file.size)}</span>
            <button
              type="button"
              className="btn btn-quiet"
              onClick={() => {
                if (input.current) input.current.value = "";
                pick(null);
              }}
              disabled={disabled}
            >
              Remove
            </button>
          </div>
        ) : (
          <div className="dz-empty">
            <strong>Drop the file here</strong>
            <span className="muted small">or click to choose a file</span>
          </div>
        )}
      </div>
      {hint && <p className="muted small dz-hint">{hint}</p>}
    </div>
  );
}

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
