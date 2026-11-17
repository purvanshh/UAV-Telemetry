"use client";

import { FileCheck2, Upload, X } from "lucide-react";
import { useId, useRef, useState } from "react";

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** One obvious place to put a file: click, keyboard or drag and drop. The native input stays for accessibility. */
export function Dropzone({ accept, formats, file, onFile, disabled }: {
  accept: string;
  formats: string[];
  file: File | null;
  onFile: (f: File | null) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  return (
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
        if (!disabled && e.dataTransfer.files?.[0]) onFile(e.dataTransfer.files[0]);
      }}
    >
      <input
        ref={input}
        id={id}
        type="file"
        accept={accept}
        disabled={disabled}
        className="dz-input"
        aria-label="Choose a flight log"
        onChange={(e) => onFile(e.target.files?.[0] ?? null)}
      />
      {file ? (
        <div className="dz-file">
          <span className="dz-file-icon" aria-hidden="true">
            <FileCheck2 size={20} />
          </span>
          <div style={{ minWidth: 0 }}>
            <div className="dz-file-name">{file.name}</div>
            <div className="small muted">{formatSize(file.size)} · ready</div>
          </div>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={disabled}
            onClick={() => {
              if (input.current) input.current.value = "";
              onFile(null);
            }}
          >
            <X size={14} aria-hidden="true" /> Remove
          </button>
        </div>
      ) : (
        <div className="dz-empty">
          <span className="dz-icon" aria-hidden="true">
            <Upload size={22} />
          </span>
          <strong>Drop the flight log here</strong>
          <span className="small muted">or click to choose a file</span>
          <span className="row" style={{ justifyContent: "center", marginTop: "0.5rem" }}>
            {formats.map((f) => (
              <span key={f} className="tag">
                {f}
              </span>
            ))}
          </span>
        </div>
      )}
    </div>
  );
}
