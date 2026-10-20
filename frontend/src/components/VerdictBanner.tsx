import type { Verdict } from "@/lib/api";

/** The one-sentence answer. Colour is secondary: the headline and caveats carry the meaning. */
export function VerdictBanner({ verdict }: { verdict: Verdict }) {
  return (
    <section className={`verdict tone-${verdict.tone}`} aria-labelledby="verdict-h">
      <h2 id="verdict-h">{verdict.headline}</h2>
      <p style={{ margin: 0 }}>{verdict.detail}</p>
      {verdict.errors?.length ? (
        <ul>
          {verdict.errors.map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
      ) : null}
      {verdict.caveats?.length ? (
        <ul>
          {verdict.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}
