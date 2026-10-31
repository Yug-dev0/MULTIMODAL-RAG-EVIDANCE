import { useEffect, useRef, useState } from "react";

const MODALITIES = [
  {
    key: "image",
    label: "Image",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <rect x="3" y="4" width="18" height="16" rx="2" />
        <circle cx="9" cy="10" r="2" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 16l-5.5-5.5L9 17" />
      </svg>
    ),
  },
  {
    key: "speech",
    label: "Speech",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path strokeLinecap="round" d="M3 12h3l2.5-6 3 12L14 9l2 3h5" />
      </svg>
    ),
  },
  {
    key: "document",
    label: "Document",
    icon: (
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 2h9l5 5v15H6z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 2v5h5M9 13h6M9 17h6" />
      </svg>
    ),
  },
];

const BOOT_LINES = [
  { text: "> initializing evidence console", delay: 60, duration: 340 },
  { text: "> linking image · speech · document", delay: 480, duration: 380 },
  { text: "> ready", delay: 940, duration: 180 },
];

// Every sub-effect is keyed off one timeline so the sequence reads as one
// choreographed boot rather than independent animations firing at once.
const T = {
  icon: [160, 340, 520],
  beam: [480, 620, 760],
  ring: [1050, 1120, 1190],
  wordmark: 1200,
  scanBar: 1150,
  subtitle: 1750,
  finalSweep: 2250,
  skipHint: 1500,
  leave: 2600,
};
const WIPE_MS = 480;
const DONE_MS = T.leave + WIPE_MS;

interface Props {
  onDone: () => void;
}

export function HomeIntro({ onDone }: Props) {
  const [leaving, setLeaving] = useState(false);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const finished = useRef(false);

  useEffect(() => {
    timers.current.push(setTimeout(() => setLeaving(true), T.leave));
    timers.current.push(setTimeout(finish, DONE_MS));
    return () => timers.current.forEach(clearTimeout);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function finish() {
    if (finished.current) return;
    finished.current = true;
    onDone();
  }

  function skip() {
    timers.current.forEach(clearTimeout);
    setLeaving(true);
    timers.current = [setTimeout(finish, WIPE_MS)];
  }

  return (
    <div
      onClick={skip}
      className="absolute inset-0 z-40 flex cursor-pointer flex-col items-center justify-center overflow-hidden bg-void"
      style={{
        clipPath: leaving ? "circle(0% at 50% 50%)" : "circle(150% at 50% 50%)",
        opacity: leaving ? 0 : 1,
        transition: `clip-path ${WIPE_MS}ms cubic-bezier(.65,0,.35,1), opacity ${WIPE_MS}ms ease-in`,
      }}
    >
      {/* Ambient grid + glow */}
      <div
        className="pointer-events-none absolute inset-0 animate-gridFadeIn"
        style={{
          backgroundImage:
            "linear-gradient(rgb(var(--c-hairline) / 0.35) 1px, transparent 1px), linear-gradient(90deg, rgb(var(--c-hairline) / 0.35) 1px, transparent 1px)",
          backgroundSize: "38px 38px",
          maskImage: "radial-gradient(ellipse at center, black 0%, transparent 75%)",
        }}
        aria-hidden
      />
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse at center, rgb(var(--c-signal) / 0.1), transparent 60%)" }}
        aria-hidden
      />
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-r from-transparent via-signal/25 to-transparent animate-sweep"
        aria-hidden
      />
      {/* Confirm sweep, fires once near the end of the boot sequence */}
      <div
        className="pointer-events-none absolute inset-x-0 top-1/2 h-px bg-gradient-to-r from-transparent via-signal to-transparent animate-introFinalSweep"
        style={{ animationDelay: `${T.finalSweep}ms`, boxShadow: "0 0 12px 1px rgb(var(--c-signal) / 0.8)" }}
        aria-hidden
      />

      {/* Boot log */}
      <div className="absolute left-1/2 top-[28%] w-72 -translate-x-1/2 font-mono text-[11px] leading-relaxed text-dim sm:left-auto sm:right-[calc(50%+150px)] sm:translate-x-0 sm:text-right">
        {BOOT_LINES.map((line, i) => (
          <div
            key={i}
            className="animate-typewriter overflow-hidden whitespace-nowrap"
            style={{
              // @ts-expect-error -- custom property consumed by the typewriter keyframe
              "--w": `${line.text.length}ch`,
              animationDelay: `${line.delay}ms`,
              animationDuration: `${line.duration}ms`,
            }}
          >
            <span className="text-signal">{line.text.slice(0, 2)}</span>
            <span>{line.text.slice(2)}</span>
          </div>
        ))}
      </div>

      <div className="relative flex flex-col items-center">
        <div className="flex items-end gap-10">
          {MODALITIES.map((m, i) => (
            <div key={m.key} className="flex flex-col items-center">
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-xl border border-signal/30 bg-panel2/70 text-signal shadow-[0_0_24px_-6px_rgb(var(--c-signal))] ${
                  i % 2 === 0 ? "animate-introIconInL" : "animate-introIconInR"
                }`}
                style={{ animationDelay: `${T.icon[i]}ms` }}
              >
                {m.icon}
              </div>
              <div
                className="relative h-8 w-px origin-top animate-beamFlow"
                style={{
                  animationDelay: `${T.beam[i]}ms`,
                  backgroundImage:
                    "repeating-linear-gradient(180deg, rgb(var(--c-signal) / 0.95) 0px, rgb(var(--c-signal) / 0.95) 3px, transparent 3px, transparent 7px)",
                }}
              />
            </div>
          ))}
        </div>

        {/* Convergence burst */}
        <div className="relative -mt-1 flex h-4 w-4 items-center justify-center">
          {T.ring.map((delay, i) => (
            <span
              key={i}
              className="absolute inline-block h-full w-full animate-shockwave rounded-full border border-signal"
              style={{ animationDelay: `${delay}ms` }}
              aria-hidden
            />
          ))}
          <span
            className="relative inline-flex h-2.5 w-2.5 animate-introPop rounded-full bg-signal shadow-[0_0_16px_2px_rgb(var(--c-signal))]"
            style={{ animationDelay: `${T.ring[0]}ms` }}
          />
        </div>

        <div className="mt-8 text-center">
          <h1 className="relative inline-block font-sans text-6xl font-bold tracking-tight text-ink">
            <span className="relative inline-block animate-wordmarkWipe text-signal" style={{ animationDelay: `${T.wordmark}ms` }}>
              MME
              <span
                className="pointer-events-none absolute inset-y-0 left-0 w-full -skew-x-12 animate-scanBar bg-gradient-to-r from-transparent via-ink/70 to-transparent"
                style={{ animationDelay: `${T.scanBar}ms` }}
                aria-hidden
              />
            </span>
          </h1>
          <p
            className="mt-3 animate-introFadeUp font-mono text-xs uppercase tracking-[0.4em] text-dim"
            style={{ animationDelay: `${T.subtitle}ms` }}
          >
            Multimodal Evidence Console
          </p>
        </div>
      </div>

      <p
        className="pointer-events-none absolute bottom-6 animate-introFadeUp font-mono text-[10px] uppercase tracking-[0.3em] text-dim/70"
        style={{ animationDelay: `${T.skipHint}ms` }}
      >
        Click to skip
      </p>
    </div>
  );
}
