/** The ops assistant panel (PRD §10 M9.3): a right-hand sheet where staff ask about invoices and
 * decisions. While the assistant works, each tool call appears as a step; the answer arrives
 * with inline citation chips that open the cited audit event or record. It can read and draft,
 * never approve or pay, and the panel says so up front. */

"use client";

import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUp, Check, Loader2, Square, X } from "lucide-react";
import { CitedText } from "@/components/assistant/citation-chip";
import { useAssistant, type AssistantMessage } from "@/components/assistant/assistant-context";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { getApiClient } from "@/lib/api/client";
import { ApiError } from "@/lib/api/contract";
import type { ChatReply, ChatStep } from "@/lib/api/types";
import { invoiceNumberOf } from "@/lib/assistant/suggestions";
import { cn } from "@/lib/utils";

const MAX_CHARS = 2000;

function Reply({ reply }: { reply: ChatReply }) {
  const hrefFor = (source: string, ref: string) =>
    reply.citations.find((c) => c.source === source && c.ref === ref)?.href ?? null;
  return (
    <div className={cn("text-sm leading-relaxed", reply.refusal && "text-muted-foreground")}>
      {reply.refusal && (
        <p className="text-2xs mb-1 font-medium tracking-wide uppercase">Declined</p>
      )}
      <p className="text-pretty">
        <CitedText text={reply.text} hrefFor={hrefFor} />
      </p>
      {reply.proposals.length > 0 && (
        <p className="text-muted-foreground mt-2 text-xs">
          {reply.proposals.length === 1 ? "1 draft" : `${reply.proposals.length} drafts`} waiting
          for a person with the right role.
        </p>
      )}
    </div>
  );
}

function Steps({ steps, working }: { steps: readonly ChatStep[]; working: boolean }) {
  return (
    <ol className="space-y-1.5" aria-label="What the assistant is doing">
      {steps.map((step, index) => (
        <li key={index} className="text-muted-foreground flex items-center gap-2 text-xs">
          {step.outcome === "ok" ? (
            <Check className="text-success size-3.5" aria-hidden />
          ) : (
            <X className="text-destructive size-3.5" aria-hidden />
          )}
          <span>
            {step.label}
            {step.outcome === "refused" && " (not allowed)"}
          </span>
        </li>
      ))}
      {working && (
        <li className="flex items-center gap-2 text-xs">
          <Loader2 className="text-accent size-3.5 animate-spin" aria-hidden />
          <span>{steps.length === 0 ? "Reading your question" : "Working"}</span>
        </li>
      )}
    </ol>
  );
}

export function AssistantPanel() {
  const { open, setOpen, messages, setMessages } = useAssistant();
  const [draft, setDraft] = useState("");
  const [steps, setSteps] = useState<readonly ChatStep[]>([]);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<readonly string[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const nextId = useRef(1);

  // Concrete starter questions about real invoices, fetched once the panel first opens.
  useEffect(() => {
    if (!open || suggestions.length > 0) return;
    let cancelled = false;
    const api = getApiClient();
    void Promise.all([
      api.listInvoices({ status: "blocked", limit: 1 }),
      api.listInvoices({ status: "awaiting_review", limit: 1 }),
    ])
      .then(async ([blocked, held]) => {
        const out: string[] = [];
        for (const [row, verb] of [
          [blocked[0], "blocked"],
          [held[0], "held for review"],
        ] as const) {
          if (!row) continue;
          const number = await invoiceNumberOf(api, row.id);
          out.push(`Why was ${number} ${verb}?`);
        }
        if (!cancelled) setSuggestions(out);
      })
      .catch(() => {
        // Suggestions are optional; the composer works without them.
      });
    return () => {
      cancelled = true;
    };
  }, [open, suggestions.length]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, steps, working]);

  useEffect(() => () => abortRef.current?.abort(), []);

  async function ask(question: string) {
    const text = question.trim();
    if (!text || working) return;
    const userMessage: AssistantMessage = { id: nextId.current++, role: "user", text };
    const history = [...messages, userMessage];
    setMessages(() => history);
    setDraft("");
    setSteps([]);
    setError(null);
    setWorking(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const reply = await getApiClient().askAssistant(
        history.map(({ role, text: body }) => ({ role, text: body })),
        (step) => setSteps((prev) => [...prev, step]),
        controller.signal,
      );
      setMessages((prev) => [
        ...prev,
        { id: nextId.current++, role: "assistant", text: reply.text, reply },
      ]);
    } catch (err: unknown) {
      if (!controller.signal.aborted) {
        setError(err instanceof ApiError ? err.message : "The assistant could not answer.");
      }
    } finally {
      setWorking(false);
      setSteps([]);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void ask(draft);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void ask(draft);
    }
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent
        side="right"
        className="w-full gap-0 p-0 data-[side=right]:sm:max-w-md"
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          inputRef.current?.focus();
        }}
      >
        <header className="border-b px-5 py-4 pe-12">
          <SheetTitle className="text-base font-semibold">Assistant</SheetTitle>
          <SheetDescription className="text-muted-foreground mt-0.5 text-xs text-pretty">
            Reads invoices and the audit trail, and cites what it finds. It can draft a request, but
            it can&apos;t approve or pay.
          </SheetDescription>
        </header>

        {/* Following a citation closes the panel so the cited record isn't hidden behind it;
            the thread is kept in the provider and returns with the panel. */}
        <div
          className="min-h-0 flex-1 overflow-y-auto px-5 py-4"
          aria-live="polite"
          onClick={(event) => {
            if ((event.target as Element).closest("a[href]")) setOpen(false);
          }}
        >
          {messages.length === 0 && !working ? (
            <div className="space-y-3">
              <p className="text-muted-foreground text-sm">
                Ask why an invoice was blocked or held, what a vendor changed, or what is waiting on
                a decision.
              </p>
              {suggestions.length > 0 && (
                <ul className="flex flex-col items-start gap-2">
                  {suggestions.map((suggestion) => (
                    <li key={suggestion}>
                      <button
                        type="button"
                        onClick={() => void ask(suggestion)}
                        className="border-border hover:border-accent/50 hover:bg-accent/5 focus-visible:ring-ring rounded-md border px-3 py-1.5 text-left text-sm transition-colors focus-visible:ring-2 focus-visible:outline-none"
                      >
                        {suggestion}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ) : (
            <ol className="space-y-5">
              {messages.map((message) => (
                <li
                  key={message.id}
                  className={cn(message.role === "user" && "flex justify-end")}
                  data-role={message.role}
                >
                  {message.role === "user" ? (
                    <p className="bg-muted max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap">
                      {message.text}
                    </p>
                  ) : message.reply ? (
                    <Reply reply={message.reply} />
                  ) : null}
                </li>
              ))}
              {working && (
                <li>
                  <Steps steps={steps} working />
                </li>
              )}
            </ol>
          )}
          {error && (
            <p role="alert" className="text-destructive mt-4 text-sm">
              {error}
            </p>
          )}
          <div ref={endRef} />
        </div>

        <form onSubmit={submit} className="border-t p-3">
          <label htmlFor="assistant-input" className="sr-only">
            Ask the assistant
          </label>
          <div className="border-input focus-within:ring-ring flex items-end gap-2 rounded-lg border px-3 py-2 focus-within:ring-2">
            <textarea
              id="assistant-input"
              ref={inputRef}
              value={draft}
              onChange={(event) => setDraft(event.target.value.slice(0, MAX_CHARS))}
              onKeyDown={onKeyDown}
              rows={2}
              placeholder="Ask about an invoice, vendor or decision"
              className="placeholder:text-muted-foreground max-h-40 min-h-10 flex-1 resize-none bg-transparent text-sm outline-none"
            />
            {working ? (
              <Button
                type="button"
                size="icon-sm"
                variant="outline"
                onClick={() => abortRef.current?.abort()}
                aria-label="Stop"
              >
                <Square className="size-3.5" aria-hidden />
              </Button>
            ) : (
              <Button type="submit" size="icon-sm" disabled={!draft.trim()} aria-label="Send">
                <ArrowUp className="size-4" aria-hidden />
              </Button>
            )}
          </div>
          <p className="text-muted-foreground mt-1.5 px-1 text-[0.6875rem]">
            Enter to send, Shift+Enter for a new line.
          </p>
        </form>
      </SheetContent>
    </Sheet>
  );
}
