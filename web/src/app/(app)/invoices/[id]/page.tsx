/** Invoice detail: PDF pane with grounded field boxes, the field list and flags panel, and the
 * approve/fix/reject review action bar (PRD §10 M5.4). The whole screen doubles as the "review
 * queue" interaction for a single invoice. */

"use client";

import { use, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { PdfPane } from "@/components/invoice/pdf-pane";
import { FieldList, type FieldListHandle } from "@/components/invoice/field-list";
import { AuditTrail } from "@/components/invoice/audit-trail";
import { FlagsPanel } from "@/components/invoice/flags-panel";
import { ReviewActionBar, type ApproveError } from "@/components/invoice/review-action-bar";
import { RouteBadge, StatusBadge } from "@/components/inbox/status-badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { getApiClient } from "@/lib/api/client";
import { ApiError } from "@/lib/api/contract";
import { parseUnresolvedFields } from "@/lib/invoice-fields";
import { useAuth } from "@/lib/auth/auth-context";
import { formatAed } from "@/lib/format";
import { useSetPageHeader } from "@/lib/shell/page-header-context";
import type { FieldEdit, InvoiceDetail } from "@/lib/api/types";

export default function InvoiceDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const { user, session } = useAuth();
  const [invoice, setInvoice] = useState<InvoiceDetail | null>(null);
  const [activeField, setActiveField] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [approveError, setApproveError] = useState<ApproveError | null>(null);
  const fieldListRef = useRef<FieldListHandle>(null);
  // `?tab=audit&event=<id>` (assistant citations) opens the Audit tab on that event. A new link
  // to this same page switches tabs again: the param is compared during render (React's
  // "adjust state when a prop changes" pattern) rather than synced in an effect.
  const searchParams = useSearchParams();
  const tabParam = searchParams.get("tab");
  const eventParam = Number(searchParams.get("event"));
  const focusEvent = Number.isInteger(eventParam) && eventParam > 0 ? eventParam : null;
  const [tab, setTab] = useState(tabParam ?? "fields");
  const [seenTabParam, setSeenTabParam] = useState(tabParam);
  if (tabParam !== seenTabParam) {
    setSeenTabParam(tabParam);
    if (tabParam) setTab(tabParam);
  }

  const load = useCallback(() => {
    getApiClient()
      .getInvoice(id)
      .then(setInvoice)
      .catch(() => setInvoice(null));
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  useSetPageHeader(useMemo(() => ({ title: invoice ? invoice.summary.id : "Invoice" }), [invoice]));

  const boxes = useMemo(
    () =>
      (invoice?.fields ?? []).flatMap((field) =>
        field.boxes.map((box) => ({ fieldName: field.name, box, accepted: field.accepted })),
      ),
    [invoice],
  );

  async function handleEditField(edit: FieldEdit) {
    await getApiClient().editField(id, edit);
    setApproveError((prev) => {
      if (!prev) return prev;
      const left = prev.fields.filter((f) => f !== edit.field);
      return left.length > 0 ? { ...prev, fields: left } : null;
    });
    load();
  }

  async function handleDecision(action: "approve" | "reject") {
    if (!invoice) return;
    setPending(true);
    setApproveError(null);
    try {
      await getApiClient().decide(id, { action });
      toast.success(action === "approve" ? "Invoice approved" : "Invoice rejected");
      router.push("/review");
    } catch (error) {
      if (action === "approve" && error instanceof ApiError && error.status === 422) {
        // ADR-0012: the server names the classified fields still unresolved. Say so inline
        // (the action bar's live region) and put focus on the first one it names.
        const fields = parseUnresolvedFields(error.message);
        setApproveError({ detail: error.message, fields });
        setTab("fields");
        requestAnimationFrame(() => {
          const list = fieldListRef.current;
          if (!list) return;
          if (!fields.some((f) => list.focusField(f))) list.focusFirstNeedsReview();
        });
        return;
      }
      toast.error("Could not record the decision. Try again.");
    } finally {
      setPending(false);
    }
  }

  if (!invoice) {
    return (
      // Mirrors the loaded layout below (status bar, then the same responsive grid) so the
      // document pane and field column do not jump when the invoice arrives (CLS budget).
      <div className="flex h-full min-h-0 flex-col" role="status" aria-busy="true">
        <div className="border-border flex items-center gap-3 border-b px-6 py-3">
          <Skeleton className="h-5 w-24" />
          <Skeleton className="h-5 w-20" />
        </div>
        <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[1fr_380px]">
          {/* Both columns' contents are absolutely positioned so, like the loaded (scrolling)
              panes, they add no intrinsic height and the one-column mobile grid splits its rows
              exactly as it will once the invoice renders. */}
          <div className="relative min-h-0">
            <div className="absolute inset-0 flex flex-col">
              <div className="border-border h-[45px] shrink-0 border-b" />
              <div className="bg-muted/40 min-h-0 flex-1 p-6">
                <Skeleton className="h-full w-full" />
              </div>
            </div>
          </div>
          <div className="border-border relative min-h-0 border-t lg:border-t-0 lg:border-s">
            <div className="absolute inset-0 space-y-3 overflow-hidden p-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-14 w-full" />
              ))}
            </div>
          </div>
        </div>
        {/* Last child, so the header and grid keep the loaded tree's positions: React then reuses
            (not swaps) those nodes and the grid does not register as shifted. */}
        <span className="sr-only">Loading invoice…</span>
      </div>
    );
  }

  const isApproverRole = user?.roles.some((r) => r === "approver" || r === "admin") ?? false;
  const overLimit =
    user?.approvalLimitAed != null && invoice.summary.total != null
      ? Number(invoice.summary.total) > user.approvalLimitAed
      : false;
  const approveBlockedReason = !isApproverRole
    ? "Approving requires the approver role — reviewers can fix and reject."
    : overLimit
      ? `Above your AED ${user!.approvalLimitAed!.toLocaleString()} approval limit — needs a higher-limit approver.`
      : null;
  const decisionsClosed = !["awaiting_review", "validated", "fraud_checked", "gated"].includes(
    invoice.summary.status,
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-border flex items-center gap-3 border-b px-6 py-3">
        <StatusBadge status={invoice.summary.status} />
        <RouteBadge route={invoice.summary.route} />
        <span
          className="text-muted-foreground min-w-0 truncate text-sm"
          title={invoice.summary.vendorName ?? undefined}
        >
          {invoice.summary.vendorName ?? "Unknown vendor"}
        </span>
        <span className="tabular-mono ms-auto text-sm font-medium">
          {formatAed(invoice.summary.total)}
        </span>
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[1fr_380px]">
        <PdfPane
          documentUrl={invoice.documentUrl}
          authToken={session?.token ?? null}
          pageCount={invoice.pageCount}
          boxes={boxes}
          activeField={activeField}
          onSelectField={setActiveField}
        />
        <div className="border-border flex min-h-0 flex-col border-t lg:border-t-0 lg:border-s">
          <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col gap-0">
            <TabsList className="mx-4 mt-3 self-start">
              <TabsTrigger value="fields">Fields</TabsTrigger>
              <TabsTrigger value="flags">
                Flags{invoice.flags.length > 0 ? ` (${invoice.flags.length})` : ""}
              </TabsTrigger>
              <TabsTrigger value="audit">Audit</TabsTrigger>
            </TabsList>
            <TabsContent value="fields" className="mt-2 flex min-h-0 flex-1 flex-col">
              <FieldList
                ref={fieldListRef}
                fields={invoice.fields}
                activeField={activeField}
                onSelectField={setActiveField}
                unresolved={approveError?.fields}
                onEditField={handleEditField}
              />
            </TabsContent>
            <TabsContent value="flags" className="mt-2 min-h-0 flex-1 overflow-y-auto">
              <FlagsPanel flags={invoice.flags} />
            </TabsContent>
            <TabsContent value="audit" className="mt-2 min-h-0 flex-1 overflow-y-auto">
              <AuditTrail invoiceId={id} focusId={focusEvent} />
            </TabsContent>
          </Tabs>
          <ReviewActionBar
            disabled={decisionsClosed}
            approveBlockedReason={approveBlockedReason}
            pending={pending}
            approveError={approveError}
            onApprove={() => void handleDecision("approve")}
            onFix={() => fieldListRef.current?.focusFirstNeedsReview()}
            onReject={() => void handleDecision("reject")}
          />
        </div>
      </div>
    </div>
  );
}
