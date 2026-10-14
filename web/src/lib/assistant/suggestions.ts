/** Starter questions for the assistant panel name real invoices by their printed
 * number, falling back to the short id when the number was not extracted. */
import type { ApiClient } from "@/lib/api/contract";

export async function invoiceNumberOf(api: ApiClient, id: string): Promise<string> {
  const invoice = await api.getInvoice(id);
  return invoice.fields.find((f) => f.name === "invoice_number")?.raw ?? id.slice(0, 12);
}
